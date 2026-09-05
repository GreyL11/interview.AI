"""Interviewer-pasted material, bound to the turn it belongs to.

Interviewers do not only speak. They paste a table, a failing query, a stack
trace, a screenshot -- and then ask about it, or ask first and paste after.
Either way the spoken question and the pasted material are one interview
turn, and the model has to receive both together, in the order they arrived,
with the pasted bytes intact.

Two rules shape everything here:

* **Exactness.** Pasted content is never summarised, reflowed or truncated.
  An attachment over the size cap is *rejected* with a reason rather than
  silently shortened, because a half table is worse than no table.
* **Binding, not broadcasting.** An attachment attaches to one turn. Once
  bound it stops being pending, so the next, unrelated question does not
  inherit it -- which is what would otherwise turn a stale paste into a
  permanent contaminant of the session.

This module owns only the buffer and its windows. Deciding *when* a turn is
complete stays entirely in `question_detector` / `LiveSession`; nothing here
participates in finality.
"""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import log_metric

logger = get_logger(__name__)


class AttachmentKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    SQL = "sql"
    TABLE = "table"
    IMAGE = "image"


class RejectReason(StrEnum):
    EMPTY = "empty"
    TOO_LARGE = "too_large"
    UNREADABLE_IMAGE = "unreadable_image"


class AttachmentError(Exception):
    """The attachment cannot be accepted. Carries a user-facing reason."""

    def __init__(self, reason: RejectReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


#: Fence language per kind, so the model is told what it is looking at without
#: anyone having to describe it in prose.
_FENCE = {
    AttachmentKind.CODE: "",
    AttachmentKind.SQL: "sql",
    AttachmentKind.TABLE: "",
    AttachmentKind.TEXT: "",
    AttachmentKind.IMAGE: "",
}


@dataclass(frozen=True)
class Attachment:
    """One piece of pasted material, exactly as it arrived."""

    kind: AttachmentKind
    content: str
    name: str = ""
    #: Monotonic arrival time. Drives the attachment window and, more
    #: importantly, the order material is presented in -- an interviewer who
    #: pastes a schema then a query means those in that order.
    at: float = 0.0
    #: Set when the content is OCR text recovered from an image rather than
    #: something the interviewer typed. Surfaced to the model, because
    #: recognised text can be wrong in ways pasted text cannot.
    from_image: bool = False

    def as_context(self) -> str:
        """Render for the prompt. Byte-for-byte content, fenced and labelled."""
        label = self.kind.value.upper()
        if self.name:
            label = f"{label} — {self.name}"
        if self.from_image:
            label = f"{label} (text recognised from a pasted image; may contain OCR errors)"
        fence = _FENCE.get(self.kind, "")
        return f"[{label}]\n```{fence}\n{self.content}\n```"


@dataclass
class AttachmentBuffer:
    """Pending and turn-bound attachments for one live session.

    `pending` holds material that has arrived but has no turn yet. `bound`
    holds what the current turn is carrying, which is what lets a follow-up
    ("and what does row 3 mean?") still see the table without the paste
    leaking onto a later, unrelated question.
    """

    pending: list[Attachment] = field(default_factory=list)
    bound: list[Attachment] = field(default_factory=list)

    # ------------------------------------------------------------- ingress

    def add(self, attachment: Attachment) -> None:
        self.pending.append(attachment)
        # Oldest first out. The cap exists so a session cannot accumulate
        # unbounded material; the window below usually evicts long before it.
        overflow = len(self.pending) - settings.context_attachment_max_items
        if overflow > 0:
            dropped = self.pending[:overflow]
            del self.pending[:overflow]
            log_metric(
                "attachment_evicted",
                reason="max_items",
                dropped=len(dropped),
                kept=len(self.pending),
            )

    # ------------------------------------------------------------- windows

    def _fresh(self, now: float) -> list[Attachment]:
        window = settings.context_attachment_window_ms / 1000
        return [a for a in self.pending if 0 <= now - a.at <= window]

    def expire(self, now: float) -> int:
        """Drop pending material too old to belong to anything. Returns count."""
        window = settings.context_attachment_window_ms / 1000
        stale = [a for a in self.pending if now - a.at > window]
        if stale:
            self.pending = [a for a in self.pending if a not in stale]
            log_metric("attachment_expired", dropped=len(stale))
        return len(stale)

    def peek(self, now: float, *, ignore_age: bool = False) -> list[Attachment]:
        """What would attach to a turn asked right now, oldest first.

        `ignore_age=True` drops the window and offers everything unclaimed.
        The window is only ever a proxy for ownership -- it guesses that a
        paste close in time to a question belongs to it. A turn that
        demonstrably refers to provided material is *direct* evidence of the
        same thing, and beats the proxy, so that path asks for all of it.
        """
        return sorted(
            self.pending if ignore_age else self._fresh(now), key=lambda a: a.at
        )

    @property
    def unclaimed(self) -> list[Attachment]:
        """Everything that has arrived and no turn has taken, oldest first.

        Deliberately age-blind, and metadata is all any caller gets from it in
        practice: this is what lets the classifier be told that material
        exists which the current turn did not bind, so it can say whether the
        question refers to it. Without that the question could never be
        judged against material the window had already excluded.
        """
        return sorted(self.pending, key=lambda a: a.at)

    def bind(
        self, now: float, extend: bool = False, *, ignore_age: bool = False
    ) -> list[Attachment]:
        """Attach pending material to the turn being asked now.

        Consuming rather than copying is the whole point: a bound attachment
        is no longer pending, so the *next* question cannot pick it up again
        and answer with material the interviewer has moved on from.

        `extend=False` starts a fresh turn and therefore *drops* whatever the
        previous turn carried -- without that, one pasted table would ride
        along on every question for the rest of the session. `extend=True` is
        the same turn being re-asked (a late paste, a correction), a follow-up
        to it, or a semantic claim -- all of which add to what the turn
        already has and must not duplicate it.
        """
        if not extend:
            self.bound.clear()
        taking = self.peek(now, ignore_age=ignore_age)
        if not ignore_age:
            self._report_unbound(now, taking)
        if taking:
            taken = {id(a) for a in taking}
            self.pending = [a for a in self.pending if id(a) not in taken]
            existing = {(a.kind, a.content) for a in self.bound}
            self.bound.extend(
                a for a in taking if (a.kind, a.content) not in existing
            )
        return list(self.bound)

    def _report_unbound(self, now: float, taking: Sequence[Attachment]) -> None:
        """Say so when material was here and this turn did not take it.

        The silence this closes was the whole reason the original failure
        could not be diagnosed from logs: material outside the window is
        skipped by `peek` and stays pending, so nothing is dropped, nothing
        is expired, and no line is written -- while the turn that should have
        carried it produces an answer with no trace of what was missing.

        Only fires when material was actually left behind, which is rare, so
        this is not a per-turn log line.
        """
        taken = {id(a) for a in taking}
        skipped = [a for a in self.pending if id(a) not in taken]
        if not skipped:
            return
        log_metric(
            "attachment_unbound",
            reason="outside_window",
            count=len(skipped),
            # Oldest first, since that is the one closest to being abandoned.
            # Negative would mean a paste that arrived after the question was
            # spoken, which the window also excludes.
            age_ms=int(max(now - a.at for a in skipped) * 1000),
        )

    def release(self) -> None:
        """The turn is over and its material is no longer in play."""
        self.bound.clear()

    @property
    def has_pending(self) -> bool:
        return bool(self.pending)


# ------------------------------------------------------------------ building


def build_attachment(
    kind: str,
    content: str,
    name: str = "",
    now: float | None = None,
    image_bytes: bytes | None = None,
) -> Attachment:
    """Validate and normalise one pasted item.

    Raises `AttachmentError` rather than trimming: the caller reports the
    reason to the interviewer, who can paste less. Silently truncating a
    table or a query would hand the model something that looks complete and
    is not.
    """
    try:
        parsed = AttachmentKind(kind)
    except ValueError:
        parsed = AttachmentKind.TEXT

    from_image = False
    if parsed is AttachmentKind.IMAGE:
        content = _read_image_text(image_bytes)
        from_image = True

    # Only the surrounding whitespace goes; interior formatting is content.
    # A table's alignment and a snippet's indentation are load-bearing.
    content = content.strip("\n\r")
    if not content.strip():
        raise AttachmentError(
            RejectReason.EMPTY, "That attachment had no readable content."
        )
    if len(content) > settings.context_attachment_max_chars:
        raise AttachmentError(
            RejectReason.TOO_LARGE,
            f"That attachment is {len(content):,} characters, over the "
            f"{settings.context_attachment_max_chars:,} limit. Paste the "
            f"relevant part instead — it is not shortened automatically, "
            f"because a partial table or query is worse than none.",
        )

    return Attachment(
        kind=parsed,
        content=content,
        name=name.strip()[:120],
        at=now if now is not None else time.monotonic(),
        from_image=from_image,
    )


def _read_image_text(image_bytes: bytes | None) -> str:
    """Recover text from a pasted screenshot.

    The configured model is text-only, so an image can only ever reach it as
    text -- there is no path where the picture itself is understood. OCR is
    therefore not an enhancement here, it is the only option, and a failure
    to read one has to be said out loud rather than silently attaching
    nothing.
    """
    if not image_bytes:
        raise AttachmentError(
            RejectReason.EMPTY, "That image was empty."
        )

    from app.documents.ocr import OcrUnavailable, ocr_available, ocr_image_bytes

    if not ocr_available():
        raise AttachmentError(
            RejectReason.UNREADABLE_IMAGE,
            "This installation cannot read text out of images. Paste the text "
            "itself and it will be used exactly as given.",
        )
    try:
        text = ocr_image_bytes(image_bytes)
    except OcrUnavailable as exc:
        raise AttachmentError(RejectReason.UNREADABLE_IMAGE, str(exc)) from exc
    except Exception as exc:
        logger.warning("attachment_ocr_failed error=%s", type(exc).__name__)
        raise AttachmentError(
            RejectReason.UNREADABLE_IMAGE,
            "No text could be read from that image. Paste the text instead.",
        ) from exc

    if not text.strip():
        raise AttachmentError(
            RejectReason.UNREADABLE_IMAGE,
            "No text could be read from that image. Paste the text instead.",
        )
    return text


def render(attachments: Sequence[Attachment] | Iterable[Attachment]) -> list[str]:
    """Prompt-ready blocks, in arrival order."""
    return [a.as_context() for a in sorted(attachments, key=lambda a: a.at)]


def summarise(attachments: Sequence[Attachment] | Iterable[Attachment]) -> list[str]:
    """Describe the material without including it.

    For the question-understanding classifier, which needs to know that a
    table exists and roughly how big it is -- enough to judge whether the
    question refers to it -- but has no use for the bytes. Sending a 20,000
    character schema there would add latency on the realtime path and put
    pasted content into a second prompt for no gain.
    """
    return [
        f"{a.kind.value}"
        + (f" named {a.name!r}" if a.name else "")
        + f", {len(a.content)} characters"
        + (" (text recognised from an image)" if a.from_image else "")
        for a in sorted(attachments, key=lambda a: a.at)
    ]
