"""Deterministic detection of interviewer prompts in a finalised transcript.

Why this exists
---------------
The phase 1 classifier decides *what kind* of question something is, and it was
built for text a user typed. Live interview speech is different in two ways that
broke it:

1. Whisper routinely omits question marks, so "what is the difference between
   shallow copy and deep copy" arrives with a full stop.
2. Interviewers do not speak in isolated questions. A single VAD utterance is
   usually acknowledgement plus filler plus the actual prompt:
   "As you said, four. Very good. So tell me what is the difference..."

The old check was anchored to the start of the whole utterance and otherwise
required a trailing "?", so a question in the last sentence was invisible.

This module works sentence by sentence instead, and is the single place that
answers "is the interviewer asking me something?". No LLM call: this runs on
every finalised utterance during a live interview, so it has to be fast and
predictable.
"""

import re
from dataclasses import dataclass

# --------------------------------------------------------------------- reasons

REASON_PUNCTUATION = "punctuation"
REASON_INTERROGATIVE = "interrogative"
REASON_INTERVIEW_PROMPT = "interview_prompt"
REASON_IMPERATIVE_TASK = "imperative_task"
REASON_NO_PATTERN = "no_question_pattern"


@dataclass(frozen=True)
class PromptMatch:
    """A detected interviewer prompt.

    `prompt` is what should be sent for coaching: everything from the start of
    the matching sentence to the end of the utterance, so a trailing correction
    ("...actually, assume 10k QPS") is not thrown away.
    """

    prompt: str
    reason: str
    sentence: str


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Acknowledgements that make up an entire sentence. Matched whole, so
# "Very good" is filler but "Very good, now tell me..." is not.
_ACK_ONLY = re.compile(
    r"""^\s*
    (?:(?:o?k(?:ay)?|alright|all\s+right|well|so|now|and|but|yeah|yep|right)[\s,]+)?
    (?:
        (?:very\s+|really\s+|pretty\s+|so\s+)?
        (?:good|great|nice|perfect|excellent|awesome|cool|fine|lovely|brilliant|superb)
      | (?:great|good|nice|excellent|perfect)\s+(?:answer|point|job|work|stuff)
      | well\s+done
      | (?:that(?:'s|s| is)\s+)?(?:correct|right|interesting|helpful|clear|fair|true|fine)
      | (?:o?k(?:ay)?|alright|all\s+right|sure|yeah|yep|yes|no|nope|hmm+|mm+[\s-]*hmm+|uh[\s-]*huh|right)
      | thank(?:s|\s+you)(?:\s+very\s+much)?
      | i\s+(?:see|understand|agree)
      | understood | got\s+it | makes\s+sense | sounds\s+good | fair\s+enough
      | no\s+problem | of\s+course | absolutely | exactly | indeed
      | let(?:'s|s|\s+us)\s+(?:continue|move\s+on|proceed|go\s+on|carry\s+on)
      | moving\s+on | next | continue | done | good\s+to\s+know
    )
    [\s.,!?]*$""",
    re.IGNORECASE | re.VERBOSE,
)

# LAYER 3 — interview prompt phrases. Safe to match anywhere in a sentence:
# these are multi-word or distinctive enough that an incidental occurrence is
# unlikely, which is what lets "So tell me what is..." be found mid-utterance.
_PROMPT_ANYWHERE = re.compile(
    r"""\b(?:
        tell\s+(?:me|us)
      | let\s+me\s+ask
      | (?:walk|take|talk)\s+(?:me|us)\s+through
      | walk\s+through
      | talk\s+about
      | tell\s+about
      | explain
      | describe
      | elaborate(?:\s+on)?
      | compare
      | contrast
      | define
      | outline
      | summari[sz]e
      | illustrate
      | demonstrate
      | give\s+(?:me|us)\s+(?:an?\s+)?(?:example|overview|idea|sense)
      | give\s+(?:me|us)\s+a\s+(?:brief|quick|short|high[\s-]level)
      | share\s+(?:an?\s+)?(?:example|experience|story)
      | help\s+(?:me|us)\s+understand
      | show\s+(?:me|us)
      | quick\s+question
      | (?:another|next|final|last|one\s+more)\s+question
      | i(?:'d|\s+would)\s+like\s+to\s+(?:know|hear|understand)
      | i\s+wan(?:t|na)\s+to\s+(?:know|hear|understand)
      | curious\s+(?:about|how|why|what)
      | your\s+thoughts\s+on
      | let'?s\s+(?:do|try|build|use|instead\s+do)
      | how\s+about
      | what\s+about\s+(?:doing|using|trying)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# LAYER 2 — interrogatives, and LAYER 3b — task imperatives.
#
# These are only trusted at the start of a clause. "I understand how you feel"
# contains "how", and a coding verb like "write" appears in ordinary speech
# ("how much you write yourself"); requiring clause-initial position is what
# keeps those from firing.
# No leading "^": this is used with .match(sentence, offset), which already
# anchors at the offset, whereas "^" would keep anchoring to the start of the
# whole sentence and silently ignore every clause boundary after the first.
_CLAUSE_INITIAL = re.compile(
    r"""(?:
        (?P<interrogative>
            (?:what|what's|whats|why|when|where|who|whom|whose|which|how)\b
          | (?:can|could|would|will|do|did|does|have|has|had|are|is|was|were|should|shall|may|might)
            \s+(?:you|we|i|it|there|they|he|she|this|that|your|any)\b
        )
      | (?P<imperative>
            (?:write|implement|code|build|design|create|solve|reverse|sort|return|find|given|
               list|name|discuss)\b
        )
        # Deliberately absent: assume, suppose, imagine, consider, take, walk.
        # Those are scenario qualifiers that accompany a question rather than
        # being one ("...actually, assume 10k QPS"), and treating them as
        # prompts made the qualifier outrank the question it was modifying.
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Where a clause can start inside a sentence: after punctuation, or after a
# discourse connective. This is what makes "Okay, what is a closure" work when
# Whisper gives no sentence punctuation at all.
_CLAUSE_BOUNDARY = re.compile(
    r"""(?:[,;:]\s*
        | \b(?:so|now|then|and|but|also|okay|ok|alright|well|next|finally|lastly|
               actually|anyway|however|plus)\b[\s,]*
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip())]
    return [part for part in parts if part]


def _clause_starts(sentence: str) -> list[int]:
    offsets = [0]
    offsets.extend(match.end() for match in _CLAUSE_BOUNDARY.finditer(sentence))
    return offsets


def _classify_sentence(sentence: str) -> str | None:
    """Which layer, if any, marks this sentence as a prompt."""
    if _ACK_ONLY.match(sentence):
        # Pure acknowledgement. Rejected even with a trailing "?" so a tag
        # question like "Right?" does not trigger coaching.
        return None

    if sentence.rstrip().endswith("?"):
        return REASON_PUNCTUATION

    if _PROMPT_ANYWHERE.search(sentence):
        return REASON_INTERVIEW_PROMPT

    for offset in _clause_starts(sentence):
        match = _CLAUSE_INITIAL.match(sentence, offset)
        if match is None:
            continue
        return (
            REASON_INTERROGATIVE
            if match.group("interrogative")
            else REASON_IMPERATIVE_TASK
        )

    return None


def _normalize(text: str) -> str:
    cleaned = text.strip().rstrip(".,;: \t")
    if not cleaned:
        return cleaned
    if cleaned.endswith("?"):
        return cleaned
    return f"{cleaned.rstrip('!')}?"


def extract_interview_prompt(text: str) -> PromptMatch | None:
    """Find the interviewer's prompt in a finalised transcript.

    Scans every sentence and keeps the *last* one that looks like a prompt: an
    interviewer who acknowledges the previous answer and then asks something new
    has put the live question at the end. Everything from that sentence onward is
    returned, so trailing qualifiers survive.
    """
    if not text or not text.strip():
        return None

    sentences = _sentences(text)
    if not sentences:
        return None

    matched_index: int | None = None
    matched_reason: str | None = None
    for index, sentence in enumerate(sentences):
        reason = _classify_sentence(sentence)
        if reason is not None:
            matched_index, matched_reason = index, reason

    if matched_index is None or matched_reason is None:
        return None

    tail = " ".join(sentences[matched_index:])
    return PromptMatch(
        prompt=_normalize(tail),
        reason=matched_reason,
        sentence=sentences[matched_index],
    )


def is_interview_prompt(text: str) -> bool:
    """True when the transcript contains something worth answering."""
    return extract_interview_prompt(text) is not None
