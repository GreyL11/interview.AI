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
#: A self-revision of a question already answered. Not produced by
#: pattern extraction -- `QuestionDetector` assigns it, because whether a
#: correction is answerable depends on conversation state this module
#: deliberately has no access to.
REASON_CORRECTION = "correction"


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
    #: True when the matched trigger phrase ("explain", "tell me about") is
    #: essentially all the interviewer has said -- nothing meaningful follows
    #: it, and nothing follows this sentence either. That is the signature of
    #: a prompt caught mid-utterance ("Can you explain") rather than a
    #: complete one ("Can you explain closures?"). See `_has_content_after`.
    sparse: bool = False
    #: Sentences of this same utterance that came *before* the prompt and are
    #: not filler -- the premise a question is asked against:
    #:
    #:     "We have ten million rows in the orders table.
    #:      How would you speed up this join?"
    #:
    #: The scanner keeps the last prompt-like sentence, which is right for
    #: dropping an acknowledgement ("Very good. So tell me...") and wrong for
    #: a load-bearing premise -- without this the model was asked to speed up
    #: an unspecified join on a table of unknown size. Kept separate from
    #: `prompt` so the coaching panel still shows the question the interviewer
    #: asked rather than the whole paragraph; the caller merges it into
    #: `Detection.effective_text`, the field that already exists for exactly
    #: this (premises spread across *separate* utterances).
    premise: str = ""


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: One actual word, so stray STT punctuation is not mistaken for a premise.
_WORDS = re.compile(r"[A-Za-z']")

#: Cap on the premise carried into the prompt. Generous next to a question but
#: firmly bounded: an interviewer who monologues for a paragraph before asking
#: should contribute the part nearest the question, not an unbounded prefix.
#: Tail-anchored for the same reason -- the sentence just before the question
#: is the one most likely to hold its constraint.
_MAX_PREMISE_CHARS = 400

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
      | understood | got\s+it | (?:that\s+)?makes\s+sense | sounds\s+good | fair\s+enough
      | no\s+problem | of\s+course | absolutely | exactly | indeed
      | let(?:'s|s|\s+us)\s+(?:continue|move\s+on|proceed|go\s+on|carry\s+on)
      | moving\s+on | next | continue | done | good\s+to\s+know
    )
    [\s.,!?]*$""",
    re.IGNORECASE | re.VERBOSE,
)

def is_acknowledgement(text: str) -> bool:
    """True if the whole utterance is filler -- "Okay.", "That makes sense."

    Exposed because the merge step needs it: folding an acknowledgement onto
    the question it follows re-asks that question and bills a second answer
    for it.
    """
    return bool(_ACK_ONLY.match(text.strip()))


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
      # "now give me the implementation" -- an ordinary progression step in a
      # coding interview -- matched none of the above and was dropped, because
      # the permitted objects were named individually and no such list can be
      # complete. Keyed on the determiner instead: a request for *content*
      # takes a definite one ("give me the implementation", "give me your
      # approach"), while an interviewer buying time takes an indefinite one
      # ("give me a second", "give me one moment"). Widening the verb itself
      # accepted those and answered them, which is worse than missing them.
      | give\s+(?:me|us)\s+(?:the|your)\b
      | share\s+(?:an?\s+)?(?:example|experience|story)
      | help\s+(?:me|us)\s+understand
      | show\s+(?:me|us)
      | quick\s+question
      | (?:another|next|final|last|one\s+more)\s+question
      | i(?:'d|\s+would)\s+like\s+to\s+(?:know|hear|understand)
      | i\s+wan(?:t|na)\s+to\s+(?:know|hear|understand)
      | curious\s+(?:about|how|why|what)
      | your\s+thoughts\s+on
      # A "let's ..." topic switch starts a new prompt, so the previous
      # problem is not carried into it. Measured gap: "let's solve X" and
      # "let's switch to X" were merging with the question they replaced.
      | let'?s\s+(?:do|try|build|use|solve|tackle|instead\s+do|
                   switch\s+to|move\s+(?:on\s+)?to|go\s+with|look\s+at|work\s+on|jump\s+to)
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
               list|name|discuss|rank|count|group|select|calculate|compute|
        # Verbs an interviewer uses to continue an existing task rather than
        # open a new one ("Now handle a stream of integers."). Without these
        # the continuation was rejected outright and the step was lost, even
        # though the turn is plainly an instruction. `optimi[sz]e` follows the
        # same both-spellings convention as `summari[sz]e` above.
               handle|optimi[sz]e|refactor|extend|modify|
               # A bare request to respond, which an interviewer uses with a
               # reference rather than a subject: "Answer this.", "Answer the
               # second one." Clause-initial anchoring keeps it away from
               # ordinary speech ("I would answer that with...").
               answer)\b
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


#: Words that never count as "the interviewer went on to say something" when
#: they're all that's left after a trigger phrase -- pronouns and auxiliaries
#: that any unfinished clause is equally likely to open with ("How would
#: you...", "Tell me about..."). Deliberately not `_DANGLING_WORDS` from
#: question_detector: that set flags a sentence by its *last* word anywhere,
#: this one only judges the tail immediately after a matched trigger.
_TRIVIAL_TAIL_WORDS = frozenset({
    "me", "us", "you", "it", "this", "that", "i", "we", "they", "he", "she",
    "can", "could", "would", "will", "do", "does", "did",
    "is", "are", "was", "were", "about", "to", "of",
})

_WORD = re.compile(r"[A-Za-z']+")


def _has_content_after(sentence: str, offset: int) -> bool:
    """Is there a real word after `offset`, or did the trigger consume
    everything worth reading in this sentence?"""
    return any(
        word.lower() not in _TRIVIAL_TAIL_WORDS for word in _WORD.findall(sentence[offset:])
    )


def _classify_sentence(sentence: str) -> tuple[str, int] | None:
    """Which layer, if any, marks this sentence as a prompt, and where its
    matched trigger ends -- callers use that offset to tell a complete prompt
    ("explain closures") from one caught mid-utterance ("explain")."""
    if _ACK_ONLY.match(sentence):
        # Pure acknowledgement. Rejected even with a trailing "?" so a tag
        # question like "Right?" does not trigger coaching.
        return None

    if sentence.rstrip().endswith("?"):
        # Whisper produced the "?" itself; trust it as closure rather than
        # second-guessing where the trigger sat inside the sentence.
        return REASON_PUNCTUATION, len(sentence)

    match = _PROMPT_ANYWHERE.search(sentence)
    if match:
        return REASON_INTERVIEW_PROMPT, match.end()

    for offset in _clause_starts(sentence):
        match = _CLAUSE_INITIAL.match(sentence, offset)
        if match is None:
            continue
        reason = (
            REASON_INTERROGATIVE
            if match.group("interrogative")
            else REASON_IMPERATIVE_TASK
        )
        return reason, match.end()

    return None


# ------------------------------------------------------- semantic openness
# A prompt can be a grammatically complete sentence and still not be a
# complete *request*. These two signals separate "the interviewer finished
# asking" from "the interviewer finished a clause", which is the difference
# between one provider call and two.

#: Premise openers. An interviewer who starts here is describing the setup,
#: and the actual request is still coming ("Given an array of integers...").
#: Only `given` currently reaches this check on its own -- assume/suppose/
#: imagine/consider are deliberately absent from `_CLAUSE_INITIAL` and so
#: never accepted alone -- but they do occur alongside a later trigger
#: ("Suppose we have a million users, tell me how you'd shard"), which is
#: why the whole family is listed.
_SCENARIO_OPENER = re.compile(
    r"""^\s*(?:(?:so|now|okay|ok|alright|well|and)[,\s]+)*
    (?:given|suppose|assume|imagine|consider|if|say)\b
      | \blet'?s\s+say\b
      | \byou\s+(?:have|are\s+given)\b
      | \bthere(?:'s|\s+is|\s+are)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: What makes a premise into a request: something actually being asked for.
#: Interrogatives, task imperatives (the `_CLAUSE_INITIAL` vocabulary), and
#: the explicit "I want you to" / "I need you to" framing.
_REQUEST_SIGNAL = re.compile(
    r"""\b(?:
        what|what's|whats|why|when|where|who|whom|whose|which|how
      | write|implement|code|build|design|create|solve|reverse|sort|return|find
      | list|name|discuss|rank|count|group|select|calculate|compute|explain
      | describe|tell|outline|summari[sz]e|compare|contrast
      | i\s+(?:want|need)\s+you\s+to
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

#: A request whose object is an unqualified indefinite plural -- "find two
#: numbers", "find some edge cases". English almost always continues these
#: with a restrictive clause ("...whose sum equals a target"), and that
#: clause is what makes the problem well-posed. Anchored to the end of the
#: prompt, so anything actually following the object (a relative clause, a
#: prepositional phrase) means the request already closed.
#:
#: The filler deliberately excludes prepositions, so "an array of integers"
#: is not read as the object of the request -- there, "integers" belongs to
#: "of", and the request ("find duplicate elements in an array of integers")
#: is complete.
#: Placeholder nouns that only mean something once a relative clause says
#: *which* one -- "tell me about a time" is not a request until "...you had
#: to ship under pressure" arrives. Distinguished from a real object by the
#: indefinite article: "your last project" is specific and complete, "a
#: project" is a placeholder.
_PLACEHOLDER_NOUNS = (
    "time|situation|example|instance|case|moment|project|experience|occasion|"
    "problem|scenario"
)

#: A quantity with no constraint on it yet: "find two numbers". Answerable in
#: principle -- badly -- so this is the ambiguous tier.
_OPEN_QUANTITY = re.compile(
    r"""\b(?:two|three|four|five|six|a|an|some|any|multiple|several)\s+
    (?:(?!of\b|in\b|on\b|for\b|with\b|from\b|to\b|that\b|which\b|whose\b)[a-z]+\s+){0,1}
    [a-z]+s\b[\s.,!?]*$""",
    re.IGNORECASE | re.VERBOSE,
)

#: A placeholder noun still waiting for its relative clause: "tell me about a
#: time". Not answerable at all -- there is no question here yet -- so this
#: belongs in the same tier as a dangling conjunction, not the ambiguous one.
_PLACEHOLDER_OBJECT = re.compile(
    rf"\b(?:a|an)\s+(?:{_PLACEHOLDER_NOUNS})\b[\s.,!?]*$",
    re.IGNORECASE,
)


#: An interviewer walking back a premise they just stated. What follows
#: replaces the setup, it does not add to it -- "Assume 1,000 QPS. Actually,
#: assume 10,000 QPS." must not reach the model as both numbers, or the model
#: is being asked to design for two contradictory loads at once.
_CORRECTION_OPENER = re.compile(
    r"""^\s*(?:
        # Lookahead, not a consumed delimiter: `no[,\s]` swallowed the comma,
        # after which the group's trailing \b sat between "," and " " -- two
        # non-word characters, so it never held and "No, make that 100
        # million" was not recognised as a correction at all. Checking the
        # delimiter without consuming it leaves the boundary on "no" itself,
        # which is what keeps "nobody" out.
        actually | sorry | no(?=[,\s]) | scratch\s+that | strike\s+that
      | instead | rather | correction | let\s+me\s+rephrase
      | i\s+mean | make\s+that | on\s+second\s+thought
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


def is_correction(text: str) -> bool:
    """True when this utterance revises what the interviewer just said."""
    return bool(_CORRECTION_OPENER.match(text.strip()))


def is_scenario_without_request(prompt: str) -> bool:
    """A premise with nothing asked for yet -- "Given an array of integers"."""
    if not _SCENARIO_OPENER.search(prompt):
        return False
    # Strip the opener before looking for a request, so the opener's own verb
    # ("given" is also a task imperative) cannot count as the request.
    remainder = _SCENARIO_OPENER.sub(" ", prompt, count=1)
    return not _REQUEST_SIGNAL.search(remainder)


def has_open_quantity(prompt: str) -> bool:
    """A request that names a quantity but not the constraint on it."""
    return bool(_OPEN_QUANTITY.search(prompt))


def has_placeholder_object(prompt: str) -> bool:
    """A request whose object is a placeholder awaiting its relative clause."""
    return bool(_PLACEHOLDER_OBJECT.search(prompt))


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
    matched_trigger_end: int | None = None
    for index, sentence in enumerate(sentences):
        result = _classify_sentence(sentence)
        if result is not None:
            matched_index, (matched_reason, matched_trigger_end) = index, result

    if matched_index is None or matched_reason is None:
        return None

    # Sparse only if nothing follows the trigger *and* nothing follows this
    # sentence either -- a trailing correction sentence ("...actually,
    # assume 10k QPS") is real content even though the matched sentence
    # itself might be bare. Never for REASON_PUNCTUATION: Whisper produced
    # the "?" itself, which this trusts as closure -- there is no "trigger"
    # to measure a tail against (`_classify_sentence` reports the whole
    # sentence as consumed), so the check would otherwise fire on every
    # ordinary complete question.
    sparse = (
        matched_reason != REASON_PUNCTUATION
        and matched_index == len(sentences) - 1
        and not _has_content_after(sentences[matched_index], matched_trigger_end)
    )

    tail = " ".join(sentences[matched_index:])
    # Everything before the question, minus the pleasantries. Filler is what
    # the last-match scan exists to discard ("Very good." / "Okay, thanks."),
    # so dropping it here keeps that behaviour while a real premise survives.
    premise = " ".join(
        sentence for sentence in sentences[:matched_index]
        if not is_acknowledgement(sentence) and _WORDS.search(sentence)
    ).strip()
    return PromptMatch(
        prompt=_normalize(tail),
        reason=matched_reason,
        sentence=sentences[matched_index],
        sparse=sparse,
        premise=premise[-_MAX_PREMISE_CHARS:] if premise else "",
    )


def is_interview_prompt(text: str) -> bool:
    """True when the transcript contains something worth answering."""
    return extract_interview_prompt(text) is not None
