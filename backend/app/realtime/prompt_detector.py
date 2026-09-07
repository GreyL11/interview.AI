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
    #: True when the innermost clause of this prompt still wants an argument
    #: -- "Explain how Kafka handles", "How would you ensure that the
    #: service". See `has_open_clause`, which is the only place that decides
    #: it; kept on the match so the finality tier reads one boolean rather
    #: than re-deriving grammar from the normalised prompt (which has had a
    #: "?" appended and so has lost the speaker's own closure signal).
    open_clause: bool = False


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


#: An interviewer navigating the interview itself ("Next question.", "Let's
#: move to the next question.", "One more question.") rather than asking one.
#: `_PROMPT_ANYWHERE` below still matches "(?:another|next|final|last|one
#: more)\s+question" wherever it occurs, on purpose -- that is what lets an
#: attached question survive ("Next question: what is Kafka?", which ends in
#: "?" and never reaches this check anyway; "Here's the next question: how
#: does Kafka work?" likewise). This pattern catches the *other* case: the
#: phrase used on its own, with nothing following it, meaning "let's move
#: on" rather than "here is a request". Anchored at both ends like
#: `_ACK_ONLY`, so it only fires when navigation is literally all the
#: utterance says -- anything attached after "question" (a colon, a real
#: question) fails the trailing `$` and falls through to ordinary detection.
_NAVIGATION_ONLY = re.compile(
    r"""^\s*
    (?:(?:o?k(?:ay)?|alright|all\s+right|so|now|well|and|but)[\s,]+)*
    (?:
        let'?s\s+(?:move\s+(?:on\s+)?to|go\s+(?:on\s+)?to|do)\s+
      | (?:we'?ll|we\s+will)\s+(?:go\s+(?:on\s+)?to|move\s+(?:on\s+)?to)\s+
    )?
    (?:the\s+)?
    (?:next|another|one\s+more|final|last)\s+question
    [\s.,!?]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_navigation_chatter(text: str) -> bool:
    """True for "Next question.", "Let's move to the next question.", "One
    more question." -- the interviewer moving the interview along, not
    asking one. An attached question ("Next question: what is Kafka?")
    still contains real content after "question" and does not match."""
    return bool(_NAVIGATION_ONLY.match(text.strip()))


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
               # "Show code", "Show the implementation" -- the same request as
               # "Show me the code" with the addressee left out, which is
               # ordinary spoken English and was being dropped entirely
               # because `_PROMPT_ANYWHERE` only recognises "show me|us".
               # Belongs here rather than there for the same reason every
               # other verb does: matched anywhere it would fire inside
               # ordinary speech ("the metrics show latency rising"), and
               # clause-initial anchoring is what keeps it a request.
               show|
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


#: Pronouns that stand in for a topic the interviewer can point at -- on
#: screen, in a paste, in the code just written. "Explain THIS", "optimize
#: IT" name no topic in words, but they do name one, which is why
#: `is_referentially_empty` accepts them and why `_has_content_after` must
#: count them as content. Defined here, above both users, because the two
#: readings have to agree: while this set was only consulted by the
#: referential check and `_TRIVIAL_TAIL_WORDS` below independently called
#: the same words trivial, "Explain this." was accepted as a real request
#: and then held for three seconds as though nothing had been asked.
_OBJECT_STANDINS = frozenset({"it", "this", "that", "them"})

#: Words that never count as "the interviewer went on to say something" when
#: they're all that's left after a trigger phrase -- pronouns and auxiliaries
#: that any unfinished clause is equally likely to open with ("How would
#: you...", "Tell me about..."). Deliberately not `_DANGLING_WORDS` from
#: question_detector: that set flags a sentence by its *last* word anywhere,
#: this one only judges the tail immediately after a matched trigger.
#:
#: Object standins are subtracted rather than absent, so the two sets cannot
#: drift back apart. A trailing "that" is still held, by `_DANGLING_WORDS` --
#: which is the right layer for it, since "that" is genuinely ambiguous
#: between an object and a complementiser.
_TRIVIAL_TAIL_WORDS = frozenset({
    "me", "us", "you", "i", "we", "they", "he", "she",
    "can", "could", "would", "will", "do", "does", "did",
    "is", "are", "was", "were", "about", "to", "of",
}) - _OBJECT_STANDINS

_WORD = re.compile(r"[A-Za-z']+")


def _has_content_after(sentence: str, offset: int) -> bool:
    """Is there a real word after `offset`, or did the trigger consume
    everything worth reading in this sentence?

    Deliberately just "is there one substantive word", for either kind of
    trigger. An earlier version required *two* after an interrogative, to
    tell "would you design" (open) from "would you design a distributed
    system" (closed) by counting. Counting cannot do that job: "what is
    Kafka" has exactly one substantive word after its trigger and is a
    finished question, so every short question of that shape paid a 3s hold
    it did not owe. Saturation is a structural property, not a word count --
    `has_open_clause` below answers it structurally, and this function is
    back to the one thing a word count *can* decide: whether the interviewer
    said anything at all after naming the kind of request.
    """
    return any(
        word.lower() not in _TRIVIAL_TAIL_WORDS
        for word in _WORD.findall(sentence[offset:])
    )


def _classify_sentence(sentence: str) -> tuple[str, int] | None:
    """Which layer, if any, marks this sentence as a prompt, and where its
    matched trigger ends -- callers use that offset to tell a complete prompt
    ("explain closures") from one caught mid-utterance ("explain")."""
    if _ACK_ONLY.match(sentence):
        # Pure acknowledgement. Rejected even with a trailing "?" so a tag
        # question like "Right?" does not trigger coaching.
        return None

    if _NAVIGATION_ONLY.match(sentence):
        # "Next question.", "One more question." -- moving the interview
        # along, not asking one. Rejected before the "?" check for the same
        # reason as an acknowledgement: an attached question ("Next
        # question: what is Kafka?") already fails this pattern's own
        # trailing anchor (see `is_navigation_chatter`) and falls through to
        # ordinary detection further down, "?" and all.
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


#: A noun phrase opened by a determiner and never closed: "a distributed".
#: `distributed` premodifies a noun that has not arrived yet -- the head of
#: "a distributed system" is "system", not "distributed" -- so this is
#: anchored to the *end* of the prompt and only matches when the participle
#: is the last thing said. "a distributed system" ends on "system" and never
#: reaches this pattern; "a distributed" ends on the modifier itself and does.
#:
#: Deliberately morphological (a past-participle shape after a determiner),
#: not a list of adjectives, so it generalises to any premodifier of that
#: shape ("a partitioned", "a sharded") rather than naming "distributed".
#:
#: Past participles only. An earlier version also matched "-ing", which is
#: where the shape stops being evidence: after a determiner an "-ing" word is
#: usually a gerund standing as the head noun of a *finished* phrase -- "the
#: encoding", "the padding", "a rating", "a string", "the meeting" -- and
#: every one of those was held for three seconds it did not owe. English has
#: no matching noun use for a bare "-ed" form, so that half of the shape
#: still carries the signal. The cost is that "a scaling" is no longer
#: caught, which is the right trade: one uncommon phrase waits less than it
#: should, instead of every ordinary "-ing" noun waiting when it should not.
#: `_is_participle` then rules out the words that merely end in those two
#: letters ("a thread", "a feed").
_OPEN_PREMODIFIER = re.compile(
    r"""\b(?:a|an|the|this|that|these|those|some|any|no|every|each)\s+
    (?:[a-z]+\s+){0,2}
    (?P<tail>[a-z]*ed)\b[\s.,!?]*$""",
    re.IGNORECASE | re.VERBOSE,
)


def has_open_premodifier(prompt: str) -> bool:
    """A determiner opened a noun phrase and the utterance ends on what looks
    like a premodifier ("a distributed") rather than the head noun it
    modifies ("a distributed system")."""
    match = _OPEN_PREMODIFIER.search(prompt)
    return bool(match) and _is_participle(match.group("tail"))


#: Shortest stem that makes an "-ed" ending a real past participle. An
#: English participle is a verb plus the suffix, so the verb has to survive
#: stripping it: "distribut(e)+ed", "partition+ed", "shard+ed", "link+ed" all
#: leave four or more characters. The words that merely *end* in those two
#: letters do not -- "thread" leaves "thr", "speed" leaves "spe", "head"
#: leaves "h" -- which is what keeps "What is a thread" from being read as an
#: unfinished noun phrase. Four is the shortest stem observed among the
#: participles this pipeline actually meets; three would admit "bed"/"red".
_MIN_PARTICIPLE_STEM = 4


def _is_participle(word: str) -> bool:
    """Is this "-ed" word a past participle, or just a noun that ends in the
    same two letters?

    Morphological, not a word list: strip the suffix and ask whether enough
    of a verb is left to have carried it. See `_MIN_PARTICIPLE_STEM`.
    """
    stem = word[:-2] if word.lower().endswith("ed") else ""
    return len(stem.rstrip("e")) >= _MIN_PARTICIPLE_STEM or len(stem) >= _MIN_PARTICIPLE_STEM


# --------------------------------------------------------- clause saturation
# The signal that separates "the interviewer finished a clause" from "the
# interviewer finished asking", without a verb lexicon and without an LLM.
#
# Everything below reads only *closed* word classes -- determiners,
# auxiliaries, pronouns, prepositions, subordinators, interrogatives. Those
# are genuinely finite sets in English (new verbs and nouns appear every
# year; new articles do not), which is what makes this generalise instead of
# being the "giant list of English keywords" this module is forbidden to
# become. Every open-class word -- every noun, verb, adjective, adverb the
# interviewer might use -- is simply CONTENT, unlisted and unnamed.
#
# The question asked of a clause is whether its predicate has an argument:
#
#     "how the database"              subject, no predicate      -> open
#     "how Kafka handles"             predicate, no complement   -> open
#     "how Kafka handles partitions"  predicate + complement     -> closed
#
# and the one genuinely undecidable case -- a predicate with no complement,
# which is a finished question for an intransitive verb ("how Kafka works")
# and an unfinished one for a transitive verb ("how Kafka handles") -- is
# decided by whoever knows more than the grammar does: the speaker. A
# terminal "." or "!" is Whisper reporting the utterance ended, which is the
# same weak-but-real closure signal `is_referentially_empty`'s caller already
# trusts. With no punctuation at all there is nothing to trust, so an
# unsaturated predicate waits.

_COPULAS = frozenset({"is", "are", "was", "were", "am", "be", "been", "being"})

_AUXILIARIES = _COPULAS | frozenset({
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
})

#: Pronouns that put the *candidate* in the subject position. A request framed
#: at the candidate ("would you design", "should we shard") names an action
#: the candidate is being asked to perform, and the thing to perform it on is
#: exactly what has not been said yet. A third-person subject ("does it
#: rebalance", "does Kafka replicate") asks about the behaviour of a thing,
#: which is a whole question with no object at all.
_ADDRESSEE_PRONOUNS = frozenset({"i", "we", "you"})

_PRONOUNS = _ADDRESSEE_PRONOUNS | frozenset({
    "it", "they", "he", "she", "there", "them", "him", "her", "me", "us",
    "this", "that", "these", "those", "one", "ones", "something", "anything",
})

_DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "no",
    "every", "each", "my", "your", "our", "their", "its", "his", "her",
    "which", "what", "whose", "both", "either", "neither",
})

_PREPOSITIONS = frozenset({
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "about",
    "into", "onto", "over", "under", "between", "among", "through", "across",
    "against", "during", "without", "within", "upon", "off", "out", "up",
    "down", "as", "than", "like", "per", "via", "behind", "before", "after",
    "below", "above", "along", "around", "beyond", "near", "toward", "towards",
})

#: Conjunctions and discourse particles. The second half is deliberately the
#: same vocabulary `_CLAUSE_BOUNDARY` already treats as "a clause may start
#: after this" -- reused rather than re-listed, so the two cannot disagree
#: about whether "Now" or "Actually" is a word the interviewer said or just
#: how they got to the next thing.
_CONNECTIVES = frozenset({
    "and", "or", "but", "nor", "so", "then", "also", "plus", "however",
    "because", "while", "though", "although", "since", "unless", "until",
    "when", "where", "whenever", "wherever", "not", "just", "very", "really",
    "now", "okay", "ok", "alright", "well", "next", "finally", "lastly",
    "actually", "anyway",
})

#: Interrogatives that can open a clause. `who/whom/what/which/whose` can be
#: the subject of the clause they open ("who owns the schema"); `how/why/
#: when/where` are adverbial and the clause still needs a subject of its own
#: ("how Kafka handles" -- "how" is not what handles anything).
_SUBJECT_WH = frozenset({"who", "whom", "what", "which", "whose"})
_ADVERBIAL_WH = frozenset({"how", "why", "when", "where"})
_CLAUSE_WH = _SUBJECT_WH | _ADVERBIAL_WH

#: Complementisers. Unlike an interrogative these are ambiguous -- "that" is a
#: relative pronoun in "numbers that sum to a target" (where it *is* the
#: subject and the next word is the verb) and a complementiser in "ensure
#: that the service ..." (where a whole clause follows). Only the second
#: reading opens a clause worth checking, and it is the one where a
#: determiner or pronoun follows, so that is the condition.
_COMPLEMENTISERS = frozenset({"that", "if", "whether"})

#: Contracted copulas, expanded before categorisation so "what's the plan"
#: reads as "what is the plan" rather than as one unknown token.
_CONTRACTIONS: dict[str, tuple[str, ...]] = {
    "what's": ("what", "is"), "whats": ("what", "is"),
    "who's": ("who", "is"), "how's": ("how", "is"),
    "where's": ("where", "is"), "when's": ("when", "is"),
    "why's": ("why", "is"), "that's": ("that", "is"),
    "it's": ("it", "is"), "there's": ("there", "is"),
    "he's": ("he", "is"), "she's": ("she", "is"),
    "you're": ("you", "are"), "we're": ("we", "are"),
    "they're": ("they", "are"), "i'm": ("i", "am"),
}

#: Every closed class together: what is left over is content.
_FUNCTION_WORDS = (
    _AUXILIARIES | _PRONOUNS | _DETERMINERS | _PREPOSITIONS
    | _CONNECTIVES | _CLAUSE_WH | _COMPLEMENTISERS
)


def _tokens(sentence: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(sentence):
        out.extend(_CONTRACTIONS.get(word.lower(), (word.lower(),)))
    return out


def _is_content(token: str) -> bool:
    return token not in _FUNCTION_WORDS


def _is_argument(token: str) -> bool:
    """Could this token be part of a predicate's argument?

    Auxiliaries, connectives and subordinators cannot -- "should" in "the
    system should recover" is not what the system recovers. Everything that
    can head or fill an argument counts: content words, determiners,
    pronouns, prepositions.
    """
    return token not in _AUXILIARIES and token not in _CONNECTIVES \
        and token not in _COMPLEMENTISERS


def _last_clause_start(tokens: list[str]) -> tuple[int, str | None]:
    """Where the innermost clause begins, and the word that opened it.

    The innermost clause is the one still being spoken, so it is the one
    whose saturation decides whether the utterance has finished.
    """
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token in _CLAUSE_WH:
            return index + 1, token
        if token in _COMPLEMENTISERS and index + 1 < len(tokens):
            following = tokens[index + 1]
            if following in _DETERMINERS or following in _PRONOUNS:
                return index + 1, token
    return 0, None


def _clause_is_open(tokens: list[str], opener: str | None, terminated: bool) -> bool:
    """Does this clause's predicate still want an argument?

    `opener` is the interrogative or complementiser that introduced the
    clause, or None for a main clause. `terminated` is whether the speaker
    marked the utterance as ended, which is the tie-breaker for the one case
    grammar alone cannot settle.
    """
    if not tokens:
        # A trigger or interrogative with nothing after it at all.
        return True

    # A copula's complement is its argument, and it comes straight after:
    # "what is Kafka" is finished, "how the system is" is not.
    for index, token in enumerate(tokens):
        if token in _COPULAS:
            return not any(_is_content(later) for later in tokens[index + 1:])

    inverted = tokens[0] in _AUXILIARIES
    cursor = 1 if inverted else 0

    # Whose action is being described. A main clause and an inverted question
    # state their subject; an adverbial interrogative's clause has to as well;
    # a subject interrogative ("who owns...") already is the subject, and a
    # bare imperative ("reverse a string") has none.
    subject_is_addressee = False
    #: A bare instruction ("reverse a string", "now handle retries") has no
    #: subject to find: the interviewer is telling the candidate to do
    #: something, and the clause is exactly a predicate plus its argument.
    imperative = opener is None and not inverted
    if not imperative:
        if cursor < len(tokens) and tokens[cursor] in _PRONOUNS:
            subject_is_addressee = tokens[cursor] in _ADDRESSEE_PRONOUNS
            cursor += 1
        elif opener in _SUBJECT_WH and not inverted:
            pass  # the interrogative itself is the subject
        else:
            # A noun-phrase subject: its determiner and modifiers, then its
            # head. The head is the last content word before anything that
            # can only start a *new* argument, which is what tells "owns the
            # schema" (predicate + object) from "the database handles"
            # (subject + predicate).
            while cursor < len(tokens) and (
                tokens[cursor] in _DETERMINERS or tokens[cursor] in _CONNECTIVES
            ):
                cursor += 1
            if cursor < len(tokens) and _is_content(tokens[cursor]):
                cursor += 1  # the subject's head noun

    # The predicate is the next open-class word: whatever auxiliaries,
    # determiners or discourse particles sit between the subject and it are
    # not the action being described ("should" in "the system should
    # recover", "actually" in "no, actually reverse a linked list").
    while cursor < len(tokens) and not _is_content(tokens[cursor]):
        cursor += 1

    if cursor >= len(tokens):
        return True  # a subject with no predicate at all: "how the database"
    cursor += 1  # the predicate

    if any(_is_argument(token) for token in tokens[cursor:]):
        return False  # the predicate has its argument

    # A predicate with nothing after it at all.
    #
    # An instruction to the candidate is unfinished whatever else is true:
    # "reverse", "now handle", "how would you design" name an action and not
    # the thing to do it to, and the thing to do it to is the question.
    if imperative or subject_is_addressee:
        return True
    # Otherwise it is genuinely undecidable from grammar -- intransitive and
    # finished ("how Kafka works"), or transitive and interrupted ("how Kafka
    # handles") -- so it turns on the two things that are not grammar: an
    # inverted main clause is a whole question already ("does it rebalance"),
    # and a speaker who terminated the utterance said they were done.
    return not terminated and not inverted and opener is not None


def has_open_clause(text: str) -> bool:
    """True when the utterance's innermost clause provably wants more.

    The one deterministic signal for a subordinate clause caught mid-flight
    -- "Explain how Kafka handles", "Tell me how the database", "How would
    you ensure that the service" -- which no amount of counting words or
    inspecting the final word can see, because none of those clauses end on a
    function word and all of them are several words long.
    """
    sentences = _sentences(text)
    if not sentences:
        return False
    last = sentences[-1]
    tokens = _tokens(last)
    if not tokens:
        return False
    start, opener = _last_clause_start(tokens)
    return _clause_is_open(
        tokens[start:], opener, terminated=last.rstrip().endswith((".", "!", "?"))
    )


#: The generic request verbs already enumerated in `_PROMPT_ANYWHERE` /
#: `_CLAUSE_INITIAL` above ("explain", "show", "tell", ...) -- reused here
#: for a second question: does one of them GOVERN an object standing in for
#: a topic (`_OBJECT_STANDINS`, checked by `_has_verb_with_standin_object`)?
#: "Explain THIS", "optimize IT", "answer THAT" name no topic in words, but
#: the pronoun points at something presentable -- on screen, on the page --
#: that can exist with no prior turn to resolve it against, unlike a bare
#: trigger with nothing after it ("Can you explain?") or one whose only
#: object is a personal pronoun ("Show ME" names who receives the answer,
#: never what the answer is about, so it does not count).
_REQUEST_VERBS = frozenset({
    "explain", "describe", "elaborate", "compare", "contrast", "define",
    "outline", "summarize", "summarise", "illustrate", "demonstrate",
    "tell", "walk", "talk", "show", "give", "share", "help", "understand",
    "write", "implement", "code", "build", "design", "create", "solve",
    "reverse", "sort", "return", "find", "given", "list", "name",
    "discuss", "rank", "count", "group", "select", "calculate", "compute",
    "handle", "optimize", "optimise", "refactor", "extend", "modify",
    "answer",
})
# `_OBJECT_STANDINS` is defined above, beside `_TRIVIAL_TAIL_WORDS`, because
# both readings of a standin pronoun have to come from the same set.

#: Words that carry no topic of their own -- interrogatives, auxiliaries and
#: subject pronouns (the same closed classes as `_TRIVIAL_TAIL_WORDS`, plus
#: the interrogatives themselves, which that set deliberately excludes
#: because it is applied *after* one has already been matched as a trigger),
#: a handful of discourse/referential particles ("then", "so", "not", "one"),
#: and `_REQUEST_VERBS` itself. None of these name a subject; a request
#: built from nothing else -- and that does not clear
#: `_has_verb_with_standin_object` either -- is leaning entirely on context
#: that has to already exist.
#: `_OBJECT_STANDINS` is added back explicitly, because the two questions the
#: two sets answer are different and a standin answers them oppositely.
#: "Did the interviewer say anything after the trigger?" -- yes, "this" in
#: "Explain this" is the object, which is why `_TRIVIAL_TAIL_WORDS` no longer
#: contains it. "Does this utterance name a topic of its own?" -- no, "it" in
#: "What about it?" points at a conversation that has to already exist. The
#: case where a standin *does* supply a topic is the one where a request verb
#: governs it, and `_has_verb_with_standin_object` is what recognises that.
_REFERENTIAL_FILLER = (
    _TRIVIAL_TAIL_WORDS | _OBJECT_STANDINS | _REQUEST_VERBS | frozenset({
        "what's", "whats", "what", "how", "why", "when", "where", "which",
        "who", "whom", "whose",
        "then", "so", "not", "one", "really", "also",
    })
)


def _has_verb_with_standin_object(words: list[str]) -> bool:
    lowered = [w.lower() for w in words]
    return any(
        w in _REQUEST_VERBS and any(tail in _OBJECT_STANDINS for tail in lowered[i + 1:])
        for i, w in enumerate(lowered)
    )


def is_referentially_empty(text: str) -> bool:
    """True when this utterance names no topic of its own -- "Which one?",
    "Can you?", "Show me.", "Can you explain?", "What about it?" -- as
    against a genuinely self-contained short request that is just as few
    words but does name one -- "What is Kafka?", "Explain Kafka.", "Use
    SQL." -- or one whose verb takes a topic-standing-in pronoun as its
    object -- "Explain this.", "Can we optimize it?". Whatever survives
    stripping the closed-class scaffolding in `_REFERENTIAL_FILLER` (or
    resolving as a governed object-standin) is that topic; nothing left
    means the request has none of its own and can only be answered by
    leaning on a conversation that already exists.
    """
    words = _WORD.findall(text)
    if not words:
        return False
    if _has_verb_with_standin_object(words):
        return False
    return all(word.lower() in _REFERENTIAL_FILLER for word in words)


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
        if not is_acknowledgement(sentence)
        and not is_navigation_chatter(sentence)
        and _WORDS.search(sentence)
    ).strip()
    return PromptMatch(
        prompt=_normalize(tail),
        reason=matched_reason,
        sentence=sentences[matched_index],
        sparse=sparse,
        premise=premise[-_MAX_PREMISE_CHARS:] if premise else "",
        # Computed on the raw tail, not on `prompt`: `_normalize` appends the
        # "?" that Whisper did not, and whether the speaker terminated the
        # utterance themselves is exactly what the check needs to know.
        open_clause=(
            matched_reason != REASON_PUNCTUATION and has_open_clause(tail)
        ),
    )


def is_interview_prompt(text: str) -> bool:
    """True when the transcript contains something worth answering."""
    return extract_interview_prompt(text) is not None
