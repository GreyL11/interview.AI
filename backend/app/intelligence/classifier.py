import re

from app.schemas.classification import Category, Classification, Domain

# ponytail: rule-based keyword matching, not an ML classifier. Good enough to
# route deterministically and stay testable without a live LLM call. Swap for
# a model (or a cheap LLM classification call) if keyword coverage proves
# too thin in practice.

_QUESTION_MARKERS = re.compile(
    r"^(how|what|why|when|where|who|which|can you|could you|would you|"
    r"tell me|describe|explain|walk me|do you|did you|have you|"
    r"is it|are you)\b",
    re.IGNORECASE,
)

_FOLLOW_UP = re.compile(r"^(and |what about|how about)\b", re.IGNORECASE)
_CODING = re.compile(
    r"\b(write a function|write code|implement|leetcode|reverse a|"
    r"given an array|write a program|two sum|linked list|binary tree|"
    r"character count|count.{0,15}character|palindrom\w*|longest substring|"
    # Spoken phrasings that carry no "write a function" preamble.
    r"in an array|of integers|sums? to|sum equals|"
    r"find duplicates?|duplicate elements|merge intervals|sliding window)\b",
    re.IGNORECASE,
)
# SQL means "produce or reason about a query", not "mentions a database".
# Bare topic words (table, index, database, query) are deliberately absent:
# "How does database indexing work?" is technical knowledge, and DATABASE is
# already carried by the Domain axis rather than the Category.
_SQL = re.compile(
    r"""(?:
        \b(?:sql|select\s+\*|join|group\s+by|database\s+schema|write\s+a\s+query)\b
        # "when would you use a window function", "do it with row_number"
      | \bwindow\s+function
      | \b(?:row_number|dense_rank|partition\s+by|correlated\s+subquery)\b
        # "the difference between WHERE and HAVING" -- HAVING alone is too
        # common an English word, so it only counts next to its SQL partner.
      | \b(?:where|group\s+by|select)\b[^.?!]{0,25}\bhaving\b
      | \bhaving\b[^.?!]{0,25}\b(?:where|group\s+by|select)\b
        # "second highest salary", "third largest order"
      | \b(?:second|third|fourth|fifth|nth|2nd|3rd)[\s-]+(?:highest|lowest|largest|smallest)\b
      | \brunning\s+total\b
        # "rank employees by salary"
      | \brank\s+\w+\s+by\b
        # Anti-join phrasing over a data entity: "customers who never placed
        # an order". Both halves are required so ordinary speech about people
        # ("a teammate who never replied") does not read as SQL.
      | \b(?:customers?|clients?|users?|employees?|students?|accounts?|orders?|
             products?|records?|rows?|transactions?|payments?)\b
        [^.?!]{0,40}?
        \b(?:who|that|which)\s+
        (?:never|no\s+longer|ha(?:ve|s)\s+not|have?n'?t|has\s?n'?t|did\s?n'?t|did\s+not|do\s?n'?t)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_DEBUGGING = re.compile(
    r"\bbug\b|\bdebug\b|\bnot working\b|\bisn'?t working\b|\bcrash(?:ing)?\b|"
    r"\bstack trace\b|\bwhy is this failing\b|\bthrows an error\b|"
    r"\bwhat'?s wrong\b|\bhow would you fix\b|\brace condition\b|"
    r"\bwhy (?:does|is|doesn'?t)\b.{0,30}\b(?:fail|failing|broken)\b",
    re.IGNORECASE,
)
_SYSTEM_DESIGN = re.compile(r"\b(design a system|design an?|scale (this|to)|architecture for|how would you design)\b", re.IGNORECASE)
_ARCHITECTURE = re.compile(r"\b(microservices|monolith|architecture)\b", re.IGNORECASE)
# A behavioral question asks for a *past personal narrative*. Two signals must
# line up: a narrative request ("tell me about", "describe", "give me an
# example") and a personal-past marker ("a time you...", "you've faced", "your
# biggest failure"). Requiring both is what keeps "Describe a situation where a
# cache would help" (technical) out while letting "Describe a situation where
# you had limited information" (behavioral) in.
_BEHAVIORAL = re.compile(
    r"""(?:
        # "tell me about a time", "describe a time you disagreed with..."
        \b(?:a|an|the)\s+time\s+(?:when\s+|that\s+)?you\b
      | \btell\s+(?:me|us)\s+about\s+a\s+time\b
        # "a situation where you led a project" -- the trailing "you" is what
        # separates it from "a situation where a cache would help".
      | \ba\s+situation\s+(?:where|when|in\s+which)\s+you\b
        # Narrative opener + an experience noun.
      | \b(?:tell\s+(?:me|us)\s+about|describe|share|
             give\s+(?:me|us)\s+an?\s+example\s+of|walk\s+(?:me|us)\s+through)\b
        [^.?!]{0,40}?
        \b(?:challenge|challenging|difficult|failure|failed|mistake|conflict|
             disagreement|setback|achievement|accomplishment|proud|
             went\s+wrong|fell\s+short)\b
        # "have you ever had a conflict", "how did you handle a difficult..."
      | \b(?:have\s+you\s+ever|did\s+you\s+ever|how\s+did\s+you)\b
        [^.?!]{0,40}?
        \b(?:conflict|disagree\w*|challenge|challenging|difficult|failure|
             mistake|stakeholder|teammate|manager|pushback|went\s+wrong)\b
        # "your biggest achievement"; and "the biggest challenge you've faced",
        # which needs the "you" so "the biggest challenge in distributed
        # systems" stays technical.
      | \byour\s+biggest\s+
        (?:challenge|failure|mistake|achievement|accomplishment|weakness|strength|success|regret)\b
      | \bbiggest\s+
        (?:challenge|failure|mistake|achievement|accomplishment|success|regret)\b
        [^.?!]{0,25}\byou(?:'ve|'re|\s+have)?\b
        # Leadership / ownership phrasings.
      | \byou\s+(?:took|take)\s+ownership\b
      | \bhow\s+have\s+you\s+(?:influenced|handled|dealt|managed|led|motivated|persuaded)\b
        # Retained from the original pattern.
      | \bbiggest\s+(?:weakness|strength)\b
      | \bhandle\s+conflict\b
      | \bteam\s+conflict\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_PERSONAL = re.compile(r"\b(have you ever|did you|your experience)\b", re.IGNORECASE)
_PROJECT = re.compile(r"\b(your project|a project you|walk me through a project)\b", re.IGNORECASE)
_RESUME = re.compile(r"\b(your resume|on your resume|your background)\b", re.IGNORECASE)
_SCENARIO = re.compile(r"\b(what would you do if|how would you handle|imagine (you|a))\b", re.IGNORECASE)

_DOMAIN_KEYWORDS: list[tuple[re.Pattern, Domain]] = [
    (re.compile(r"\b(pipeline|etl|data engineer|duplicate records|ingestion)\b", re.IGNORECASE), Domain.DATA_ENGINEERING),
    (re.compile(r"\b(model|machine learning|ml |feature engineering|training data)\b", re.IGNORECASE), Domain.DATA_SCIENCE),
    (re.compile(r"\b(react|css|frontend|dom|browser rendering)\b", re.IGNORECASE), Domain.FRONTEND),
    (re.compile(r"\b(kubernetes|ci/cd|deploy(ment)?|docker|infrastructure)\b", re.IGNORECASE), Domain.DEVOPS),
    (re.compile(r"\b(sql|database|index|query|schema|table)\b", re.IGNORECASE), Domain.DATABASE),
    (re.compile(r"\b(api|server|backend|microservice)\b", re.IGNORECASE), Domain.BACKEND),
]

_PERSONAL_CATEGORIES = {Category.PERSONAL_EXPERIENCE, Category.RESUME, Category.PROJECT, Category.BEHAVIORAL}
_REASONING_CATEGORIES = {
    Category.TECHNICAL_KNOWLEDGE,
    Category.SYSTEM_DESIGN,
    Category.SCENARIO,
    Category.DEBUGGING,
    Category.ARCHITECTURE,
    Category.CODING,
    Category.SQL,
}
_CODE_CATEGORIES = {Category.CODING, Category.SQL}


def _looks_like_question(text: str) -> bool:
    return text.strip().endswith("?") or bool(_QUESTION_MARKERS.match(text.strip()))


def _detect_category(text: str) -> tuple[Category, float]:
    checks: list[tuple[re.Pattern, Category]] = [
        (_FOLLOW_UP, Category.FOLLOW_UP),
        # SQL before CODING: "write a query to find duplicate records" carries
        # an unambiguous SQL marker, while CODING's array/duplicate phrasings
        # are ambiguous. _SQL does not match plain array problems, so this
        # ordering costs the coding path nothing.
        (_SQL, Category.SQL),
        (_CODING, Category.CODING),
        # Before DEBUGGING/SYSTEM_DESIGN: "tell me about a time you fixed a
        # production bug" is a story about the candidate, not a debugging task.
        (_BEHAVIORAL, Category.BEHAVIORAL),
        (_DEBUGGING, Category.DEBUGGING),
        (_SYSTEM_DESIGN, Category.SYSTEM_DESIGN),
        (_PROJECT, Category.PROJECT),
        (_RESUME, Category.RESUME),
        (_PERSONAL, Category.PERSONAL_EXPERIENCE),
        (_SCENARIO, Category.SCENARIO),
        (_ARCHITECTURE, Category.ARCHITECTURE),
    ]
    for pattern, category in checks:
        if pattern.search(text):
            return category, 0.9

    if _looks_like_question(text):
        return Category.TECHNICAL_KNOWLEDGE, 0.55

    return Category.UNKNOWN, 0.3


def _detect_domain(text: str) -> Domain:
    for pattern, domain in _DOMAIN_KEYWORDS:
        if pattern.search(text):
            return domain
    return Domain.GENERAL


def classify(question: str) -> Classification:
    category, confidence = _detect_category(question)

    # An imperative task ("Write a function...") is a valid interview prompt even
    # though it isn't phrased as a question, so a confident category match counts.
    is_question = category != Category.UNKNOWN or _looks_like_question(question) or "?" in question

    if not is_question:
        category = Category.UNKNOWN
        confidence = 0.9

    domain = _detect_domain(question)

    return Classification(
        is_question=is_question,
        category=category,
        domain=domain,
        requires_personal_context=category in _PERSONAL_CATEGORIES,
        requires_rag=category in _PERSONAL_CATEGORIES,
        requires_reasoning=category in _REASONING_CATEGORIES,
        requires_code=category in _CODE_CATEGORIES,
        confidence=confidence,
    )
