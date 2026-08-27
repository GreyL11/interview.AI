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
    r"given an array|write a program|two sum|linked list|binary tree)\b",
    re.IGNORECASE,
)
_SQL = re.compile(r"\b(sql|select \*|join|group by|database schema|write a query)\b", re.IGNORECASE)
_DEBUGGING = re.compile(r"\b(bug|debug|not working|crash(ing)?|stack trace|why is this failing|throws an error)\b", re.IGNORECASE)
_SYSTEM_DESIGN = re.compile(r"\b(design a system|design an?|scale (this|to)|architecture for|how would you design)\b", re.IGNORECASE)
_ARCHITECTURE = re.compile(r"\b(microservices|monolith|architecture)\b", re.IGNORECASE)
_BEHAVIORAL = re.compile(r"\b(tell me about a time|describe a situation|biggest (weakness|strength)|handle conflict|team conflict)\b", re.IGNORECASE)
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
        (_CODING, Category.CODING),
        (_SQL, Category.SQL),
        (_DEBUGGING, Category.DEBUGGING),
        (_SYSTEM_DESIGN, Category.SYSTEM_DESIGN),
        (_BEHAVIORAL, Category.BEHAVIORAL),
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
