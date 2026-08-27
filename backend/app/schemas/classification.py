from enum import StrEnum

from pydantic import BaseModel, Field


class Category(StrEnum):
    PERSONAL_EXPERIENCE = "PERSONAL_EXPERIENCE"
    RESUME = "RESUME"
    PROJECT = "PROJECT"
    BEHAVIORAL = "BEHAVIORAL"
    TECHNICAL_KNOWLEDGE = "TECHNICAL_KNOWLEDGE"
    SYSTEM_DESIGN = "SYSTEM_DESIGN"
    SCENARIO = "SCENARIO"
    CODING = "CODING"
    SQL = "SQL"
    DEBUGGING = "DEBUGGING"
    ARCHITECTURE = "ARCHITECTURE"
    FOLLOW_UP = "FOLLOW_UP"
    UNKNOWN = "UNKNOWN"


class Domain(StrEnum):
    GENERAL = "GENERAL"
    SOFTWARE_ENGINEERING = "SOFTWARE_ENGINEERING"
    DATA_ENGINEERING = "DATA_ENGINEERING"
    DATA_SCIENCE = "DATA_SCIENCE"
    DEVOPS = "DEVOPS"
    FRONTEND = "FRONTEND"
    BACKEND = "BACKEND"
    DATABASE = "DATABASE"


class Classification(BaseModel):
    is_question: bool
    category: Category
    domain: Domain = Domain.GENERAL
    requires_personal_context: bool = False
    requires_rag: bool = False
    requires_reasoning: bool = False
    requires_code: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
