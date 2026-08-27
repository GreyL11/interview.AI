from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import health, question
from app.intelligence.answer_validator import AnswerValidationError
from app.llm.base import LLMError

app = FastAPI(title="Interview Coach API", version="0.1.0")

app.include_router(health.router)
app.include_router(question.router)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(AnswerValidationError)
async def answer_validation_error_handler(request: Request, exc: AnswerValidationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})
