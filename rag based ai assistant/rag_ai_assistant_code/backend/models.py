from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    TEACHER = "teacher"
    STUDENT = "student"


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str = Field(min_length=1)
    role: UserRole


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: UserRole
    full_name: str
    user_id: str


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: UserRole


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    message: str


class StudentScore(BaseModel):
    student_id: str
    full_name: str
    email: EmailStr
    quizzes_taken: int
    average_score: float
    last_quiz_at: datetime | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    document_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []


class QuizGenerateRequest(BaseModel):
    document_id: str
    num_questions: int = Field(default=5, ge=1, le=15)


class QuizQuestion(BaseModel):
    question_id: str
    question: str
    options: list[str]
    correct_index: int


class QuizQuestionPublic(BaseModel):
    question_id: str
    question: str
    options: list[str]


class QuizGenerateResponse(BaseModel):
    quiz_id: str
    document_id: str
    questions: list[QuizQuestionPublic]


class QuizPreviewQuestion(BaseModel):
    question_id: str
    question: str
    options: list[str]
    correct_index: int
    correct_answer: str


class QuizPreviewResponse(BaseModel):
    quiz_id: str
    document_id: str
    questions: list[QuizPreviewQuestion]


class QuizAnswer(BaseModel):
    question_id: str
    selected_index: int


class QuizReviewItem(BaseModel):
    question_id: str
    question: str
    options: list[str]
    selected_index: int | None = None
    correct_index: int
    correct_answer: str
    is_correct: bool


class QuizSubmitRequest(BaseModel):
    quiz_id: str
    answers: list[QuizAnswer]


class QuizSubmitResponse(BaseModel):
    quiz_id: str
    score: float
    correct: int
    total: int
    feedback: str
    results: list[QuizReviewItem] = []


class MessageResponse(BaseModel):
    message: str
    detail: Any | None = None
