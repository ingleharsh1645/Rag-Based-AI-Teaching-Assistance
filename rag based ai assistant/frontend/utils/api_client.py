import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120


def backend_is_running() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.ok
    except requests.RequestException:
        return False


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _handle(response: requests.Response) -> Any:
    if response.ok:
        if response.content:
            return response.json()
        return None
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    raise APIError(str(detail), response.status_code)


def register(email: str, password: str, full_name: str, role: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={"email": email, "password": password, "full_name": full_name, "role": role},
        timeout=TIMEOUT,
    )
    return _handle(response)


def login(email: str, password: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": email, "password": password},
        timeout=TIMEOUT,
    )
    return _handle(response)


def upload_pdf(token: str, file_bytes: bytes, filename: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/teacher/upload-pdf",
        headers=_headers(token),
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=TIMEOUT,
    )
    return _handle(response)


def get_students(token: str) -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/teacher/students",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    return _handle(response)


def list_teacher_documents(token: str) -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/teacher/documents",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    return _handle(response)


def generate_teacher_preview_quiz(token: str, document_id: str, num_questions: int = 5) -> dict:
    response = requests.post(
        f"{BASE_URL}/teacher/generate-preview-quiz",
        headers=_headers(token),
        json={"document_id": document_id, "num_questions": num_questions},
        timeout=TIMEOUT,
    )
    return _handle(response)


def list_documents(token: str) -> list[dict]:
    response = requests.get(
        f"{BASE_URL}/student/documents",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    return _handle(response)


def chat(token: str, message: str, document_id: str | None = None) -> dict:
    payload: dict[str, Any] = {"message": message}
    if document_id:
        payload["document_id"] = document_id
    response = requests.post(
        f"{BASE_URL}/student/chat",
        headers=_headers(token),
        json=payload,
        timeout=TIMEOUT,
    )
    return _handle(response)


def generate_quiz(token: str, document_id: str, num_questions: int = 5) -> dict:
    response = requests.post(
        f"{BASE_URL}/student/generate-quiz",
        headers=_headers(token),
        json={"document_id": document_id, "num_questions": num_questions},
        timeout=TIMEOUT,
    )
    return _handle(response)


def submit_quiz(token: str, quiz_id: str, answers: list[dict]) -> dict:
    response = requests.post(
        f"{BASE_URL}/student/submit-quiz",
        headers=_headers(token),
        json={"quiz_id": quiz_id, "answers": answers},
        timeout=TIMEOUT,
    )
    return _handle(response)
