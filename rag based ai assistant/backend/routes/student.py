from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.deps import get_current_user, require_role
from backend.models import (
    ChatRequest,
    ChatResponse,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizQuestionPublic,
    QuizSubmitRequest,
    QuizSubmitResponse,
    UserRole,
)
from backend.services import db, llm, rag

router = APIRouter(prefix="/student", tags=["student"])


@router.get("/documents")
async def list_documents(current_user: Annotated[dict, Depends(get_current_user)]):
    require_role(current_user, UserRole.STUDENT)
    docs = await db.list_all_documents()
    return [
        {
            "document_id": str(d["_id"]),
            "filename": d["filename"],
            "uploaded_at": d.get("uploaded_at"),
        }
        for d in docs
    ]


async def _resolve_namespace(document_id: str | None) -> str:
    if document_id:
        doc = await db.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return doc["pinecone_namespace"]

    docs = await db.list_all_documents()
    if not docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No course materials available",
        )
    return docs[0]["pinecone_namespace"]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    require_role(current_user, UserRole.STUDENT)
    namespace = await _resolve_namespace(body.document_id)
    chunks = await rag.retrieve_context(body.message, namespace)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content found for this PDF. Ask your teacher to upload the PDF again.",
        )
    try:
        answer, sources = await llm.answer_with_context(body.message, chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return ChatResponse(answer=answer, sources=sources)


@router.post("/generate-quiz", response_model=QuizGenerateResponse)
async def generate_quiz(
    body: QuizGenerateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    require_role(current_user, UserRole.STUDENT)
    doc = await db.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    namespace = doc["pinecone_namespace"]
    chunk_count = doc.get("chunk_count", 0)

    chunks = await rag.retrieve_context(
        "key concepts definitions facts topics",
        namespace,
        top_k=12,
    )
    if len(chunks) < chunk_count:
        all_chunks = await rag.retrieve_all_chunks(namespace, max_chunks=max(chunk_count, 12))
        if all_chunks:
            chunks = all_chunks

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No content indexed for this document. Re-upload the PDF as teacher.",
        )

    # Small PDFs: fewer questions work better (you had only 3 chunks)
    num_questions = min(body.num_questions, max(1, len(chunks)))

    try:
        raw_questions = await llm.generate_quiz_questions(chunks, num_questions)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    quiz_id = await db.save_quiz(
        student_id=current_user["id"],
        document_id=body.document_id,
        questions=raw_questions,
    )

    questions = [
        QuizQuestionPublic(
            question_id=q["question_id"],
            question=q["question"],
            options=q["options"],
        )
        for q in raw_questions
    ]
    return QuizGenerateResponse(
        quiz_id=quiz_id,
        document_id=body.document_id,
        questions=questions,
    )


@router.post("/submit-quiz", response_model=QuizSubmitResponse)
async def submit_quiz(
    body: QuizSubmitRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    require_role(current_user, UserRole.STUDENT)
    quiz = await db.get_quiz(body.quiz_id)
    if not quiz:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Quiz not found")
    if quiz["student_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your quiz")

    answer_map = {a.question_id: a.selected_index for a in body.answers}
    correct = 0
    total = len(quiz["questions"])
    results = []

    for q in quiz["questions"]:
        selected = answer_map.get(q["question_id"])
        correct_index = int(q["correct_index"])
        is_correct = selected is not None and selected == correct_index
        if is_correct:
            correct += 1
        options = q["options"]
        results.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "options": options,
                "selected_index": selected,
                "correct_index": correct_index,
                "correct_answer": options[correct_index],
                "is_correct": is_correct,
            }
        )

    score = round((correct / total) * 100, 2) if total else 0.0
    feedback = (
        f"Excellent work! You scored {score}%."
        if score >= 80
        else f"You scored {score}%. Review the material and try again."
    )

    await db.save_quiz_submission(
        student_id=current_user["id"],
        quiz_id=body.quiz_id,
        score=score,
        correct=correct,
        total=total,
    )

    return QuizSubmitResponse(
        quiz_id=body.quiz_id,
        score=score,
        correct=correct,
        total=total,
        feedback=feedback,
        results=results,
    )
