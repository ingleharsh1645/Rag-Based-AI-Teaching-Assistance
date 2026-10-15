import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.deps import get_current_user, require_role
from backend.models import (
    DocumentUploadResponse,
    QuizGenerateRequest,
    QuizPreviewQuestion,
    QuizPreviewResponse,
    StudentScore,
    UserRole,
)
from backend.services import db, llm, rag

router = APIRouter(prefix="/teacher", tags=["teacher"])


@router.post("/upload-pdf", response_model=DocumentUploadResponse)
async def upload_pdf(
    current_user: Annotated[dict, Depends(get_current_user)],
    file: UploadFile = File(...),
):
    require_role(current_user, UserRole.TEACHER)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    namespace = f"doc-{uuid.uuid4().hex}"
    chunk_count, _ = await rag.index_pdf(content, namespace)

    if chunk_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract text from PDF",
        )

    document_id = await db.save_document(
        teacher_id=current_user["id"],
        filename=file.filename,
        pinecone_namespace=namespace,
        chunk_count=chunk_count,
    )

    if chunk_count <= 2:
        msg = (
            f"PDF indexed with only {chunk_count} chunk(s). "
            "Little text was found — use a text-based PDF (not scanned images) for better chat/quiz."
        )
    else:
        msg = "PDF indexed successfully"

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_indexed=chunk_count,
        message=msg,
    )


@router.get("/documents")
async def list_documents(current_user: Annotated[dict, Depends(get_current_user)]):
    require_role(current_user, UserRole.TEACHER)
    docs = await db.list_documents_for_teacher(current_user["id"])
    return [
        {
            "document_id": str(d["_id"]),
            "filename": d["filename"],
            "chunk_count": d.get("chunk_count", 0),
            "uploaded_at": d.get("uploaded_at"),
        }
        for d in docs
    ]


@router.post("/generate-preview-quiz", response_model=QuizPreviewResponse)
async def generate_preview_quiz(
    body: QuizGenerateRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    require_role(current_user, UserRole.TEACHER)
    doc = await db.get_document(body.document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.get("teacher_id") != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your document")

    namespace = doc["pinecone_namespace"]
    chunk_count = doc.get("chunk_count", 0)
    chunks = await rag.retrieve_context(
        "key concepts definitions facts topics exam instructions",
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
            detail="No content indexed for this document. Upload the PDF again.",
        )

    num_questions = min(body.num_questions, max(1, len(chunks)))
    try:
        raw_questions = await llm.generate_quiz_questions(chunks, num_questions)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    questions = [
        QuizPreviewQuestion(
            question_id=q["question_id"],
            question=q["question"],
            options=q["options"],
            correct_index=q["correct_index"],
            correct_answer=q["options"][q["correct_index"]],
        )
        for q in raw_questions
    ]
    return QuizPreviewResponse(
        quiz_id="teacher-preview",
        document_id=body.document_id,
        questions=questions,
    )


@router.get("/students", response_model=list[StudentScore])
async def list_students(current_user: Annotated[dict, Depends(get_current_user)]):
    require_role(current_user, UserRole.TEACHER)
    rows = await db.get_student_scores()
    return [StudentScore(**row) for row in rows]
