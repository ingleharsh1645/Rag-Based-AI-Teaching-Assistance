import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bcrypt
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: AsyncIOMotorClient | None = None
_db: AsyncIOMotorDatabase | None = None
_local_data: dict[str, list[dict[str, Any]]] = {}
_collections = ("users", "documents", "quizzes", "quiz_submissions")
_date_fields = {"created_at", "uploaded_at", "submitted_at", "last_quiz_at"}


def _local_db_path() -> Path:
    return Path(os.environ.get("LOCAL_DB_PATH", "local_db.json"))


def _empty_local_data() -> dict[str, list[dict[str, Any]]]:
    return {name: [] for name in _collections}


def _deserialize_value(key: str, value: Any) -> Any:
    if key == "_id" and isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    if key in _date_fields and isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, dict):
        return {k: _deserialize_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize_value("", item) for item in value]
    return value


def _serialize_value(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _load_local_data() -> None:
    global _local_data
    import json

    path = _local_db_path()
    if not path.exists():
        _local_data = _empty_local_data()
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    _local_data = _empty_local_data()
    for name in _collections:
        _local_data[name] = [
            {key: _deserialize_value(key, value) for key, value in doc.items()}
            for doc in raw.get(name, [])
        ]


def _save_local_data() -> None:
    import json

    path = _local_db_path()
    payload = {
        name: [_serialize_value(doc) for doc in docs]
        for name, docs in _local_data.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _local_collection(name: str) -> list[dict[str, Any]]:
    if not _local_data:
        _load_local_data()
    return _local_data[name]


def _users():
    return _db["users"]


def _documents():
    return _db["documents"]


def _quizzes():
    return _db["quizzes"]


def _submissions():
    return _db["quiz_submissions"]


async def connect_db() -> None:
    global _client, _db
    uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB", "teaching_assistant")

    if uri:
        import asyncio
        import logging

        try:
            _client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )
            await asyncio.wait_for(_client.admin.command("ping"), timeout=6)
            _db = _client[db_name]
            await asyncio.wait_for(_users().create_index("email", unique=True), timeout=5)
            return
        except Exception as exc:
            if _client:
                _client.close()
            _client = None
            _db = None
            logging.warning("MongoDB unavailable; using local file database: %s", exc)
    else:
        import logging

        logging.warning("MONGODB_URI is not set; using local file database")

    _load_local_data()


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


async def create_user(
    email: str,
    password: str,
    full_name: str,
    role: str,
) -> dict[str, Any]:
    doc = {
        "email": email.lower(),
        "password_hash": hash_password(password),
        "full_name": full_name,
        "role": role,
        "created_at": datetime.now(timezone.utc),
    }
    if _db is None:
        doc["_id"] = ObjectId()
        _local_collection("users").append(doc)
        _save_local_data()
        return doc

    result = await _users().insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    if _db is None:
        normalized = email.lower()
        return next(
            (user for user in _local_collection("users") if user["email"] == normalized),
            None,
        )
    return await _users().find_one({"email": email.lower()})


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    if not ObjectId.is_valid(user_id):
        return None
    if _db is None:
        return next(
            (user for user in _local_collection("users") if str(user["_id"]) == user_id),
            None,
        )
    return await _users().find_one({"_id": ObjectId(user_id)})


async def save_document(
    teacher_id: str,
    filename: str,
    pinecone_namespace: str,
    chunk_count: int,
) -> str:
    doc = {
        "teacher_id": teacher_id,
        "filename": filename,
        "pinecone_namespace": pinecone_namespace,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now(timezone.utc),
    }
    if _db is None:
        doc["_id"] = ObjectId()
        _local_collection("documents").append(doc)
        _save_local_data()
        return str(doc["_id"])

    result = await _documents().insert_one(doc)
    return str(result.inserted_id)


async def get_document(document_id: str) -> dict[str, Any] | None:
    if not ObjectId.is_valid(document_id):
        return None
    if _db is None:
        return next(
            (doc for doc in _local_collection("documents") if str(doc["_id"]) == document_id),
            None,
        )
    return await _documents().find_one({"_id": ObjectId(document_id)})


async def list_documents_for_teacher(teacher_id: str) -> list[dict[str, Any]]:
    if _db is None:
        docs = [
            doc
            for doc in _local_collection("documents")
            if doc["teacher_id"] == teacher_id
        ]
        return sorted(docs, key=lambda doc: doc["uploaded_at"], reverse=True)[:100]

    cursor = _documents().find({"teacher_id": teacher_id}).sort("uploaded_at", -1)
    return await cursor.to_list(length=100)


async def list_all_documents() -> list[dict[str, Any]]:
    if _db is None:
        docs = list(_local_collection("documents"))
        return sorted(docs, key=lambda doc: doc["uploaded_at"], reverse=True)[:200]

    cursor = _documents().find().sort("uploaded_at", -1)
    return await cursor.to_list(length=200)


async def save_quiz(
    student_id: str,
    document_id: str,
    questions: list[dict[str, Any]],
) -> str:
    doc = {
        "student_id": student_id,
        "document_id": document_id,
        "questions": questions,
        "created_at": datetime.now(timezone.utc),
    }
    if _db is None:
        doc["_id"] = ObjectId()
        _local_collection("quizzes").append(doc)
        _save_local_data()
        return str(doc["_id"])

    result = await _quizzes().insert_one(doc)
    return str(result.inserted_id)


async def get_quiz(quiz_id: str) -> dict[str, Any] | None:
    if not ObjectId.is_valid(quiz_id):
        return None
    if _db is None:
        return next(
            (quiz for quiz in _local_collection("quizzes") if str(quiz["_id"]) == quiz_id),
            None,
        )
    return await _quizzes().find_one({"_id": ObjectId(quiz_id)})


async def save_quiz_submission(
    student_id: str,
    quiz_id: str,
    score: float,
    correct: int,
    total: int,
) -> str:
    doc = {
        "student_id": student_id,
        "quiz_id": quiz_id,
        "score": score,
        "correct": correct,
        "total": total,
        "submitted_at": datetime.now(timezone.utc),
    }
    if _db is None:
        doc["_id"] = ObjectId()
        _local_collection("quiz_submissions").append(doc)
        _save_local_data()
        return str(doc["_id"])

    result = await _submissions().insert_one(doc)
    return str(result.inserted_id)


async def get_student_scores() -> list[dict[str, Any]]:
    if _db is None:
        grouped: dict[str, dict[str, Any]] = {}
        for submission in _local_collection("quiz_submissions"):
            student_id = submission["student_id"]
            row = grouped.setdefault(
                student_id,
                {
                    "student_id": student_id,
                    "quizzes_taken": 0,
                    "score_total": 0.0,
                    "last_quiz_at": None,
                },
            )
            row["quizzes_taken"] += 1
            row["score_total"] += submission["score"]
            submitted_at = submission.get("submitted_at")
            if row["last_quiz_at"] is None or submitted_at > row["last_quiz_at"]:
                row["last_quiz_at"] = submitted_at

        results: list[dict[str, Any]] = []
        for row in grouped.values():
            user = await get_user_by_id(row["student_id"])
            if not user or user.get("role") != "student":
                continue
            results.append(
                {
                    "student_id": row["student_id"],
                    "full_name": user.get("full_name", "Unknown"),
                    "email": user.get("email", ""),
                    "quizzes_taken": row["quizzes_taken"],
                    "average_score": round(row["score_total"] / row["quizzes_taken"], 2),
                    "last_quiz_at": row.get("last_quiz_at"),
                }
            )
        return sorted(results, key=lambda x: x["average_score"], reverse=True)

    pipeline = [
        {
            "$group": {
                "_id": "$student_id",
                "quizzes_taken": {"$sum": 1},
                "average_score": {"$avg": "$score"},
                "last_quiz_at": {"$max": "$submitted_at"},
            }
        }
    ]
    aggregates = await _submissions().aggregate(pipeline).to_list(length=500)
    results: list[dict[str, Any]] = []

    for agg in aggregates:
        user = await get_user_by_id(agg["_id"])
        if not user or user.get("role") != "student":
            continue
        results.append(
            {
                "student_id": agg["_id"],
                "full_name": user.get("full_name", "Unknown"),
                "email": user.get("email", ""),
                "quizzes_taken": agg["quizzes_taken"],
                "average_score": round(agg["average_score"], 2),
                "last_quiz_at": agg.get("last_quiz_at"),
            }
        )
    return sorted(results, key=lambda x: x["average_score"], reverse=True)
