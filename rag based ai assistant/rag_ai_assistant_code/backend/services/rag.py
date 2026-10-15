import os
import uuid
import hashlib
import json
import math
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 150
CHUNK_OVERLAP = 30
TOP_K = 5

_encoder = None
_pinecone: Pinecone | None = None
_index = None
_pinecone_unavailable = False
_local_vectors: dict[str, list[dict[str, Any]]] | None = None


def _local_vector_path() -> Path:
    return Path(os.environ.get("LOCAL_VECTOR_PATH", "local_vectors.json"))


def _load_local_vectors() -> dict[str, list[dict[str, Any]]]:
    global _local_vectors
    if _local_vectors is not None:
        return _local_vectors

    path = _local_vector_path()
    if not path.exists():
        _local_vectors = {}
        return _local_vectors

    _local_vectors = json.loads(path.read_text(encoding="utf-8"))
    return _local_vectors


def _save_local_vectors() -> None:
    if _local_vectors is None:
        return
    _local_vector_path().write_text(
        json.dumps(_local_vectors, indent=2),
        encoding="utf-8",
    )


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            import logging

            logging.warning(
                "sentence-transformers is not installed; using lightweight local embeddings: %s",
                exc,
            )
            _encoder = False
            return _encoder
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def _get_index():
    global _pinecone, _index, _pinecone_unavailable
    if _encoder is False:
        return None

    if _pinecone_unavailable:
        return None

    if _index is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not api_key or not index_name:
            _pinecone_unavailable = True
            return None

        try:
            _pinecone = Pinecone(api_key=api_key)
            existing = [idx.name for idx in _pinecone.list_indexes()]
        except Exception as exc:
            import logging

            logging.warning("Pinecone unavailable; using local vector store: %s", exc)
            _pinecone_unavailable = True
            return None

        cloud = os.environ.get("PINECONE_CLOUD", "aws")
        region = os.environ.get("PINECONE_REGION", "us-east-1")

        try:
            if index_name not in existing:
                _pinecone.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
            _index = _pinecone.Index(index_name)
        except Exception as exc:
            import logging

            logging.warning("Could not initialize Pinecone index; using local vector store: %s", exc)
            _pinecone_unavailable = True
            return None
    return _index


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def embed_texts(texts: list[str]) -> list[list[float]]:
    encoder = _get_encoder()
    if encoder:
        vectors = encoder.encode(texts, show_progress_bar=False)
        return vectors.tolist()
    return [_hash_embed(text) for text in texts]


def _hash_embed(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _tokens(text: str) -> set[str]:
    stop_words = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "that",
        "the",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def _local_upsert(namespace: str, records: list[dict[str, Any]]) -> None:
    vectors = _load_local_vectors()
    vectors[namespace] = records
    _save_local_vectors()


def _local_query(query: str, query_vec: list[float], namespace: str, top_k: int) -> list[dict[str, Any]]:
    records = _load_local_vectors().get(namespace, [])
    query_tokens = _tokens(query)
    scored = []
    for record in records:
        text = (record.get("metadata") or {}).get("text", "")
        overlap = len(query_tokens.intersection(_tokens(text)))
        phrase_bonus = 1 if query.strip().lower() in text.lower() else 0
        score = (
            _cosine_similarity(query_vec, record.get("values", []))
            + (overlap * 10)
            + (phrase_bonus * 20)
        )
        scored.append({**record, "score": score})
    scored.sort(key=lambda record: record["score"], reverse=True)
    return scored[:top_k]


async def index_pdf(
    file_bytes: bytes,
    namespace: str,
) -> tuple[int, list[str]]:
    text = extract_pdf_text(file_bytes)
    chunks = _chunk_text(text)
    if not chunks:
        return 0, []

    vectors = embed_texts(chunks)
    records = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        records.append(
            {
                "id": f"{namespace}-{uuid.uuid4().hex[:12]}",
                "values": vec,
                "metadata": {"text": chunk[:1000], "chunk_index": i},
            }
        )

    index = _get_index()
    if index is None:
        _local_upsert(namespace, records)
    else:
        try:
            batch_size = 100
            for i in range(0, len(records), batch_size):
                index.upsert(vectors=records[i : i + batch_size], namespace=namespace)
        except Exception as exc:
            import logging

            logging.warning("Pinecone upsert failed; storing vectors locally: %s", exc)
            _local_upsert(namespace, records)

    return len(chunks), chunks


async def retrieve_context(
    query: str,
    namespace: str,
    top_k: int = TOP_K,
) -> list[str]:
    query_vec = embed_texts([query])[0]
    index = _get_index()
    if index is None:
        matches = _local_query(query, query_vec, namespace, top_k)
    else:
        try:
            result = index.query(
                vector=query_vec,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
            matches = result.get("matches", [])
        except Exception as exc:
            import logging

            logging.warning("Pinecone query failed; reading vectors locally: %s", exc)
            matches = _local_query(query, query_vec, namespace, top_k)

    chunks: list[str] = []
    seen: set[str] = set()
    for match in matches:
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        if text and text not in seen:
            seen.add(text)
            chunks.append(text)
    return chunks


async def retrieve_all_chunks(namespace: str, max_chunks: int = 20) -> list[str]:
    """Fetch all indexed chunks in a namespace (for small PDFs)."""
    query_vec = embed_texts(["course material content text"])[0]
    index = _get_index()
    if index is None:
        matches = _local_query("course material content text", query_vec, namespace, max_chunks)
    else:
        try:
            result = index.query(
                vector=query_vec,
                top_k=max_chunks,
                namespace=namespace,
                include_metadata=True,
            )
            matches = result.get("matches", [])
        except Exception as exc:
            import logging

            logging.warning("Pinecone query failed; reading vectors locally: %s", exc)
            matches = _local_query("course material content text", query_vec, namespace, max_chunks)

    chunks: list[str] = []
    seen: set[str] = set()
    for match in matches:
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        if text and text not in seen:
            seen.add(text)
            chunks.append(text)
    chunks.sort(key=lambda t: t[:50])
    return chunks
