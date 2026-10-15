import json
import os
import re
from collections import Counter

import google.generativeai as genai
from groq import AsyncGroq

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.0-flash-lite"

_groq: AsyncGroq | None = None
_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "both",
    "can",
    "could",
    "does",
    "each",
    "from",
    "had",
    "has",
    "have",
    "into",
    "its",
    "may",
    "more",
    "must",
    "not",
    "only",
    "other",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "through",
    "under",
    "was",
    "were",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def _groq_configured() -> bool:
    key = os.environ.get("GROQ_API_KEY", "")
    return bool(key) and not key.startswith("your-") and key != "gsk_your_key"


def _get_groq() -> AsyncGroq:
    global _groq
    if _groq is None:
        _groq = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _groq


def _configure_gemini() -> None:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


async def _groq_chat(system: str, user: str) -> str:
    client = _get_groq()
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


async def _gemini_chat(system: str, user: str) -> str:
    import asyncio

    _configure_gemini()
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=system)

    def _run():
        return model.generate_content(user)

    response = await asyncio.to_thread(_run)
    return response.text or ""


async def generate_text(system: str, user: str) -> str:
    errors: list[str] = []

    if _groq_configured():
        try:
            return await _groq_chat(system, user)
        except Exception as exc:
            errors.append(f"Groq: {exc}")

    try:
        return await _gemini_chat(system, user)
    except Exception as exc:
        errors.append(f"Gemini: {exc}")

    if not _groq_configured():
        gemini_err = errors[-1] if errors else "unknown"
        raise RuntimeError(
            "GROQ_API_KEY is missing or still a placeholder in .env. "
            "Get a free key at https://console.groq.com (starts with gsk_). "
            f"Gemini also failed: {gemini_err}"
        )
    raise RuntimeError(" / ".join(errors))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _sentences(context_chunks: list[str]) -> list[str]:
    text = _clean_text(" ".join(context_chunks))
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [_clean_text(part) for part in parts if len(_clean_text(part)) >= 35]
    if sentences:
        return sentences[:80]

    words = text.split()
    return [" ".join(words[i : i + 28]) for i in range(0, len(words), 28) if words[i : i + 28]]


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
    return [word.lower() for word in words if word.lower() not in _STOP_WORDS]


def _keyword_matches(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", text)
    return [word for word in words if word.lower() not in _STOP_WORDS]


def _source_snippets(context_chunks: list[str]) -> list[str]:
    return [c[:120] + "..." if len(c) > 120 else c for c in context_chunks[:3]]


def _fallback_answer(question: str, context_chunks: list[str]) -> tuple[str, list[str]]:
    sentences = _sentences(context_chunks)
    if not sentences:
        return (
            "I could not find readable text for this PDF. Try uploading a text-based PDF instead of a scanned image.",
            [],
        )

    question_words = set(_keywords(question))
    scored: list[tuple[int, int, str]] = []
    for sentence in sentences:
        overlap = len(question_words.intersection(_keywords(sentence)))
        if overlap:
            scored.append((overlap, -len(sentence), sentence))

    if scored:
        scored.sort(reverse=True)
        selected = [item[2] for item in scored[:2]]
    else:
        selected = sentences[:2]

    answer = "Based on the uploaded PDF: " + " ".join(selected)
    return answer, _source_snippets(context_chunks)


def _option_set(correct: str, distractors: list[str], index: int) -> tuple[list[str], int]:
    unique: list[str] = []
    for option in [correct, *distractors]:
        clean = _clean_text(option)
        if clean and clean.lower() not in {item.lower() for item in unique}:
            unique.append(clean)
    while len(unique) < 4:
        unique.append(
            [
                "The PDF does not support this statement.",
                "This topic is not mentioned in the uploaded material.",
                "The material says the opposite of this option.",
            ][len(unique) % 3]
        )

    options = unique[:4]
    shift = index % 4
    rotated = options[shift:] + options[:shift]
    return rotated, rotated.index(correct)


def _fallback_quiz_questions(context_chunks: list[str], num_questions: int) -> list[dict]:
    sentences = _sentences(context_chunks)
    if not sentences:
        raise ValueError("No readable PDF text found to generate a quiz")

    all_terms = []
    for sentence in sentences:
        all_terms.extend(_keyword_matches(sentence))
    common_terms = [term for term, _ in Counter(term.lower() for term in all_terms).most_common()]

    questions: list[dict] = []
    used_terms: set[str] = set()
    for sentence in sentences:
        terms = [
            term
            for term in _keyword_matches(sentence)
            if term.lower() not in used_terms and len(term) >= 5
        ]
        if not terms:
            continue

        answer = terms[0]
        used_terms.add(answer.lower())
        blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", sentence, count=1, flags=re.IGNORECASE)
        distractors = [
            term
            for term in common_terms
            if term != answer.lower() and term not in used_terms
        ][:3]
        options, correct_index = _option_set(answer, distractors, len(questions))
        questions.append(
            {
                "question_id": f"q{len(questions) + 1}",
                "question": f"Complete this statement from the PDF: {blanked}",
                "options": options,
                "correct_index": correct_index,
            }
        )
        if len(questions) >= num_questions:
            return questions

    for sentence in sentences:
        options, correct_index = _option_set(
            sentence[:220],
            [
                "The uploaded PDF contains no useful course content.",
                "The uploaded PDF says this subject is unrelated to the lesson.",
                "The uploaded PDF says students should ignore this concept.",
            ],
            len(questions),
        )
        questions.append(
            {
                "question_id": f"q{len(questions) + 1}",
                "question": "Which statement is supported by the uploaded PDF?",
                "options": options,
                "correct_index": correct_index,
            }
        )
        if len(questions) >= num_questions:
            break

    return questions


async def answer_with_context(question: str, context_chunks: list[str]) -> tuple[str, list[str]]:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context available."
    system = (
        "You are an AI teaching assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say you do not have enough information "
        "from the course materials. Be clear, educational, and concise."
    )
    user = f"Context:\n{context}\n\nStudent question: {question}"
    try:
        answer = await generate_text(system, user)
        return answer, _source_snippets(context_chunks)
    except RuntimeError:
        return _fallback_answer(question, context_chunks)


async def generate_quiz_questions(
    context_chunks: list[str],
    num_questions: int,
) -> list[dict]:
    context = "\n\n".join(context_chunks[:12])
    system = (
        "You generate multiple-choice quiz questions from educational content. "
        "Return ONLY valid JSON: an array of objects with keys: "
        "question (string), options (array of 4 strings), correct_index (0-3 integer)."
    )
    user = (
        f"Create exactly {num_questions} multiple-choice questions from this content:\n\n"
        f"{context}\n\n"
        "Return JSON array only, no markdown."
    )
    try:
        raw = await generate_text(system, user)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", raw)
            if not match:
                raise ValueError("LLM did not return valid quiz JSON") from None
            data = json.loads(match.group())
    except (RuntimeError, ValueError):
        return _fallback_quiz_questions(context_chunks, num_questions)

    questions: list[dict] = []
    for i, item in enumerate(data[:num_questions]):
        questions.append(
            {
                "question_id": f"q{i + 1}",
                "question": item["question"],
                "options": item["options"][:4],
                "correct_index": int(item["correct_index"]),
            }
        )
    return questions
