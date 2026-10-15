import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from frontend.utils import api_client

st.set_page_config(page_title="Teacher Dashboard", layout="wide")

if not st.session_state.get("token"):
    st.warning("Please log in from the home page.")
    st.stop()

if st.session_state.get("role") != "teacher":
    st.error("Teacher access only.")
    st.stop()

st.title("👩‍🏫 Teacher Dashboard")
st.caption("Upload course PDFs and monitor student quiz performance")

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.subheader("Upload course material")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded and st.button("Index PDF", type="primary"):
        with st.spinner("Extracting text, embedding, and indexing in Pinecone..."):
            try:
                result = api_client.upload_pdf(
                    st.session_state.token,
                    uploaded.getvalue(),
                    uploaded.name,
                )
                st.success(
                    f"Indexed **{result['filename']}** — "
                    f"{result['chunks_indexed']} chunks stored."
                )
            except api_client.APIError as e:
                st.error(str(e))

with col_info:
    st.subheader("How it works")
    st.markdown(
        """
        1. Upload a PDF lecture or textbook chapter.
        2. Content is chunked and embedded (MiniLM, 384-dim).
        3. Vectors are stored in Pinecone per document namespace.
        4. Students can chat and take quizzes on indexed material.
        """
    )

st.divider()
st.subheader("Preview quiz")

try:
    documents = api_client.list_teacher_documents(st.session_state.token)
    if documents:
        doc_labels = {
            f"{doc['filename']} ({doc.get('chunk_count', 0)} chunks)": doc["document_id"]
            for doc in documents
        }
        selected_doc_label = st.selectbox("Course material", list(doc_labels.keys()))
        preview_questions = st.slider("Number of questions", 1, 10, 3, key="teacher_preview_count")

        if st.button("Generate preview quiz", type="primary"):
            with st.spinner("Generating quiz from the selected PDF..."):
                try:
                    st.session_state.teacher_preview_quiz = api_client.generate_teacher_preview_quiz(
                        st.session_state.token,
                        doc_labels[selected_doc_label],
                        preview_questions,
                    )
                except api_client.APIError as e:
                    st.error(str(e))

        quiz = st.session_state.get("teacher_preview_quiz")
        if quiz:
            for q in quiz["questions"]:
                st.markdown(f"**{q['question_id']}. {q['question']}**")
                for index, option in enumerate(q["options"], start=1):
                    st.write(f"{index}. {option}")
                if q.get("correct_answer"):
                    st.success(f"Correct answer: {q['correct_answer']}")
    else:
        st.info("Upload and index a PDF before generating a preview quiz.")
except api_client.APIError as e:
    st.error(str(e))

st.divider()
st.subheader("Student performance")

try:
    students = api_client.get_students(st.session_state.token)
    if students:
        df = pd.DataFrame(students)
        if "last_quiz_at" in df.columns:
            df["last_quiz_at"] = pd.to_datetime(df["last_quiz_at"], errors="coerce")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No quiz submissions yet.")
except api_client.APIError as e:
    st.error(str(e))
