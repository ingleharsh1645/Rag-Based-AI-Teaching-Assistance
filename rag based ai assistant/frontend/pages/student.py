import os
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from frontend.utils import api_client

st.set_page_config(page_title="Student Dashboard", layout="wide")

if not st.session_state.get("token"):
    st.warning("Please log in from the home page.")
    st.stop()

if st.session_state.get("role") != "student":
    st.error("Student access only.")
    st.stop()

st.title("🎓 Student Dashboard")

try:
    documents = api_client.list_documents(st.session_state.token)
except api_client.APIError as e:
    st.error(str(e))
    st.stop()

if not documents:
    st.info("No course materials yet. Ask your teacher to upload a PDF.")
    st.stop()

doc_options = {d["filename"]: d["document_id"] for d in documents}
selected_name = st.selectbox("Course material", list(doc_options.keys()))
document_id = doc_options[selected_name]

tab_chat, tab_quiz = st.tabs(["💬 Chat", "📝 Quiz"])

with tab_chat:
    st.subheader("Ask the teaching assistant")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(s)

    prompt = st.chat_input("Ask a question about the material...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = api_client.chat(
                    st.session_state.token,
                    prompt,
                    document_id,
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )
                st.rerun()
            except api_client.APIError as e:
                st.error(str(e))

with tab_quiz:
    st.subheader("Practice quiz")
    num_questions = st.slider("Number of questions", 1, 10, 3)
    st.caption("Use 1–3 questions if the PDF has little text (few chunks indexed).")

    if st.button("Generate quiz", type="primary"):
        with st.spinner("Building quiz from course content..."):
            try:
                quiz = api_client.generate_quiz(
                    st.session_state.token,
                    document_id,
                    num_questions,
                )
                st.session_state.quiz_data = quiz
                st.session_state.quiz_answers = {}
                st.session_state.quiz_result = None
            except api_client.APIError as e:
                st.error(str(e))

    quiz = st.session_state.get("quiz_data")
    if quiz:
        st.markdown(f"**Quiz ID:** `{quiz['quiz_id']}`")
        for q in quiz["questions"]:
            st.markdown(f"**{q['question_id']}.** {q['question']}")
            choice = st.radio(
                "Choose an answer",
                q["options"],
                key=f"quiz_{quiz['quiz_id']}_{q['question_id']}",
                label_visibility="collapsed",
            )
            idx = q["options"].index(choice)
            st.session_state.quiz_answers[q["question_id"]] = idx

        if st.button("Submit quiz"):
            answers = [
                {"question_id": qid, "selected_index": idx}
                for qid, idx in st.session_state.quiz_answers.items()
            ]
            try:
                result = api_client.submit_quiz(
                    st.session_state.token,
                    quiz["quiz_id"],
                    answers,
                )
                st.session_state.quiz_result = result
                st.success(result["feedback"])
                st.metric("Score", f"{result['score']}%")
                st.write(f"Correct: **{result['correct']}** / **{result['total']}**")
            except api_client.APIError as e:
                st.error(str(e))

        result = st.session_state.get("quiz_result")
        if result and result.get("quiz_id") == quiz["quiz_id"]:
            st.divider()
            st.subheader("Answer review")
            for item in result.get("results", []):
                selected = item.get("selected_index")
                selected_answer = (
                    item["options"][selected]
                    if selected is not None and 0 <= selected < len(item["options"])
                    else "No answer selected"
                )
                status = "Correct" if item["is_correct"] else "Incorrect"
                st.markdown(f"**{item['question_id']}. {status}**")
                st.write(item["question"])
                st.write(f"Your answer: {selected_answer}")
                st.write(f"Correct answer: **{item['correct_answer']}**")
