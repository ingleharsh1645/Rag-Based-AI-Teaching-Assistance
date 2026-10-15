import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.utils import api_client

load_dotenv()

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "token" not in st.session_state:
    st.session_state.token = None
if "role" not in st.session_state:
    st.session_state.role = None
if "full_name" not in st.session_state:
    st.session_state.full_name = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None


def logout():
    st.session_state.token = None
    st.session_state.role = None
    st.session_state.full_name = None
    st.session_state.user_id = None
    st.session_state.pop("quiz_data", None)
    st.session_state.pop("quiz_answers", None)


def main():
    st.title("📚 AI Teaching Assistant")
    st.caption("Modular RAG platform for teachers and students")

    if not api_client.backend_is_running():
        st.error(
            f"Backend is not running at **{api_client.BASE_URL}**. "
            "Start it first in a separate terminal:"
        )
        st.code(
            "python -m uvicorn backend.main:app --reload --port 8000",
            language="powershell",
        )
        st.stop()

    if st.session_state.token:
        st.sidebar.success(f"Signed in as **{st.session_state.full_name}**")
        st.sidebar.write(f"Role: **{st.session_state.role.title()}**")
        if st.sidebar.button("Log out"):
            logout()
            st.rerun()
        st.info("Use the sidebar pages to open your dashboard (Teacher or Student).")
        return

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                try:
                    data = api_client.login(email, password)
                    st.session_state.token = data["access_token"]
                    st.session_state.role = data["role"]
                    st.session_state.full_name = data["full_name"]
                    st.session_state.user_id = data["user_id"]
                    st.success("Logged in successfully.")
                    st.rerun()
                except api_client.APIError as e:
                    st.error(str(e))

    with tab_register:
        with st.form("register_form"):
            full_name = st.text_input("Full name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_password")
            role = st.selectbox("Role", ["teacher", "student"])
            submitted = st.form_submit_button("Create account", type="primary")
            if submitted:
                try:
                    data = api_client.register(email, password, full_name, role)
                    st.session_state.token = data["access_token"]
                    st.session_state.role = data["role"]
                    st.session_state.full_name = data["full_name"]
                    st.session_state.user_id = data["user_id"]
                    st.success("Account created. You are now logged in.")
                    st.rerun()
                except api_client.APIError as e:
                    st.error(str(e))


if __name__ == "__main__":
    main()
