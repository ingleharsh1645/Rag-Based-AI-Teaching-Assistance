import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

# 1. Page Config
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

# 2. API Key (Use Streamlit Secrets for security!)
# Add GOOGLE_API_KEY to your Streamlit Cloud Secrets dashboard
api_key = st.secrets.get("GOOGLE_API_KEY") 

# 3. RAG Logic (Directly inside the app)
def process_docs(files):
    all_docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            loader = PyPDFLoader(tmp.name)
            all_docs.extend(loader.load())
            os.unlink(tmp.name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(all_docs)

# 4. App UI
st.title("📚 AI Teaching Assistant")
uploaded_files = st.file_uploader("Upload course materials (PDF)", accept_multiple_files=True)

if uploaded_files and api_key:
    if st.button("Index Documents"):
        docs = process_docs(uploaded_files)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        st.session_state.db = FAISS.from_documents(docs, embeddings)
        st.success("Indexed!")

# ... (Add Chat Interface logic here)
