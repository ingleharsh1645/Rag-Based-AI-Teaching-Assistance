import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

# Page Configuration
st.set_page_config(page_title="AI Teaching Assistant", page_icon="📚", layout="wide")

# Sidebar - Settings and Document Upload
st.sidebar.title("📚 AI Teaching Assistant")
st.sidebar.markdown("---")

# Use Streamlit Secrets for the API Key
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.error("API Key not found in secrets! Please add it in Streamlit Cloud.")
    st.stop()

uploaded_files = st.sidebar.file_uploader("Upload Course PDFs", accept_multiple_files=True, type="pdf")

# Session State for memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Indexing Documents
if st.sidebar.button("Process & Index Documents"):
    if uploaded_files:
        with st.spinner("Processing..."):
            all_docs = []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    all_docs.extend(loader.load())
                    os.unlink(tmp.name)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = splitter.split_documents(all_docs)
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
            st.sidebar.success("Documents Indexed!")
    else:
        st.sidebar.warning("Please upload files first.")

# Chat Interface
st.title("🎓 AI Teaching Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your course..."):
    if not st.session_state.vector_store:
        st.warning("Please index documents first!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=st.session_state.vector_store.as_retriever()
        )
        response = chain({"question": prompt, "chat_history": []})["answer"]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

