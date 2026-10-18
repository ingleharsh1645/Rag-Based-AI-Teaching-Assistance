import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

# Page Configuration
st.set_page_config(page_title="AI Teaching Assistant", page_icon="??", layout="wide")

# Sidebar - Settings and Document Upload
st.sidebar.title("?? AI Teaching Assistant")
st.sidebar.markdown("---")

# Use Streamlit Secrets for the API Key
api_key = st.secrets.get("GOOGLE_API_KEY")

if not api_key:
    st.sidebar.error("API Key not found in secrets! Please add it in Streamlit Cloud Settings -> Secrets.")
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
        with st.spinner("Processing and indexing documents..."):
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
            st.sidebar.success("Documents Indexed Successfully!")
    else:
        st.sidebar.warning("Please upload files first.")

# Chat Interface
st.title("?? AI Teaching Assistant")
st.markdown("Ask questions about your uploaded course materials and textbooks.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your course..."):
    if not st.session_state.vector_store:
        st.warning("Please upload and index documents first using the sidebar!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Direct Gemini LLM Call with Context
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
                
                prompt_text = f"""You are an expert AI teaching assistant. Answer the student's question accurately using only the provided context below. If the answer cannot be found in the context, say "I cannot find the answer in the provided course materials."

Context:
{context}

Question:
{prompt}

Answer:"""
                
                response = llm.invoke(prompt_text)
                answer_text = response.content
                
                st.markdown(answer_text)
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            except Exception as e:
                st.error(f"Error generating response: {e}")
