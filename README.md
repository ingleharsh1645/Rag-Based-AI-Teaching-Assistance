# RAG-Based AI Teaching Assistant

An intelligent, context-aware **Retrieval-Augmented Generation (RAG)** teaching assistant designed to help educators and students interact with course materials, syllabus documents, and textbooks seamlessly. This system extracts information from uploaded educational documents to provide accurate, source-backed answers to student queries.

## 🚀 Features

- **Document Ingestion:** Supports PDF, DOCX, Markdown, and TXT formats (syllabi, lecture notes, textbooks).
- **Smart Chunking & Embedding:** Splits long educational texts into meaningful contexts and converts them into dense vector embeddings.
- **Vector Search:** Fast and accurate semantic retrieval to find relevant course material matching a student's question.
- **Contextual Generation:** Generates human-like, accurate responses grounded strictly in the provided course context to minimize hallucinations.
- **Source Attribution:** Cites specific documents and sections so students can reference the exact page or source material.

## 🏗️ Architecture

1. **Data Prep:** Documents $\rightarrow$ Text Extraction $\rightarrow$ Text Chunking $\rightarrow$ Vector Embeddings.
2. **Storage:** Vector Database (e.g., ChromaDB, FAISS, or Pinecone).
3. **Retrieval & Generation:** User Query $\rightarrow$ Vector Similarity Search $\rightarrow$ Prompt Construction (Query + Context) $\rightarrow$ LLM $\rightarrow$ Synthesized Answer.

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **RAG Framework:** LangChain / LlamaIndex
- **LLM Provider:** OpenAI GPT-4 / Anthropic Claude / Ollama (for local deployment)
- **Vector Database:** ChromaDB / FAISS
- **UI Frontend:** Streamlit / Chainlit

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/rag-teaching-assistant.git
cd rag-teaching-assistant
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB_DIR=./data/chroma
```

### 5. Run the Application
Place your course documents inside the `data/source_docs/` folder, then launch the interface:
```bash
streamlit run app.py
```

## 📖 Usage File Structure
```text
├── data/
│   ├── source_docs/       # Drop course PDFs/textbooks here
│   └── chroma/            # Persistent vector storage
├── src/
│   ├── ingestion.py       # Document processing and embedding pipeline
│   └── retrieval.py       # Query processing and LLM generation
├── app.py                 # Streamlit web application
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
