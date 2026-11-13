# DocuAgent – AI-Powered Document Analysis & Question Answering

DocuAgent is an AI-powered document analysis application that allows users to upload documents, ask questions, generate summaries, and retrieve relevant information using **Retrieval-Augmented Generation (RAG)**.

Built with Python, LangGraph, ChromaDB, Hugging Face Transformers, and Ollama, DocuAgent provides an interactive workspace for querying documents using a locally running Large Language Model (LLM).

## Key Features

- **Multi-Format Document Upload:** Upload and process PDF, DOCX, and TXT files.
- **AI-Powered Question Answering:** Ask natural-language questions and receive context-aware answers from uploaded documents.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document chunks to ground AI-generated responses in source content.
- **Semantic Search:** Find relevant information using Hugging Face sentence embeddings.
- **Intelligent Reranking:** Improve retrieval relevance using a Cross-Encoder model.
- **Document Summarization:** Generate concise summaries of document content.
- **Multi-Document Analysis:** Select and query multiple documents within the same workspace.
- **Conversation History:** Maintain context across follow-up questions.
- **Source Citations:** Display relevant document sources and page references where available.
- **Report Export:** Export AI-generated conversations and answers as PDF and DOCX reports.
- **Local LLM Integration:** Generate responses using Ollama without requiring a cloud-based LLM API.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Llama 3.2 |
| LLM Runtime | Ollama |
| AI Workflow | LangGraph |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranking | Cross-Encoder |
| Reranker Model | ms-marco-MiniLM-L-6-v2 |
| Vector Database | ChromaDB |
| Document Processing | PyMuPDF, python-docx |
| Report Generation | ReportLab, python-docx |

## System Architecture

```text
              User
               |
               v
        Streamlit Interface
               |
               v
       Document Upload
               |
               v
       Document Processing
        (PDF / DOCX / TXT)
               |
               v
         Text Chunking
               |
               v
       Hugging Face Embeddings
               |
               v
          ChromaDB
        (Vector Storage)
               |
               v
       User Query / Question
               |
               v
        LangGraph Workflow
               |
               v
       Semantic Retrieval
               |
               v
       Cross-Encoder Reranking
               |
               v
       Context Construction
               |
               v
        Ollama (Llama 3.2)
               |
               v
        AI-Generated Answer
               |
               v
       Sources & Citations
```

## Project Structure

```text
DocuAgent/
│
├── app.py                  # Streamlit application
├── agent.py                # LangGraph AI agent workflow
├── document_processor.py   # Document extraction
├── vector_store.py         # Chunking, embeddings and retrieval
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignored files
├── chroma_db/               # Persistent vector database (generated)
├── venv/                    # Python virtual environment
└── README.md                # Project documentation
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Luff12/DocuAgent.git
cd DocuAgent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Download Ollama from:

https://ollama.com/

Pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before starting the application.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## How to Use

1. Launch the DocuAgent application.
2. Upload one or more PDF, DOCX, or TXT documents.
3. Wait for document processing and indexing to complete.
4. Select the documents you want to query.
5. Enter a question in the AI Workspace.
6. Receive an AI-generated answer based on the retrieved document content.
7. Review the supporting sources and citations.
8. Export your results as PDF or DOCX reports.

## AI Models

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

Used to convert document chunks and user questions into vector representations for semantic retrieval.

**Reranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

Used to rerank retrieved document chunks according to their relevance to the user's question.

**Language Model:** `llama3.2`

Used through Ollama to generate natural-language responses based on retrieved document context.

## Future Enhancements

- OCR support for scanned PDF documents.
- Support for additional document formats.
- Advanced document comparison and analytics.
- Improved retrieval evaluation and answer quality.
- Document-level access control.
- Enhanced conversational memory.
- Deployment to a cloud platform.

## Project Goals

The goal of DocuAgent is to make document analysis faster and more accessible by combining semantic search, RAG, and locally hosted language models into a single interactive application.

## Author

**Shibin Mathew**

- GitHub: [@Luff12](https://github.com/Luff12)
- Project Repository: [DocuAgent](https://github.com/Luff12/DocuAgent)

---

⭐ If you find this project interesting, consider giving the repository a star!
