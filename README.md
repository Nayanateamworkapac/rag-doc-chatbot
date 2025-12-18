# RAG Document Chatbot

This project demonstrates a **Retrieval-Augmented Generation (RAG)** style
document question-answering system.

The chatbot answers user queries by retrieving the most relevant content
from provided documents using **semantic search** instead of generating
ungrounded responses.

## 🚀 Key Features
- Document ingestion and chunking
- Semantic embeddings using HuggingFace models
- Vector similarity search with FAISS
- Relevance score thresholding
- Interactive command-line Q&A interface
- Reduces hallucinations by grounding answers in documents

## 🛠 Tech Stack
- Python
- LangChain
- HuggingFace Embeddings
- FAISS Vector Database
- NLP

## 🧠 How It Works
1. Documents are split into semantic chunks
2. Embeddings are generated for each chunk
3. Stored in a FAISS vector database
4. User queries are embedded and compared
5. Most relevant document chunks are returned as answers

## ▶ How to Run
```bash
pip install -r requirements.txt
python ingest.py
python query.py
