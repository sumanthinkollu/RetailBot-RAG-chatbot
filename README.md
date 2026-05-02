# 🛍️ RetailBot AI — RAG-Powered Retail Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for retail product Q&A.  
Built with Python, LangChain, FAISS, HuggingFace Embeddings, and Streamlit.

---

## 📐 Architecture

```
User Question
     │
     ▼
[Streamlit UI]
     │
     ▼
[Embedding Model]  ←── all-MiniLM-L6-v2 (local, free)
     │  converts query to vector
     ▼
[FAISS Vector Store]
     │  similarity search → top 3 matching product chunks
     ▼
[Prompt Builder]
     │  context + question → prompt
     ▼
[LLM]  ←── Ollama (local) / OpenAI / Groq
     │
     ▼
[Grounded Answer] → Streamlit UI
```

---

## 🗂️ Project Structure

```
retail-rag-chatbot/
│
├── data/
│   └── products.json       ← 10 dummy retail products (knowledge base)
│
├── vectorstore/            ← created after running ingest.py
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py               ← Sprint 1: Build vector store
├── rag_pipeline.py         ← Sprint 2: RAG retrieval + LLM generation
├── app.py                  ← Sprint 3: Streamlit chat UI
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup Guide (Step by Step)

### Step 1 — Clone / Extract the project

```bash
cd retail-rag-chatbot
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Activate it:
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First install downloads the embedding model (~90MB). This is one-time only.

### Step 4 — Choose your LLM

#### Option A: Ollama (FREE, runs locally — recommended)

```bash
# Install Ollama from https://ollama.com
# Then pull a model:
ollama pull llama3        # 4.7GB — best quality
ollama pull mistral       # 4.1GB — good alternative
ollama pull phi3          # 2.3GB — fastest, lightweight

# Start Ollama server (keep this running in background):
ollama serve
```

#### Option B: Groq (FREE API, fastest cloud option)

```bash
# Get free API key at https://console.groq.com
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

pip install langchain-groq
```

#### Option C: OpenAI (paid, best quality)

```bash
# Get API key at https://platform.openai.com
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 5 — Build the vector store (ONE TIME)

```bash
python ingest.py
```

Expected output:
```
📦 Loading product data...
   Found 10 products.
📝 Converting to documents...
🔢 Generating embeddings...
💾 Storing in FAISS vector store...
✅ Done! Vector store saved to 'vectorstore/'
```

### Step 6 — Run the chatbot

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501** 🎉

---

## 💬 Example Questions to Try

| Question | What it tests |
|---|---|
| "Does SmartPhone X1 support 5G?" | Feature lookup |
| "Compare SmartPhone X1 and X2" | Multi-product comparison |
| "What is the return policy for laptops?" | Policy retrieval |
| "Which products are available under ₹30,000?" | Price filtering |
| "Is the LaptopPro Z5 in stock?" | Stock status |
| "What warranty does the SmartWatch have?" | Warranty lookup |
| "Tell me about noise cancelling earbuds" | Category search |
| "Who is the Prime Minister of India?" | Hallucination guard ✅ |

---

## 🧠 How RAG Works (Quick Explainer)

| Step | What Happens |
|---|---|
| **Ingest** | Product text → embeddings → stored in FAISS |
| **Query** | User question → converted to embedding vector |
| **Retrieve** | FAISS finds top-3 most similar product chunks |
| **Augment** | Retrieved context injected into prompt |
| **Generate** | LLM answers ONLY from that context |
| **Result** | Grounded, accurate answer with no hallucination |

---

## 🔧 Customization

### Add more products
Edit `data/products.json` and add entries following the same format, then re-run:
```bash
python ingest.py
```

### Change number of retrieved chunks
In `rag_pipeline.py`, change:
```python
search_kwargs={"k": 3}  # increase to 5 for broader retrieval
```

### Add PDF product brochures
Install `pypdf` and use LangChain's `PyPDFLoader` to load PDFs into documents before ingesting.

---

## 🎯 Evaluation Checklist (HCL Criteria)

- [x] RAG architecture with clear separation of retrieval & generation
- [x] Embeddings generated with sentence-transformers (FAISS)
- [x] Prompt engineering to prevent hallucination
- [x] Modular, clean Python code
- [x] Streamlit chat UI with conversation history
- [x] Source citation for answers
- [x] Error handling for empty/off-topic queries
- [x] End-to-end demo: Ask → Retrieve → Answer

---

## 🆘 Troubleshooting

| Problem | Solution |
|---|---|
| `Vector store not found` | Run `python ingest.py` first |
| `Ollama connection refused` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3` |
| `OpenAI API error` | Check your API key in `.env` |
| Slow first run | Embedding model downloads once (~90MB) |
