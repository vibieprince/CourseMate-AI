# 🚀 CourseMate AI

### AI-Powered PDF Learning Assistant using RAG Architecture

---

## 👋 About the Project

**CourseMate AI** is a full-stack Generative AI application that transforms static PDF documents into an **interactive, conversational learning experience**.

Instead of manually searching through notes or textbooks, users can:

* Upload any PDF 📄
* Ask questions in natural language 💬
* Get **context-aware, accurate answers** grounded strictly in the document

---

## 🎯 Why I Built This

As a Data Science student, I realized:

> Most learning resources are static, time-consuming, and inefficient to navigate.

I wanted to solve a real problem:

* Students waste time searching PDFs
* Traditional keyword search lacks semantic understanding
* LLMs often hallucinate without context

👉 **CourseMate AI bridges this gap using Retrieval-Augmented Generation (RAG)**

---

## 🧠 What This Project Demonstrates

This project reflects my ability to:

### 🔹 Apply Core Data Science Concepts

* Vector embeddings
* Semantic similarity search
* Information retrieval

### 🔹 Work with Modern GenAI Stack

* Building real-world RAG pipelines
* Integrating LLMs into applications
* Handling hallucination via grounded context

### 🔹 Engineer End-to-End Systems

* Backend API development
* Frontend UI/UX design
* Real-time interaction systems

---

## ⚙️ Tech Stack

### 🧠 AI / ML Layer

* **LangChain** → Orchestrates RAG pipeline
* **ChromaDB** → Vector database for semantic retrieval
* **HuggingFace Embeddings** → Converts text into vectors
* **Mistral AI** → LLM for generating answers

### ⚡ Backend

* **FastAPI** → High-performance API handling
* Session-based vector DB creation

### 🎨 Frontend

* **HTML, CSS, JavaScript**
* Responsive UI (ChatGPT-style interaction)

---

## 🔄 How It Works (RAG Pipeline)

```text
User uploads PDF
        ↓
Text is extracted & split into chunks
        ↓
Embeddings generated (vector representation)
        ↓
Stored in ChromaDB
        ↓
User asks a question
        ↓
Retriever finds relevant chunks
        ↓
Mistral LLM generates grounded answer
```

---

## ✨ Key Features

* 📄 Upload any PDF document
* 💬 Ask questions in natural language
* 🎯 Context-aware answers (no hallucination)
* ⚡ Fast semantic search
* 🧠 MMR-based retrieval (diversity + relevance)
* 🔄 Session-based document handling

---

## 🧩 System Architecture

```text
Frontend (UI)
     ↓
FastAPI Backend
     ↓
Chroma Vector DB
     ↓
Retriever (MMR)
     ↓
Mistral LLM
     ↓
Response to User
```

---

## 📌 My Learning Outcomes

Through building this project, I:

* Understood **how LLMs actually work beyond APIs**
* Learned to control hallucination using **retrieval-based grounding**
* Implemented **vector databases and similarity search**
* Built a **production-style AI system**, not just a notebook model
* Improved my **system design thinking in AI applications**

---

## 🚧 Challenges Faced

* Managing embedding consistency across vector DB
* Handling session-based document storage
* Preventing LLM hallucinations
* Designing a responsive AI chat UI
* Integrating multiple AI services reliably

---

## 🚀 Future Improvements

* 🔐 User authentication & multi-user support
* ☁️ Cloud vector DB (Pinecone / Weaviate)
* 📊 Analytics dashboard
* 💬 Streaming responses (ChatGPT-like typing)
* 📚 Multi-document querying

---

## 👨‍💻 About Me

**Prince Singh**
🎓 Computer Science (Data Science Specialization)
💡 Interested in AI, Data Science & Real-world ML Systems

---

## 📬 Let's Connect

* GitHub: *[https://github.com/vibieprince]*
* LinkedIn: *[https://linkedin.com/in/vibieprince]*

---

## ⭐ Final Thought

> This project is not just about using AI —
> it’s about **engineering intelligent systems that solve real problems.**

---

⭐ If you find this interesting, feel free to explore and give feedback!
