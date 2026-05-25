# 🚀 CourseMate AI

### Enterprise-Grade Multimodal RAG & AI Learning Assistant

---

## 👋 About the Project

**CourseMate AI** is a full-stack, production-ready Generative AI application that transforms static documents (PDFs, DOC/DOCX) into an **interactive, conversational learning environment**.

Designed to solve the inefficiency of traditional document navigation, CourseMate AI goes beyond standard keyword searches. By leveraging **Retrieval-Augmented Generation (RAG)**, secure user isolation, and **Multi-Modal Vision OCR**, it reads, understands, and accurately answers questions grounded strictly in your academic or professional documents.

This marks the evolution from a prototype into a highly resilient, cloud-deployable Single Page Application (SPA).

---

## 🔥 v2.0 Architectural Advancements 

This project has been completely re-architected for production:

* **Dual-Engine Vision OCR Pipeline:** Natively processes scanned PDFs and images using **Gemini 2.5 Flash** as the primary vision extractor, with automatic failover to **Mistral Pixtral-12B** if rate limits are hit.
* **Secure User Identity & Isolation:** Integrated **Firebase Authentication** (Google Sign-In) with strict backend JWT validation. Every user gets a private, encrypted workspace and dedicated chat histories.
* **Persistent Chat History:** Migrated to **SQLAlchemy (PostgreSQL/SQLite)**. Conversations are saved sequentially, allowing users to return to previous document sessions seamlessly.
* **Ephemeral Data Security:** Uploaded documents and vector blocks are automatically wiped from server disks after 10 minutes of inactivity to ensure strict data privacy and prevent memory bloat.
* **Resilient LLM Failovers:** If the primary LLM (Mistral) experiences capacity limits (429 errors), the backend instantly falls back to Gemini, ensuring zero downtime for the end user.
* **SPA-Grade Frontend:** Completely refactored Vanilla JS interface with glassmorphic UI, responsive mobile sidebars, real-time typing effects, and asynchronous DOM updates (zero page reloads).

---

## ⚙️ Tech Stack

### 🧠 AI / ML Layer
* **LangChain** → Orchestrates the RAG and chunking pipelines
* **ChromaDB** → Local vector database for rapid semantic retrieval
* **HuggingFace** → `all-MiniLM-L6-v2` for dense vector embeddings
* **Mistral AI & Google Gemini** → Dynamic, dual-engine LLM & OCR architecture

### ⚡ Backend Architecture
* **FastAPI** → High-performance, async API routing
* **SQLAlchemy** → ORM for PostgreSQL/SQLite relational mapping
* **Firebase Admin SDK** → Cryptographic JWT validation
* **PyMuPDF (fitz) & python-docx** → Deep document parsing

### 🎨 Frontend UI
* **HTML5, CSS3, Vanilla JavaScript**
* **Firebase Client SDK** → OAuth 2.0 Google Sign-In
* Fully responsive Single Page Application (SPA) design

---

## 🔄 The Multimodal RAG Flow