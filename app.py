from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import shutil
import time
import gc
from datetime import datetime

from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DEBUG SESSION TRACKING =====================

active_sessions = {}

def log_session(session_id):
    active_sessions[session_id] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active"
    }
    print(f"🆕 New session created: {session_id}")

# ===================== MODELS =====================

embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(model="mistral-small-latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a strict document-based AI assistant.

    You MUST follow these rules:
    1. Answer ONLY using the provided context.
    2. If the answer is NOT clearly present in the context, respond EXACTLY with:
    "I could not find the answer in the document."
    3. DO NOT use your own knowledge.
    4. DO NOT guess or assume anything.
    5. Keep answers concise and formatted in markdown when needed.
    """),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

# ===================== CLEANUP =====================

def scheduled_cleanup(session_id: str):
    print(f"🧹 Starting cleanup for {session_id}")
    time.sleep(600)  # 10 minutes
    
    db_path = f"db/{session_id}"
    temp_file = f"temp/{session_id}.pdf"

    gc.collect()

    if os.path.exists(db_path):
        for i in range(5):
            try:
                shutil.rmtree(db_path)
                print(f"🗑 DB removed: {db_path}")
                break
            except PermissionError:
                print(f"⚠️ DB locked, retrying... ({i+1}/5)")
                time.sleep(2)
            except Exception as e:
                print(f"❌ DB delete error: {e}")
                break

    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"🗑 File removed: {temp_file}")
        except Exception as e:
            print(f"❌ File delete error: {e}")

    active_sessions.pop(session_id, None)
    print(f"✅ Cleanup completed for {session_id}")

# ===================== UPLOAD =====================

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    log_session(session_id)  # 🔥 tracking

    upload_path = f"temp/{session_id}.pdf"
    db_path = f"db/{session_id}"

    os.makedirs("temp", exist_ok=True)
    os.makedirs("db", exist_ok=True)

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    print(f"📄 File saved at: {upload_path}")

    loader = PyPDFLoader(upload_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )

    del vectorstore

    print(f"📂 DB created at: {db_path}")

    background_tasks.add_task(scheduled_cleanup, session_id)

    return {"session_id": session_id}

# ===================== CHAT =====================

@app.post("/chat")
async def chat(data: dict):
    query = data["query"]
    session_id = data["session_id"]
    db_path = f"db/{session_id}"

    if not os.path.exists(db_path):
        return {"answer": "Session expired. Please re-upload your document."}

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    results = vectorstore.similarity_search_with_score(query, k=4)

    threshold = 0.5
    filtered_docs = [doc for doc, score in results if score < threshold]

    if not filtered_docs:
        return {"answer": "I could not find the answer in the document."}

    context = "\n\n".join(doc.page_content for doc in filtered_docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    del vectorstore
    gc.collect()

    return {"answer": response.content}

# ===================== DEBUG ROUTES =====================

@app.get("/debug/files")
def list_files():
    return {
        "temp_files": os.listdir("temp") if os.path.exists("temp") else [],
        "db_folders": os.listdir("db") if os.path.exists("db") else [],
        "active_sessions": active_sessions
    }

@app.get("/debug/cleanup/{session_id}")
def manual_cleanup(session_id: str):
    db_path = f"db/{session_id}"
    temp_file = f"temp/{session_id}.pdf"

    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        active_sessions.pop(session_id, None)

        return {"status": "manual cleanup successful", "session": session_id}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug/status")
def system_status():
    return {
        "status": "running",
        "active_sessions": len(active_sessions),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }