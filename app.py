from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import shutil
import time
import gc # Garbage collection to force-release file locks

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

# Initialize Models
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(model="mistral-small-latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful study assistant. Use the provided context to answer questions clearly. Use markdown for formatting (bullets, bolding, etc.)."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

def scheduled_cleanup(session_id: str):
    time.sleep(600)  # 10 minutes
    
    db_path = f"db/{session_id}"
    temp_file = f"temp/{session_id}.pdf"
    
    # 1. Force Python to release memory and file handles
    gc.collect() 

    # 2. Robust directory deletion
    if os.path.exists(db_path):
        # We try multiple times because Windows is slow to release locks
        for i in range(5): 
            try:
                shutil.rmtree(db_path)
                print(f"✅ Cleaned up DB: {session_id}")
                break 
            except PermissionError:
                print(f"⚠️ DB locked, retrying in 2s... ({i+1}/5)")
                time.sleep(2)
            except Exception as e:
                print(f"❌ Error: {e}")
                break

    # 3. Simple file deletion
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"✅ Cleaned up PDF: {session_id}")
        except Exception as e:
            print(f"❌ Error cleaning PDF: {e}")

@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    upload_path = f"temp/{session_id}.pdf"
    db_path = f"db/{session_id}"

    os.makedirs("temp", exist_ok=True)
    os.makedirs("db", exist_ok=True)

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(upload_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    # Explicitly trigger a save/close if possible
    del vectorstore 

    # Use FastAPI BackgroundTasks instead of raw threading for better stability
    background_tasks.add_task(scheduled_cleanup, session_id)

    return {"session_id": session_id}

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

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    # Clean up reference to release lock
    del vectorstore
    gc.collect()

    return {"answer": response.content}