from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import shutil
import time
import gc

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
# Ensure MISTRAL_API_KEY and HUGGINGFACEHUB_API_TOKEN are in your Environment Variables
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict study assistant. Use ONLY the provided context to answer questions. "
        "If the answer is not contained within the context, specifically state: "
        "'I am sorry, but the uploaded document does not contain information regarding this.' "
        "Do not use outside knowledge or mention other companies/topics not in the text. "
        "Use markdown for formatting."
    )),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def scheduled_cleanup(session_id: str):
    print(f"⏳ Cleanup timer started for session: {session_id} (10 minutes)")
    time.sleep(600) 
    
    db_path = f"db/{session_id}"
    temp_file = f"temp/{session_id}.pdf"
    
    gc.collect() 

    if os.path.exists(db_path):
        for i in range(5): 
            try:
                shutil.rmtree(db_path)
                print(f"✅ SUCCESS: Deleted DB folder for {session_id}")
                break 
            except PermissionError:
                print(f"⚠️ Retrying DB deletion for {session_id}...")
                time.sleep(2)
            except Exception as e:
                print(f"❌ ERROR deleting DB: {e}")
                break

    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"✅ SUCCESS: Deleted PDF file for {session_id}")
        except Exception as e:
            print(f"❌ ERROR deleting PDF: {e}")
    
    print(f"🏁 Finished cleanup routine for {session_id}")

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

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    
    del vectorstore 
    background_tasks.add_task(scheduled_cleanup, session_id)

    return {"session_id": session_id}

@app.post("/chat")
async def chat(data: dict):
    query = data.get("query")
    session_id = data.get("session_id")
    db_path = f"db/{session_id}"

    if not session_id or not os.path.exists(db_path):
        return {"answer": "Session expired or invalid. Please re-upload your document."}

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 3, "fetch_k": 10}
    )
    
    docs = retriever.invoke(query)
    
    if not docs:
        return {"answer": "I am sorry, but I couldn't find any relevant sections in the document to answer that."}

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    del vectorstore
    gc.collect()

    return {"answer": response.content}

@app.get("/storage-status")
async def get_storage_status():
    temp_files = os.listdir("temp") if os.path.exists("temp") else []
    db_folders = os.listdir("db") if os.path.exists("db") else []
    
    return {
        "active_pdfs_count": len(temp_files),
        "active_pdfs": temp_files,
        "active_db_sessions_count": len(db_folders),
        "active_db_sessions": db_folders,
        "location_info": {
            "pdf_storage": os.path.abspath("temp"),
            "vector_storage": os.path.abspath("db")
        }
    }