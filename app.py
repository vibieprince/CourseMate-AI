from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid
import shutil
import time
import gc
import base64
from datetime import datetime
from langchain_core.documents import Document

from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import fitz  # PyMuPDF

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allow your specific GitHub Pages frontend domain
    allow_origins=[
        "https://vibieprince.github.io",
        "http://127.0.0.1:5500",  # Keeps local Live Server testing working
        "http://localhost:3000"   
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Models
embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
ocr_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a strict study assistant. Use ONLY the provided context to answer questions. "
        "If the answer is not contained within the context, specifically state: "
        "'I am sorry, but the uploaded document does not contain information regarding this.' "
        "Do not use outside knowledge or mention other companies/topics not in the text. "
        "Use markdown for formatting."
        "However if someone asks you general questions like how are you, what do you do, what's the day today, what will be the day on the date mentioned in the document, etc... you have to answer accordingly."
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

    try:
        doc = fitz.open(upload_path)
    except Exception as e:
        return {"error": f"Failed to read PDF: {e}"}

    cleaned_docs = []
    for i in range(len(doc)):
        page = doc[i]
        text = " ".join(page.get_text().split()).strip()

        # Check if the page is image-only/scanned (fewer than 40 characters extracted)
        if len(text) < 40:
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            
            try:
                response = ocr_llm.invoke([
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "Extract all readable text from this document page. Return only extracted text exactly as written."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                            }
                        ]
                    )
                ])
                ocr_text = response.content.strip()
                if ocr_text:
                    cleaned_docs.append(Document(page_content=ocr_text, metadata={"source": file.filename, "page": i+1}))
                elif text:
                    cleaned_docs.append(Document(page_content=text, metadata={"source": file.filename, "page": i+1}))
            except Exception:
                if text:
                    cleaned_docs.append(Document(page_content=text, metadata={"source": file.filename, "page": i+1}))
        else:
            cleaned_docs.append(Document(page_content=text, metadata={"source": file.filename, "page": i+1}))

    if not cleaned_docs:
        return {"error": "No readable text could be processed from this document."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(cleaned_docs)

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

    # Injecting system date and time explicitly within the context block
    # This allows the system prompt's general conversational exception instructions to fire seamlessly
    current_time_str = datetime.now().strftime("%A, %B %d, %Y")
    context_elements = [f"Current Live System Date/Time: {current_time_str}"]
    context_elements.extend(doc.page_content for doc in docs)
    context = "\n\n".join(context_elements)

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