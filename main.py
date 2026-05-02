from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma 
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="CourseMate AI/Chroma-db",
    embedding_function= embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5 #diversity of result
    }
)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Use only the provided context to answer the question. If the answer is not present in the context, say 'I could not find the answer in the document.'"),
        ("human", "Context: {context}\nQuestion: {question}")
    ]
)

print("Rag System created")
print("Press 0 to exit")

while True:
    query  = input("You : ")
    if query == "0":
        break
    docs = retriever.invoke(query)
    context = "\n\n".join((doc.page_content for doc in docs))

    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")