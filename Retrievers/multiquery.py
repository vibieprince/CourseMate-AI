from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# 📄 Documents
documents = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# 🔥 Embeddings
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 🧠 Vector DB
vectorstore = Chroma.from_documents(documents, embeddings)

retriever = vectorstore.as_retriever()

# 🤖 LLM (for query generation)
llm = ChatMistralAI(model="mistral-small-latest")

# 🚀 MultiQuery Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# 🔍 Query
query = "What is gradient descent?"

retrieved_docs = multi_query_retriever.invoke(query)

# 🖨 Output
print("\n===== MultiQuery Retrieved Documents =====\n")

for doc in retrieved_docs:
    print("-", doc.page_content)