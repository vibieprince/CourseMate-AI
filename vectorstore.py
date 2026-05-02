from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.",metadata={"source":"AI Book"}),
    Document(page_content="Pandas is used for data analysis in Python.",metadata={"source":"Data Science Book"}),
    Document(page_content="Neural Networks are used in Deep Learning.",metadata={"source":"Deep Learning Book"})
]

embedding_model = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "CourseMate AI/chroma-db"
)

result = vectorstore.similarity_search("What is used for data analysis?",k=2)
for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning.")

for d in docs:
    print(d.page_content)
