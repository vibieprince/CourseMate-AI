from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

data = PyPDFLoader("CourseMate AI/files/GRU.pdf")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system","you are a Ai agent that summarizes the text"),
     ("human","{data}")]
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = template.format_messages(data=docs)

result = model.invoke(prompt)

print(result.content)
