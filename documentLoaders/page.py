from langchain_community.document_loaders import WebBaseLoader
url = "https://vibieprince.vercel.app"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)