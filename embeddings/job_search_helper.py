import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma


llm = OllamaEmbeddings(model="gemma:2b")
document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks, llm)
retriver = db.as_retriever()
text = input("Enter the query:")
embedding_vector = llm.embed_query(text)
docs = retriver.invoke(text)
for doc in docs:
    print(doc.page_content)