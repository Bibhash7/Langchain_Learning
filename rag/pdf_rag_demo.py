import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

embeddings = OllamaEmbeddings(model="gemma:2b")
llm = ChatOllama(model="gemma:2b")
document = PDFPlumberLoader("academic_research_data.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriver = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions. Use the provided context to respond.If the answer isn't clear, acknowledge that you don't know.
                        Limit your response to three concise sentences.
                        {context}"""),
        ("human","{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(retriver, qa_chain)

print("Chat with documents")
question = input("Your question:")
response = rag_chain.invoke({"input":question})
if question:
    print(response['answer'])

