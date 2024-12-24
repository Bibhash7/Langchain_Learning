import os
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
embeddings = OllamaEmbeddings(model="gemma:2b")
llm = ChatOllama(model="gemma:2b")
document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)
persist_directory = "C:/Users/bibha/Desktop/ALL-Project-Oct-23/TalentCraft/chroma_store"
vector_store = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=persist_directory,
)
retriver = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an assistant for answering questions. Use the provided context to respond.If the answer isn't clear, acknowledge that you don't know.
                        Limit your response to three concise sentences.
                        {context}"""),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human","{input}")
    ]
)

history_aware_retriver = create_history_aware_retriever(llm,retriver, prompt_template)
qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriver, qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("Chat with documents")
question = st.text_input("Your question:")
response = chain_with_history.invoke({"input":question}, {"configurable": {"session_id": "abc123"}})
if question:
    st.write(response['answer'])

