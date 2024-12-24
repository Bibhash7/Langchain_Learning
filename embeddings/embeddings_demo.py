import os
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


llm = OllamaEmbeddings(model="gemma:2b")
text = input("Enter text:")
response = llm.embed_query(text)
print(response)