import os
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
st.title("Agile Guide:")
llm = ChatOllama(model="gemma:2b")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a agile coach, answer any questions related to the agile process."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)

input = st.text_input("Enter any agile related question: ")
chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)
if input:
    response = chain_with_history.invoke({"input": input}, {"configurable": {"session_id": "abc123"}})
    st.write(response.content)

st.write("History")
st.write(history_for_chain)