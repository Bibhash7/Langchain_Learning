import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
import streamlit as st
st.title("Agile Guide:")
llm = ChatOllama(model="gemma:2b")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a agile coach, answer any questions related to the agile process."),
        ("human","{input}")
    ]
)

input = st.text_input("Enter any agile related question: ")
chain = prompt_template | llm
if input:
    response = chain.invoke({"input": input})
    st.write(response.content)