import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
st.title("Speech Generator:")
llm1 = ChatOllama(model="gemma:2b")
llm2 = ChatOllama(model="llama3.2:latest")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    You need to craft an impactful title for a speech
    on the following topic: {topic}
    Answer exactly with one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

first_chain = title_prompt | llm1 | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt | llm2
final_chain = first_chain | second_chain

topic = st.text_input("Enter topic: ")

if topic:
    response = final_chain.invoke(
        {
            "topic": topic,
        }
    )
    st.write(response.content)