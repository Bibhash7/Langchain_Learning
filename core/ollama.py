import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st
st.title("Cuisine Details")
llm = ChatOllama(model="gemma:2b")
prompt_template = PromptTemplate(
    input_variables=["country","no_of_paras","language"],
    template="""
    You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country. Only output the name and some short details.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paras, in {language}
    """
)
country = st.text_input("Enter Country: ")
paras = st.number_input("Enter number of paras: ", min_value=1, max_value=4)
language = st.text_input("Enter language: ")
if country:
    response = llm.invoke(prompt_template.format(
        country=country,
        no_of_paras=paras,
        language=language
    ))
    st.write(response.content)