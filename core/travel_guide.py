import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
import streamlit as st
st.title("AI travel guide:")
llm = ChatOllama(model="gemma:2b")
prompt_template = PromptTemplate(
    input_variables=["city","month","language","budget"],
    template="""
    Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)
city = st.text_input("Enter City: ")
month = st.text_input("Month: ")
language = st.text_input("Enter language: ")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])
if city and month and language and budget:
    response = llm.invoke(prompt_template.format(
        city=city,
        month=month,
        language=language,
        budget=budget,
    ))
    st.write(response.content)