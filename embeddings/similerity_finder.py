import os
from langchain_ollama import OllamaEmbeddings
import numpy as np
llm = OllamaEmbeddings(model="gemma:2b")
text1 = input("Enter text1:")
text2 = input("Enter text2:")
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)

similerity_score = np.dot(response1, response2)
print(similerity_score*100, "%")