# Streamlit_app.py
import streamlit as st
import financedatabase as fd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd

st.title("Company Similarity Search")

# User Input
api_key = st.text_input("OpenAI API Key:", type="password")
theme = st.text_input("Theme:")
k = st.slider("Number of companies:", min_value=1, max_value=50, value=10)

@st.cache(allow_output_mutation=True)
def load_database():
    equities = fd.Equities()
    return equities.select(country="United States").dropna(subset=['summary'])

equities_united_states = load_database()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def generate_faiss_index(texts):
    return FAISS.from_texts(texts, OpenAIEmbeddings())

if api_key and theme:
    os.environ["OPENAI_API_KEY"] = api_key

    faiss_index = generate_faiss_index(equities_united_states['summary'].tolist())
    results_with_scores = faiss_index.similarity_search_with_score(theme, k=k)

    contents = []
    scores = []

    for doc, score in results_with_scores:
        contents.append(doc.page_content)
        scores.append(score)

    df_temp = pd.DataFrame({'summary': contents, 'Score': scores})
    df_temp_2 = df_temp.merge(equities_united_states, on=['summary'], how='left')
    
    # Display the table
    st.write(df_temp_2)

# Add a table filter functionality
st.write("Filtered Table:")
filtered_column = st.selectbox("Choose a column to filter:", df_temp_2.columns)
filter_value = st.text_input(f"Filter by {filtered_column}:")
if filter_value:
    st.write(df_temp_2[df_temp_2[filtered_column].str.contains(filter_value)])