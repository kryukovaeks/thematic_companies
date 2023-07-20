# Streamlit_app.py
import streamlit as st
import financedatabase as fd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd
st.set_page_config(layout='wide') 
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
    # Expand the maximum width of each cell to display more content
    pd.set_option('display.max_colwidth', None)   
    # Check if the selected columns are in the session state. If not, initialize it.
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = ['summary', 'Score', 'name', 'industry']

    # Update the multiselect widget to use session state.
    selected_columns = st.multiselect("Select columns", df_temp_2.columns, default=st.session_state.selected_columns)
    st.session_state.selected_columns = selected_columns

    if selected_columns:
        df_selected = df_temp_2[selected_columns]
        html_table = df_selected.to_html(escape=False, index=False)

        # Wrap the HTML table in a div with fixed height and overflow
        html_table_with_scroll = f"""
        <div style="height:300px;overflow:auto;">
            {html_table}
        </div>
        """

        # Use Streamlit's markdown renderer to display the wrapped table
        st.markdown(html_table_with_scroll, unsafe_allow_html=True)
