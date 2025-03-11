import pandas as pd
import streamlit as st

def export_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"DataFrame exported to {file_path}")

def import_from_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"DataFrame imported from {file_path}")
    return df



st.title("Export/Import DataFrame")
if 'reviews_df'  in st.session_state:

    csv = st.session_state.reviews_df.to_csv(index=False, escapechar='\\').encode('utf-8')
    st.download_button(
        label="Exporter les données sous forme de CSV",
        data=csv,
        file_name='exported_reviews.csv',
        mime='text/csv',
    )


uploaded_file = st.file_uploader("Envoyer un CSV", type="csv")
if uploaded_file is not None:
    st.session_state.reviews_df = pd.read_csv(uploaded_file, escapechar='\\')
    st.write(st.session_state.reviews_df)
