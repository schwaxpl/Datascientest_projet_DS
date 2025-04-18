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

if 'reviews_df' in st.session_state:
    if 'Mots_importants' in st.session_state.reviews_df.columns and 'Mots_importants_reponse' in st.session_state.reviews_df.columns:
        unique_words = pd.Series(
            pd.concat([
                st.session_state.reviews_df['Mots_importants'].dropna().str.split().explode(),
                st.session_state.reviews_df['Mots_importants_reponse'].dropna().str.split().explode()
            ])
        )

        word_counts = unique_words.value_counts(ascending=False)  # Sort by occurrences in descending order
        unique_words_df = pd.DataFrame({
            "Unique Words": word_counts.index,
            "Occurrences": word_counts.values
        })

        csv = unique_words_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Télécharger les mots uniques avec occurrences",
            data=csv,
            file_name='unique_words_with_occurrences.csv',
            mime='text/csv',
        )
