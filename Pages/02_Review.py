
import streamlit as st
import pandas as pd 
import numpy as np  


if 'reviews_df' in st.session_state:
    company_filter = st.selectbox("Filtrer par entreprise", options=["Toutes"] + list(st.session_state.reviews_df['Entreprise'].unique()))
    rating_filter = st.selectbox("Filtrer par note", options=["Toutes"] + list(st.session_state.reviews_df['Note'].unique()))

    # Apply filters
    filtered_df = st.session_state.reviews_df
    if company_filter != "Toutes":
        filtered_df = filtered_df[filtered_df['Entreprise'] == company_filter]
    if rating_filter != "Toutes":
        filtered_df = filtered_df[filtered_df['Note'] == rating_filter]
    
    
    st.write(filtered_df)

    if not filtered_df.empty:
        all_reviews_text = ' '.join(filtered_df['Avis'])
        words = all_reviews_text.split()
        words = [word for word in words if len(word) > 4]
        word_counts = pd.Series(words).value_counts().head(10)
        st.write("Les 10 mots les plus fréquents dans les avis:")
        st.write(word_counts)
        avg_rating = filtered_df['Note'].replace("Pas de note", np.nan).astype(float).mean()
        st.write(f"Note moyenne: {avg_rating:.2f}")
