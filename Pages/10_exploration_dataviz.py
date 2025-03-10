import streamlit as st  
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt

if 'reviews_df' in st.session_state:
    df = st.session_state.reviews_df.copy()
    df['Réponse'] = df['Réponse'].replace(['nan', 'Pas de réponse'], '')


    # Convert all columns to text except 'Note' which should be an integer
    for column in df.columns:
        if column == 'Note':
            df[column] = df[column].astype(int)
    tab1, tab2, tab3 = st.tabs(["Colonnes", "Graphiques", "Tests statistiques"])

    with tab1:

        st.write("Informations sur les colonnes du DataFrame")
        for column in df.columns:
            with st.expander(f"Colonne: {column}"):
                st.write(f"**{column}**")
                st.write(f"Type: {df[column].dtype}")
                st.write(f"Nombre de valeurs uniques: {df[column].nunique()}")
                st.write(f"Valeurs manquantes: {df[column].isnull().sum()}")
                if df[column].dtype == 'object':
                    st.write("Valeurs les plus fréquentes:")
                    st.dataframe(df[column].value_counts().head().reset_index().rename(columns={'index': column, column: 'Count'}))
                else:
                    st.write(f"Statistiques descriptives:")
                    st.dataframe(df[column].describe())
                st.write("---")

    with tab2:
        st.write("Histogramme de la répartition des notes")
        hist_values = np.histogram(df['Note'], bins=5, range=(1, 6))[0]
        plt.figure()
        plt.bar(range(1, 6), hist_values, tick_label=[1, 2, 3, 4, 5])
        plt.xlabel('Notes')
        plt.ylabel('Nombre d\'avis')
        st.pyplot(plt)

        st.write("Nuage de mots des mots les plus fréquents")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('french'))

        text = " ".join(review for review in df['Avis'])
        text = " ".join(word for word in text.split() if word.lower() not in stop_words)
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", min_word_length=4).generate(text)

        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        st.write("Scatterplot entre la longueur des avis et la note")
        df['Avis_length'] = df['Avis'].apply(len)
        plt.figure()
        plt.scatter(df['Note'], df['Avis_length'])
        plt.plot(np.unique(df['Note']), np.poly1d(np.polyfit(df['Note'], df['Avis_length'], 1))(np.unique(df['Note'])), color='red')
        plt.xlabel('Longueur des avis')
        plt.ylabel('Note')
        st.pyplot(plt)

        st.write("Taux de réponse aux avis et note moyenne par entreprise")
        response_rate = df['Réponse'].notnull().groupby(df['Entreprise']).mean()
        average_rating = df.groupby('Entreprise')['Note'].mean()

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Note moyenne')
        ax1.set_ylabel('Taux de réponse')
        ax1.scatter(average_rating.values, response_rate.values, color='blue', marker='o')
        sns.lineplot(x=average_rating.values, y=response_rate.values, ax=ax1,  color='red')
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.write("Tests statistiques à venir")
