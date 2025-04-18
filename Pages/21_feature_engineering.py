import pandas as pd
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
import re
import spacy


spacy.require_gpu()
if 'reviews_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
    with open('liste_fr.txt', 'r', encoding='utf-8') as file:
        french_vocab = set(word.strip() for word in file.readlines())
        df = st.session_state.reviews_df.copy()
        st.title("Feature Engineering")

        # Charger les stop words en français
        stop_words = set(stopwords.words('french'))

        # Initialiser le lemmatizer
        nlp = spacy.load("fr_core_news_sm")
            
        # Fonction pour nettoyer, lemmatiser et extraire les mots importants
        def preprocess_text(text):
            if pd.isnull(text):
                return ""
            # Supprimer les caractères spéciaux et mettre en minuscule
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            # Charger le dictionnaire français à partir de liste_fr.txt
            

            # Supprimer les mots qui ne sont pas dans le dictionnaire français
            text = " ".join([word for word in text.split() if word in french_vocab])
            # Tokenizer, supprimer les stop words et appliquer la lemmatisation avec spaCy
            doc = nlp(text)
            words = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct and not token.is_space]
            return " ".join(words)

        # Appliquer la fonction à la colonne 'Avis'
        #df['Mots_importants'] = df['Avis'].apply(preprocess_text)
        st.text("Lemmatization et extraction des mots importants en cours...")
        st.text("Cette opération peut prendre un certain temps en fonction de la taille du DataFrame.")
        st.text("On retire les caractères spéciaux et les mots qui ne sont pas dans le vocabulaire français")
        st.text("Puis on applique la lemmatisation avec spacy et on retire les stop words.")
        @st.cache_data
        def process_avis_batch(dataframe, start_idx, end_idx):
            
            for i in range(start_idx, min(end_idx, len(dataframe))):
                dataframe.at[i, 'Mots_importants'] = preprocess_text(dataframe.at[i, 'Avis'])
            return dataframe

        total_rows = len(df)
        batch_size = 500
        statut_progres = st.empty()
        progress_bar = st.progress(0)

        for start_idx in range(0, total_rows, batch_size):
            end_idx = start_idx + batch_size
            statut_progres.text(f"Traitement des lignes {start_idx} à {end_idx}...")
            df = process_avis_batch(df, start_idx, end_idx)
            progress_bar.progress(min(end_idx, total_rows) / total_rows)
        st.text("Nouvelle colonne 'Mots_importants' créée avec succès.")
        st.dataframe(df[['Avis', 'Mots_importants']].head())
        @st.cache_data
        def process_reponse_column(dataframe):
            st.text("Reponses :")
            progress_bar = st.progress(0)
            total_rows = len(dataframe)
            for i, (index, row) in enumerate(dataframe.iterrows()):
                dataframe.at[index, 'Mots_importants_reponse'] = preprocess_text(row['Réponse'])
                progress_bar.progress((i + 1) / total_rows)
            return dataframe

        df = process_reponse_column(df)
        st.text("Nouvelle colonne 'Mots_importants_reponse' créée avec succès.")
        st.dataframe(df[['Réponse', 'Mots_importants_reponse']].head())
        
        # Charger le fichier CSV contenant les thèmes et les mots associés
        classification_df = pd.read_csv('Classification_mots.csv')
        classification_df['Mots'] = classification_df['Mots'].apply(lambda x: x.split(','))

        # Fonction pour déterminer les thèmes dominants dans un texte
        def determine_themes(text, classification_df, max_themes=5):
            theme_counts = {}
            for _, row in classification_df.iterrows():
                theme = row['Theme']
            mots = set(row['Mots'])
            count = sum(1 for word in text.split() if word in mots)
            if count > 0:
                theme_counts[theme] = count
            # Trier les thèmes par nombre de mots correspondants et sélectionner les top N
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, _ in sorted_themes[:max_themes]]

        # Appliquer la fonction pour déterminer les thèmes sur les colonnes 'Mots_importants' et 'Mots_importants_reponse'
        st.text("Détermination des thèmes pour les avis et les réponses à partir d'une liste obtenue par chatGPT ( experimental )...")
        df['Themes_Avis'] = df['Mots_importants'].apply(lambda x: determine_themes(x, classification_df))
        df['Themes_Réponse'] = df['Mots_importants_reponse'].apply(lambda x: determine_themes(x, classification_df))

        st.text("Colonnes 'Thèmes_Avis' et 'Thèmes_Réponse' créées avec succès.")
        st.dataframe(df[['Avis', 'Themes_Avis', 'Réponse', 'Themes_Réponse']].head())

        #créer une colonne sentiment positif / négatif à partir de la note
        st.text("Création de la colonne 'Sentiment' à partir de la colonne 'Note'")
        df['Sentiment'] = df['Note'].apply(lambda x: 'Positif' if x > 3 else 'Négatif' )

        if st.button("Appliquer les modifications"):
            st.session_state.reviews_df = df
            st.success("Les modifications ont été appliquées au DataFrame en mémoire.")