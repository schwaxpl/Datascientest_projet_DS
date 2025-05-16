import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
st.title("Utilisation du modèle de prédiction")

# Chargement du modèle PKL
# Vectorization
@st.cache_data
def vectorize_tfidf(df):
    # Load the TF-IDF vectorizer from the saved file
    with open('vectorizers/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    X = tfidf_vectorizer.transform(df['Mots_importants'])
    return pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

if 'model' not in st.session_state:
    model_path = "models/tf_idf_mdl.pkl"
    if os.path.exists(model_path):
        if st.button("Charger le modèle"):
            with open(model_path, 'rb') as file:
                st.session_state['model'] = pickle.load(file)
            st.success("Modèle tf_idf_mdl.pkl chargé avec succès!")
    else:
        st.error("Le fichier models/tf_idf_mdl.pkl n'existe pas")
if 'model' in st.session_state:
    # Prédiction sur un texte personnalisé
    st.subheader("Prédiction sur un texte personnalisé")
    custom_text = st.text_area("Entrez votre texte ici")

    if st.button("Prédire le sentiment"):
        if custom_text and 'model' in st.session_state:
            # Créer un DataFrame avec le texte personnalisé
            custom_df = pd.DataFrame({'Mots_importants': [custom_text]})
            # Vectoriser le texte
            custom_vector = vectorize_tfidf(custom_df)
            # Faire la prédiction
            prediction = st.session_state['model'].predict(custom_vector)
            classe_index = prediction.argmax(axis=1)[0]
            resultat = "positif" if classe_index == 1 else "négatif"
            st.write(f"Résultat : {resultat}")
        else:
            st.warning("Veuillez entrer un texte et charger un modèle")

# Sélection des données aléatoires
if 'reviews_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
    if 'random_sample' not in st.session_state:
        st.session_state['random_sample'] = st.session_state['reviews_df'].sample(n=50)

    if st.button("Générer 50 nouvelles lignes aléatoires"):
        st.session_state['random_sample'] = st.session_state['reviews_df'].sample(n=50)

    # Affichage de l'échantillon
    st.subheader("Échantillon aléatoire de 50 lignes")
    st.write(st.session_state['random_sample'])

    # Prédiction sur l'échantillon
    if 'model' in st.session_state:
        if st.button("Lancer la prédiction"):
            # Vectorize the text using TF-IDF
            sample_vector = vectorize_tfidf(st.session_state['random_sample'])
            # Make predictions
            predictions = st.session_state['model'].predict(sample_vector)
            # Create results dataframe
            result_df = st.session_state['random_sample'].copy()
            # Get class index and convert to sentiment labels
            class_indices = predictions.argmax(axis=1)
            result_df['Predicted_Sentiment'] = ['positif' if idx == 1 else 'négatif' for idx in class_indices]
            
            st.subheader("Résultats des prédictions")
            st.write(result_df)
    else:
        st.warning("Veuillez d'abord charger un modèle")