import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time

st.title("Appli Trustpilot ⭐⭐⭐⭐⭐")

st.text("Super Appli, elle permet d'obtenir des avis Trustpilot, de les analyser et de créer un modèle de prédiction de sentiment.")
st.markdown("<div style='text-align: right'>Camille Hamel ✅<br/><small>Le 16 mai 2025 à 16h00</small></div>", unsafe_allow_html=True)

st.header("Projet Data Science")

st.subheader("1. Obtention du Dataset")
st.write("""
Le dataset a été constitué par scraping de Trustpilot en utilisant BeautifulSoup. 
Nous avons collecté les avis clients de plusieurs entreprises, incluant le texte des avis et leurs notes associées.
""")

st.subheader("2. Visualisation des Données")
st.write("""
L'analyse exploratoire des données a révélé :
- La distribution des notes (1-5 étoiles)
- La répartition des avis vérifiés ou non vérifiés
- La longueur des avis 
- la saisonnalité et l'évolution dans le temps
- Les mots les plus fréquents
""")

st.subheader("3. Prétraitement des Données")
st.write("""
Le texte des avis a subi plusieurs étapes de prétraitement :
- Suppression des caractères spéciaux
- Conversion en minuscules
- Retrait des stop words (mots non significatifs comme 'le', 'la', 'les', etc.)
- Lemmatisation pour réduire les mots à leur forme racine
""")

st.subheader("4. Modélisation")
st.write("""
Nous avons utilisé une approche de classification binaire avec :
- Vectorisation du texte (TF-IDF)
- Test de plusieurs algorithmes simples (RandomForest, SVM, GradientBoosting ...)
- Validation croisée pour évaluer les performances
- Implémentation d'un réseau de neurones
""")

st.subheader("5. Choix de la Classification Binaire")
st.write("""
La prédiction binaire (positif/négatif) a été préférée à la prédiction de notes (1-5) car :
- Meilleure précision sur deux classes
- Plus pertinent pour l'analyse de sentiment
- Modèles plus simples et plus robustes
""")

st.subheader("6. Prédiction")
st.write("""
- Démonstration du modèle sur de nouvelles données
""")

st.subheader("7. Import/Export du Modèle")
st.write("""
Le modèle est :
- Entraîné et sauvegardé au format pickle
- Facilement importable pour de nouvelles prédictions
""")

st.divider()

