import streamlit as st  
import pandas as pd
import numpy as np
import requests


if 'reviews_df' in st.session_state:
    df = st.session_state.reviews_df.copy()
    df['Réponse'] = df['Réponse'].replace(['nan', 'Pas de réponse'], None)
    df['Avis'] = df['Avis'].replace(['nan', "Pas de texte d'avis"], None)

    entreprises = sorted(df['Entreprise'].unique())
    selected_entreprise = st.selectbox('Sélectionnez une entreprise', ['Toutes'] + list(entreprises))

    notes = sorted(df['Note'].unique())
    selected_note = st.selectbox('Sélectionnez une note', ['Toutes'] + list(notes))

    reponse_options = ['Toutes', 'Avec réponse', 'Sans réponse']
    selected_reponse = st.selectbox('Sélectionnez la présence de réponse', reponse_options)

    if selected_entreprise != 'Toutes':
        df = df[df['Entreprise'] == selected_entreprise]

    if selected_note != 'Toutes':
        df = df[df['Note'] == selected_note]

    if selected_reponse == 'Avec réponse':
        df = df[df['Réponse'].notna()]
    elif selected_reponse == 'Sans réponse':
        df = df[df['Réponse'].isna()]

    st.dataframe(df)

    avis_id = st.text_input('Entrez l\'ID de l\'avis')
    if st.button('Proposer des réponses'):
        if avis_id:
            avis_text = df.loc[int(avis_id), 'Avis']
            if avis_text:

                API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
                headers = {"Authorization": f"Bearer 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"}

                def query(payload):
                    response = requests.post(API_URL, headers=headers, json=payload)
                    return response.json()

                data = query({"inputs": avis_text, "parameters": {"max_length": 50, "num_return_sequences": 1}})
                entreprise_name = df.loc[int(avis_id), 'Entreprise']
                themes =["Formation","Formateur","Suivi","Plateforme"]
                prompt = f"Tu es un assistant IA qui aide à répondre à des avis clients pour améliorer l'image et la confiance d'une entreprise. \n Utilisateur : Écrit moi une réponse à cet avis concernant l'entreprise {entreprise_name}, sachant que les thèmes importants à mettre en avant pour l'entreprise sont {themes}. Dans le cas où un ou plusieurs problèmes ne semblent pas être résolus, leur proposer de prendre contact à l'adresse \"contact@{entreprise_name}.com\". Tu seras pénalisé si tu ne réponds pas dans la langue de l'avis. Tu seras récompensé si ta réponse est rassurante, professionnelle et bienveillante. \n L'avis est le suivant : {avis_text} \n Réponse :"
                st.write(f"Prompt envoyé : {prompt}")
                data = query({"inputs": prompt, "parameters": {"max_length": 50, "max_new_tokens":2048, "num_return_sequences": 1,"temperature":0.7,"repetition_penalty":1.0}})
                st.write('Proposition de réponse:')
                st.write(data)
            else:
                st.write('ID d\'avis non trouvé ou avis vide.')
        else:
            st.write('Veuillez entrer un ID d\'avis.')