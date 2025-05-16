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
            try:
                avis_text = df.loc[int(avis_id), 'Avis']
                if avis_text:

                    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
                    headers = {"Authorization": f"Bearer YOUR_HUGGINGFACE_API_KEY"} #remplacer par votre clé API Hugging Face

                    def query(payload):
                        response = requests.post(API_URL, headers=headers, json=payload)
                        response.raise_for_status()
                        return response.json()

                    entreprise_name = df.loc[int(avis_id), 'Entreprise']
                    themes = ["Formation", "Formateur", "Suivi", "Plateforme"]

                    prompt = (
                        f"Tu es un assistant IA. Ta seule tâche est d'écrire une réponse professionnelle, bienveillante et rassurante à l'avis suivant. "
                        f"Cette réponse représente l'entreprise {entreprise_name}, et doit mettre en valeur les thèmes clés suivants : {themes}. "
                        f"Si l'avis contient des problèmes non résolus, invite poliment le client à contacter \"contact@{entreprise_name}.com\" pour un suivi. "
                        f"Tu dois répondre dans la langue de l'avis. "
                        f"N'écris qu'une seule réponse complète. N'ajoute aucune explication, commentaire ou autre contenu. Écris uniquement la réponse elle-même, comme si elle allait être directement envoyée au client."
                        f"\n\nAvis : {avis_text}"
                        f"\n\nRéponse :"
                    )


                    data = query({
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 200,
                            "temperature": 0.7,
                            "repetition_penalty": 1.0
                        }
                    })

                    # ✅ Récupérer la réponse
                    if isinstance(data, list) and "generated_text" in data[0]:
                        generated = data[0]["generated_text"]
                        generated = generated.split("Réponse :")[1].strip()
                        st.write("Proposition de réponse :")
                        st.write(generated)
                    else:
                        st.write("Erreur dans la réponse du modèle :", data)

                else:
                    st.warning("ID d'avis non trouvé ou avis vide.")
            except Exception as e:
                st.error(f"Erreur : {e}")
        else:
            st.warning("Veuillez entrer un ID d'avis.")