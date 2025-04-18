import pandas as pd
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler


st.title("Preprocessing")
if 'reviews_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
        
    df = st.session_state.reviews_df.copy()
    st.text("Informations sur le DataFrame")
    # Extraire les infos sous forme de DataFrame
    df_info = pd.DataFrame({
        "Colonnes": df.columns,
        "Type": df.dtypes.astype(str),
        "Valeurs Non Nul": df.count(),
        "Valeurs Manquantes": df.isna().sum(),
        "% Manquantes": (df.isna().sum() / len(df) * 100).round(2)
    }).reset_index(drop=True)
    st.dataframe(df_info)

    st.text("Types de données")
    if df['Note'].dtype != np.int64:
        st.text("La colonne 'Note' n'est pas de type int. Conversion en cours...")
        df['Note'] = pd.to_numeric(df['Note'], errors='coerce').fillna(0).astype(int)
        st.text("Conversion terminée. Voici le type après conversion :")
        st.text(df['Note'].dtype)
    if df['Date'].dtype != np.datetime64:
        st.text("La colonne 'Date' n'est pas de type datetime. Conversion en cours...")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        st.text("Conversion terminée. Voici le type après conversion :")
        st.text(df['Date'].dtype)

    st.text("Les valeurs manquantes de Réponse et Avis ne sont pas des valeurs nulles, mais des chaînes de caractères. Nous les remplaçons par des valeurs nulles.")
    df['Réponse'] = df['Réponse'].replace(['nan', 'Pas de réponse'], None)
    df['Avis'] = df['Avis'].replace(['nan', "Pas de texte d'avis"], None)
    st.text("Apperçu après remplacement:")
    st.text("Réponse:")
    st.text(f"Valeurs Non Nulles: {df['Réponse'].notnull().sum()}")
    st.text(f"Valeurs Nulles: {df['Réponse'].isnull().sum()}")

    st.text("Avis:")
    st.text(f"Valeurs Non Nulles: {df['Avis'].notnull().sum()}")
    st.text(f"Valeurs Nulles: {df['Avis'].isnull().sum()}")

    st.text("Encodage de la colonne 'Vérifié' en booléen")
    st.text(f"Valeurs uniques avant encodage : {df['Vérifié'].unique()}")
    if not df['Vérifié'].isin([True, False]).all():
        df['Vérifié'] = df['Vérifié'].apply(lambda x: True if x == 'Vérifié' else False)
        st.text("Encodage terminé. Voici un aperçu des valeurs uniques après encodage :")
    else:
        df['Vérifié'] = df['Vérifié'].astype(bool)
        st.text("Les valeurs étaient déjà encodées. Conversion du type en booléen effectuée.")
    st.text(df['Vérifié'].unique())

    if st.button("Appliquer les modifications",type="primary"):
        st.session_state.reviews_df = df
        st.success("Les modifications ont été appliquées au DataFrame en mémoire.")

    st.write(":red[Attention la classe note est déséquilibrée.] Nous proposons un sous-échantillonnage des classes majoritaires pour équilibrer les classes.")
    note_counts = df['Note'].value_counts()
    st.text(f"Distribution initiale des classes 'Note':\n{note_counts}")

    min_count = note_counts.min()
    st.text(f"Nombre minimum d'échantillons par classe: {min_count}")

    df_under = df.groupby('Note').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    st.text("Sous-échantillonnage terminé. Nouvelle distribution des classes 'Note':")
    st.text(df_under['Note'].value_counts())

    if st.button("Appliquer le sous-echantillonnage",type="primary"):
        st.session_state.reviews_df = df_under
        st.success("Les modifications ont été appliquées au DataFrame en mémoire.")
    
    st.write(":blue[Option pour équilibrer les classes avec sur-échantillonnage SMOTE.]")

    st.text("Application de RandomOverSampler pour équilibrer les classes.")
    ros = RandomOverSampler(random_state=42)
    try:
        X = df.drop(columns=['Note'])
        y = df['Note'].astype(str)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        y_resampled = y_resampled.astype(int)  # Convertir les labels en int après sur-échantillonnage
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Note'])], axis=1)
        st.text("RandomOverSampler a été appliqué avec succès. Nouvelle distribution des classes 'Note':")
        st.text(df_resampled['Note'].value_counts())

        if st.button("Enregistrer les modifications après RandomOverSampler", type="primary"):
            st.session_state.reviews_df = df_resampled
            st.success("Les modifications après RandomOverSampler ont été enregistrées dans le DataFrame en mémoire.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de l'application de RandomOverSampler: {e}")