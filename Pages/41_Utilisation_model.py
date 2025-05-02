import streamlit as st
import pandas as pd

st.title("Utilisation du modèle de prédiction")

if 'processed_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
    if 'rf_model' not in st.session_state:
        st.error("Veuillez entraîner un modèle de prédiction avant d'utiliser cette page.")
    else:
        df = st.session_state['processed_df'].copy()
         # Prédire les thèmes pour les 5 premières lignes
        model = st.session_state['rf_model']
        predictions = model.predict(df.iloc[:5])
        df['Predicted_Themes'] = pd.Series(predictions, index=df.index[:5])

        # Afficher les résultats
        st.subheader("Prédictions pour les 5 premières lignes")
        st.write(df.head(5))
        

   