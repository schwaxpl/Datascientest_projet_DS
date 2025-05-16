import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from transformers import AutoTokenizer
from sklearn.metrics import classification_report   
import pickle
import io
# Vectorization
@st.cache_data
def vectorize_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.3)
    X = tfidf_vectorizer.fit_transform(df['Mots_importants'])
    with open('vectorizers/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    return pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
@st.cache_data
def vectorize_embedding(df):
    df["tokens"] = df["Mots_importants"].apply(word_tokenize)
    model = SentenceTransformer("dangvantuan/sentence-camembert-base")
    embeddings = df["tokens"].apply(lambda x: model.encode(" ".join(x), show_progress_bar=False))
    return np.array(embeddings.tolist())


st.title("Modélisation en utilisant un réseau de neurones")

if 'reviews_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
    df = st.session_state['reviews_df'].copy()
    
    
    X_tfidf = vectorize_tfidf(df)
    y = df["Sentiment"].values


    # Convert sentiment labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Sentiment'].values)

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    f1_tfidf,f1, accuracy_tfidf, accuracy = 0,0,0,0
    # Train a first model using TF-IDF features
    tab1, tab2, tab3 = st.tabs(["Modèle avec des caractéristiques TF-IDF", "Modèle avec des caractéristiques d'embedding", "Conclusion"])
    
    with tab1:

        # Split TF-IDF data
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Create and compile model
        model_tfidf = Sequential()
        model_tfidf.add(Dense(256, activation='relu', input_shape=(X_train_tfidf.shape[1],), kernel_regularizer='l2'))
        model_tfidf.add(Dropout(0.3))
        model_tfidf.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model_tfidf.add(Dropout(0.3))
        model_tfidf.add(Dense(64, activation='relu', kernel_regularizer='l2'))
        model_tfidf.add(Dropout(0.2))
        model_tfidf.add(Dense(len(label_encoder.classes_), activation='softmax'))

        model_tfidf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        def train_tfidf_model(X_train, y_train):
            history = model_tfidf.fit(X_train, y_train, epochs=30, validation_split=0.1, 
                        verbose=1, callbacks=[early_stopping])
            return history

        history_tfidf = train_tfidf_model(X_train_tfidf, y_train)

        # Evaluate
        loss_tfidf, accuracy_tfidf = model_tfidf.evaluate(X_test_tfidf, y_test, verbose=0)
        st.write(f"Perte sur le test TF-IDF : {loss_tfidf:.4f}")
        st.write(f"Précision sur le test TF-IDF : {accuracy_tfidf:.4f}")

        # F1 Score
        y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
        y_pred_classes_tfidf = np.argmax(y_pred_tfidf, axis=1)
        f1_tfidf = f1_score(y_test, y_pred_classes_tfidf, average='weighted')
        st.write(f"Score F1 TF-IDF : {f1_tfidf:.4f}")

        # Classification Report
        st.write("Rapport de classification TF-IDF :")
        report_tfidf = classification_report(y_test, y_pred_classes_tfidf, target_names=label_encoder.classes_)
        st.text(report_tfidf)

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history_tfidf.history['accuracy'], label='Précision sur l\'entraînement')
        ax[0].plot(history_tfidf.history['val_accuracy'], label='Précision sur la validation')
        ax[0].set_title('Précision du modèle TF-IDF')
        ax[0].set_ylabel('Précision')
        ax[0].set_xlabel('Époque')
        ax[0].legend(loc='upper left')

        ax[1].plot(history_tfidf.history['loss'], label='Perte sur l\'entraînement')
        ax[1].plot(history_tfidf.history['val_loss'], label='Perte sur la validation')
        ax[1].set_title('Perte du modèle TF-IDF')
        ax[1].set_ylabel('Perte')
        ax[1].set_xlabel('Époque')
        ax[1].legend(loc='upper left')
        st.pyplot(fig)
        
        #explication des résultats avec shap
        
        # Create a SHAP explainer using the model and the training data
        def calculate_shap_values(model_tfidf, X_train_tfidf, X_test_tfidf):
            explainer = shap.Explainer(model_tfidf.predict, X_train_tfidf[:40])
            shap_values = explainer(X_test_tfidf[:10], max_evals=2000)
            return shap_values

        shap_values = calculate_shap_values(model_tfidf, X_train_tfidf, X_test_tfidf)
        # Plot the SHAP values for the first instance in the test data
        shap.initjs()
        st.write("Valeurs SHAP pour la première instance des données de test :") 
        print(type(shap_values))
        print(shap_values.shape)
        fig, ax = plt.subplots()   
        shap.plots.waterfall(shap_values[0,:,0], max_display=20,show=False)
        st.pyplot(fig)

        # Add a download button for the TF-IDF model



        # Save model to file 
        with open('models/tf_idf_mdl.pkl', 'wb') as f:
            pickle.dump(model_tfidf, f)
    with tab2:
        X = vectorize_embedding(df)
        texts = df["Mots_importants"]
        X_train, X_test, y_train, y_test, texts_train,texts_test = train_test_split(X, y,texts, test_size=0.2, random_state=33)

        # Define the model
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.2))
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))


        # Compiler le modèle
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entraîner le modèle
        def train_embedding_model(X_train, y_train):
            history = model.fit(X_train, y_train, epochs=30, validation_split=0.1, 
                      verbose=1, callbacks=[early_stopping])
            return history
            
        history = train_embedding_model(X_train, y_train)

        # Évaluer le modèle
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Perte sur le test : {loss:.4f}")
        st.write(f"Précision sur le test : {accuracy:.4f}")
        #f1 score   


        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        st.write(f"Score F1 : {f1:.4f}")

        st.write("Rapport de classification :")

        report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
        st.text(report)
        # Display the training history

        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history.history['accuracy'], label='Précision sur l\'entraînement')
        ax[0].plot(history.history['val_accuracy'], label='Précision sur la validation')
        ax[0].set_title('Précision du modèle')
        ax[0].set_ylabel('Précision')
        ax[0].set_xlabel('Époque')
        ax[0].legend(loc='upper left')
        ax[1].plot(history.history['loss'], label='Perte sur l\'entraînement')
        ax[1].plot(history.history['val_loss'], label='Perte sur la validation')
        ax[1].set_title('Perte du modèle')
        ax[1].set_ylabel('Perte')
        ax[1].set_xlabel('Époque')
        ax[1].legend(loc='upper left')

        # SHAP explanation for embedding model
        st.write("Explication des résultats avec SHAP pour le modèle d'embedding")

        # Charge SentenceTransformer
        embedding_model = SentenceTransformer("dangvantuan/sentence-camembert-base")

        # Wrapper de prédiction prenant en entrée du texte brut
        def predict_from_text(texts):
            # Texte → Embeddings
            X_embed = embedding_model.encode(texts)
            # Embeddings → Prédictions via ton modèle Keras
            return model.predict(X_embed)


        def get_shap_values_for_text( texts_test):
            tokenizer = AutoTokenizer.from_pretrained("dangvantuan/sentence-camembert-base")

            # Utilise shap.maskers.Text pour permettre l'interprétation mot à mot
            masker = shap.maskers.Text(tokenizer)
            # Create explainer from raw text
            explainer_text = shap.Explainer(predict_from_text, masker)
            # Calculate SHAP values
            shap_values_text = explainer_text(texts_test.tolist()[:10])
            return shap_values_text

        # Get SHAP values using the cached function
        shap_values_text = get_shap_values_for_text(  texts_test)
        # # Create a SHAP explainer using the model and the training data
        # explainer_embedding = shap.Explainer(model.predict, X_train[:40])
        
        # # Calculate SHAP values for the test data
        # shap_values_embedding = explainer_embedding(X_test[:3], max_evals=5000)
        
        # # Plot the SHAP values for the first instance in the test data
        shap.initjs()
        st.write("Valeurs SHAP pour la première instance des données de test (modèle d'embedding) :")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values_text[0, :, 0], max_display=20, show=False)
        st.pyplot(fig)

        
        # Save model to file
        with open('models/embedding_mdl.pkl', 'wb') as f:
            pickle.dump(model, f)
    with tab3:
        st.write("Comparaison des modèles")
        results_df = pd.DataFrame({
            'Model': ['TF-IDF', 'Embedding'],
            'F1 Score': [f1_tfidf, f1],
            'Accuracy': [accuracy_tfidf, accuracy]
        })
        st.dataframe(results_df)
        st.text("Le modèle TF-IDF a montré de meilleures performances en termes de précision et de score F1 par rapport au modèle d'embedding.")
        st.text("Les valeurs SHAP fournissent également des informations sur l'importance des caractéristiques pour chaque modèle.")
        st.text("L'explication SHAP pour le modèle d'embedding est difficile à interpréter car elle nous ressort des \"features\" qui ne sont pas compréhensibles par l'homme.")
        st.text("Il est donc préférable d'utiliser le modèle TF-IDF pour la prédiction des sentiments dans les avis Trustpilot.")