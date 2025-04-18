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

from sklearn.metrics import classification_report   
# Vectorization
@st.cache_data
def vectorize_tfidf(df):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.3)
    X = tfidf_vectorizer.fit_transform(df['Mots_importants'])
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

    # Train a first model using TF-IDF features
    tab1, tab2 = st.tabs(["Model with TF-IDF Features", "Model with Embedding Features"])
    
    with tab1:

        # Split TF-IDF data
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=33)

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
        history_tfidf = model_tfidf.fit(X_train_tfidf, y_train, epochs=30, validation_split=0.1, 
                                    verbose=1, callbacks=[early_stopping])

        # Evaluate
        loss_tfidf, accuracy_tfidf = model_tfidf.evaluate(X_test_tfidf, y_test, verbose=0)
        st.write(f"TF-IDF Test Loss: {loss_tfidf:.4f}")
        st.write(f"TF-IDF Test Accuracy: {accuracy_tfidf:.4f}")

        # F1 Score
        y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
        y_pred_classes_tfidf = np.argmax(y_pred_tfidf, axis=1)
        f1_tfidf = f1_score(y_test, y_pred_classes_tfidf, average='weighted')
        st.write(f"TF-IDF F1 Score: {f1_tfidf:.4f}")

        # Classification Report
        st.write("TF-IDF Classification Report:")
        report_tfidf = classification_report(y_test, y_pred_classes_tfidf, target_names=label_encoder.classes_)
        st.text(report_tfidf)

        # Plot training history
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history_tfidf.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history_tfidf.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('TF-IDF Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(loc='upper left')

        ax[1].plot(history_tfidf.history['loss'], label='Train Loss')
        ax[1].plot(history_tfidf.history['val_loss'], label='Validation Loss')
        ax[1].set_title('TF-IDF Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(loc='upper left')
        st.pyplot(fig)
        
        #explication des résultats avec shap
        
        # Create a SHAP explainer using the model and the training data
        explainer = shap.Explainer(model_tfidf.predict, X_train_tfidf[:40])
        # Calculate SHAP values for the test data
        shap_values = explainer(X_test_tfidf[:3],max_evals=1000)
        # Plot the SHAP values for the first instance in the test data
        shap.initjs()
        st.write("SHAP values for the first instance in the test data:") 
        print(type(shap_values))
        print(shap_values.shape)
        fig, ax = plt.subplots()   
        shap.plots.waterfall(shap_values[0,:,0], max_display=20,show=False)
        st.pyplot(fig)
    with tab2:
        X = vectorize_embedding(df)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

        # Define the model
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.2))
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))



        # afficher la structure du modèle
        model.summary(print_fn=lambda x: st.text(x))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=30,  validation_split=0.1, verbose=1,callbacks=[early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Loss: {loss:.4f}")
        st.write(f"Test Accuracy: {accuracy:.4f}")
        #f1 score   


        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')
        st.write(f"F1 Score: {f1:.4f}")

        st.write("Classification Report:")

        report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
        st.text(report)
        # Display the training history

        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].set_title('Model Accuracy')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(loc='upper left')
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Model Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(loc='upper left')
        # st.pyplot(fig)
        # # explication des résultats avec shap
        # explainer = shap.DeepExplainer(model, X_train[:5])
        # shap_values = explainer.shap_values(X_test[5:10])
        # shap.initjs()
        # st.write("SHAP values for the first instance in the test data:")
        # fig, ax = plt.subplots()
        # shap.plots.waterfall(shap_values[0,:,0], max_display=10,show=False)
        # st.pyplot(fig)

   