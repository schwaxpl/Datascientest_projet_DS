# Data manipulation and analysis
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Text Processing
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# Metrics and Visualization
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Web App
import streamlit as st

# Count the number of reviews and responses containing each theme
def count_theme_occurrences(tfidf_vectorizer, tfidf_matrix, themes):
    feature_array = tfidf_vectorizer.get_feature_names_out()
    theme_counts = {}
    for theme in themes:
        if theme in feature_array:
            index = list(feature_array).index(theme)
        count = (tfidf_matrix[:, index] > 0).sum()
        theme_counts[theme] = count
    return theme_counts
# Add the top 5 themes for each row to new columns
def get_top_themes(tfidf_vectorizer, tfidf_matrix, top_n=5):
    feature_array = tfidf_vectorizer.get_feature_names_out()
    top_themes_per_row = []
    for row in tfidf_matrix:
        sorted_indices = row.toarray().flatten().argsort()[::-1]
        top_features = [feature_array[i] for i in sorted_indices[:top_n]]
        top_themes_per_row.append(top_features)
    return top_themes_per_row
def vectorize_text(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
#deprecated
def analyze_themes(df_subset, label):
        # Vectorize the 'Mots_importants' and 'Mots_importants_reponse' columns
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_subset['Mots_importants'])
        y = vectorizer.transform(df_subset['Mots_importants_reponse'])

        # Vectorize 'Mots_importants' using TF-IDF
        tfidf_vectorizer_avis = TfidfVectorizer(max_df=0.3,ngram_range=(1,2))
        X_tfidf = tfidf_vectorizer_avis.fit_transform(df_subset['Mots_importants'])

        # Vectorize 'Mots_importants_reponse' using TF-IDF
        tfidf_vectorizer_reponse = TfidfVectorizer(max_df=0.3)
        y_tfidf = tfidf_vectorizer_reponse.fit_transform(df_subset['Mots_importants_reponse'])

        # Extract themes using the most frequent words from TF-IDF vectors
        def extract_themes(tfidf_vectorizer, tfidf_matrix, top_n=50):
            feature_array = tfidf_vectorizer.get_feature_names_out()
            sorted_indices = tfidf_matrix.sum(axis=0).A1.argsort()[::-1]
            top_features = [feature_array[i] for i in sorted_indices[:top_n]]
            return top_features

        themes_avis = extract_themes(tfidf_vectorizer_avis, X_tfidf)
        themes_reponse = extract_themes(tfidf_vectorizer_reponse, y_tfidf)

        df_subset['Themes_Avis'] = get_top_themes(tfidf_vectorizer_avis, X_tfidf, top_n=10)
        df_subset['Themes_Reponse'] = get_top_themes(tfidf_vectorizer_reponse, y_tfidf, top_n=2)

        themes_avis_counts = count_theme_occurrences(tfidf_vectorizer_avis, X_tfidf, themes_avis)
        themes_reponse_counts = count_theme_occurrences(tfidf_vectorizer_reponse, y_tfidf, themes_reponse)

        # Sort themes by their counts
        sorted_themes_avis = sorted(themes_avis_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_themes_reponse = sorted(themes_reponse_counts.items(), key=lambda x: x[1], reverse=True)

        # Convert sorted themes to DataFrame
        df_themes_avis = pd.DataFrame(sorted_themes_avis, columns=["Thème", "Fréquence"])
        df_themes_reponse = pd.DataFrame(sorted_themes_reponse, columns=["Thème", "Fréquence"])
        st.write(f"**{label}** : Thèmes extraits classés par fréquence")

        # Display the themes side by side as DataFrames
        col1, col2 = st.columns(2)

        with col1:
            st.write("Avis :")
            st.dataframe(df_themes_avis)
        with col2:
            st.write("Réponses :")
            st.dataframe(df_themes_reponse)
st.title("Création du modèle de prédiction")
if 'reviews_df' not in st.session_state:
    st.error("Veuillez importer des données contenant les avis Trustpilot.")
else:
    df = st.session_state['reviews_df'].copy()
    df = df[df['Avis'].notna()]

    # Dropdown for vectorization method
    vectorization_method = st.selectbox(
        "Choisir la méthode de vectorisation",
        ["TF-IDF", "Word2Vec", "Embedding"]
    )

    # Dropdown for model selection
    model_type = st.selectbox(
        "Choisir le modèle",
        ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "XGBoost", "Tous"],
        index=0
    )

    entrainement = st.empty()
    entrainement.text("Préparation des données...")

    # Vectorization
    @st.cache_data
    def vectorize_tfidf(df):
        tfidf_vectorizer = TfidfVectorizer(max_df=0.3)
        X = tfidf_vectorizer.fit_transform(df['Mots_importants'])
        return pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    @st.cache_data
    def vectorize_word2vec(df):
        df["tokens"] = df["Mots_importants"].apply(word_tokenize)
        model = Word2Vec(sentences=df["tokens"], vector_size=200, window=5, workers=-1, sg=1, epochs=10)
        vectors = df["tokens"].apply(lambda x: vectorize_text(x, model))
        X = np.array(vectors.tolist())
        X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        return np.nan_to_num(X)

    @st.cache_data
    def vectorize_embedding(df):
        df["tokens"] = df["Mots_importants"].apply(word_tokenize)
        model = SentenceTransformer("dangvantuan/sentence-camembert-base")
        embeddings = df["tokens"].apply(lambda x: model.encode(" ".join(x), show_progress_bar=False))
        return np.array(embeddings.tolist())

    # Get features based on selected vectorization method
    if vectorization_method == "TF-IDF":
        X = vectorize_tfidf(df)
    elif vectorization_method == "Word2Vec":
        X = vectorize_word2vec(df)
    else:
        X = vectorize_embedding(df)

    # Convert sentiment labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Sentiment'].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    # Model training
    entrainement.text("Entrainement du modèle...")
    
    def create_model_config():
        return {
            "Random Forest": {
                "model": RandomForestClassifier(random_state=33),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            "SVM": {
                "model": SVC(random_state=33),
                "params": {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            },
            "Logistic Regression": {
                "model": LogisticRegression(random_state=33),
                "params": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=33),
                "params": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(random_state=33),
                "params": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
    models_config = create_model_config()
    @st.cache_data
    def train_model_with_gridsearch(X_train, y_train, _model, param_grid):
        clf = GridSearchCV(_model, param_grid, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)
        return clf

    if model_type == "Tous":
        best_score = 0
        best_model = None
        best_params = None
        
        results = []
        for name, config in models_config.items():
            entrainement.text(f"Entrainement du modèle {name}...")
            clf = train_model_with_gridsearch(X_train, y_train, config["model"], config["params"])
            score = clf.score(X_test, y_test)
            results.append({"name": name, "score": score, "params": clf.best_params_, "model": clf})
            
            if score > best_score:
                best_score = score
                best_model = clf
                best_params = clf.best_params_
            st.write(f"\nRésultats pour {name}:")
            st.write(f"Score: {score:.4f}")
            st.write("Meilleurs paramètres:", clf.best_params_)

        results_df = pd.DataFrame([(r["name"], r["score"]) for r in results], 
                                columns=["Modèle", "Score"])
        st.write("Résultats de tous les modèles:")
        st.dataframe(results_df)
        
        st.write(f"Meilleur modèle: {results_df.iloc[results_df['Score'].argmax()]['Modèle']}")
        clf = best_model
        
    else:
        config = models_config[model_type]
        clf = train_model_with_gridsearch(X_train, y_train, config["model"], config["params"])

    y_pred = clf.predict(X_test)

    # Results display
    entrainement.text("Modèle entrainé !")
    st.write("Meilleurs paramètres:", clf.best_params_)
    st.write(f"F1 Score: {clf.score(X_test, y_test):.2f}")

    # Classification report
    st.write("Rapport de classification:")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(np.unique(y_test)), 
                yticklabels=sorted(np.unique(y_test)), ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    