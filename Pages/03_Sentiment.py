import streamlit as st
import pandas as pd 
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


if 'reviews_df' in st.session_state:
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('stopwords')

    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Function to analyze sentiment
    def analyze_sentiment(text):
        scores = sid.polarity_scores(text)
        return scores

    # Function to extract themes
    def extract_themes(text):
        stop_words = set(stopwords.words('french'))
        word_tokens = word_tokenize(text,language="french")
        filtered_words = [w for w in word_tokens if not w.lower() in stop_words and w.isalpha()]
        return filtered_words

    # Create a temporary copy of the DataFrame
    temp_df = st.session_state['reviews_df'].copy()

    # Apply sentiment analysis and theme extraction
    temp_df = temp_df[temp_df['Avis'].notna() & (temp_df['Avis'].str.strip() != '')]
    temp_df['sentiment'] = temp_df['Avis'].apply(analyze_sentiment)
    temp_df['themes'] = temp_df['Avis'].apply(extract_themes)

    st.dataframe(temp_df)