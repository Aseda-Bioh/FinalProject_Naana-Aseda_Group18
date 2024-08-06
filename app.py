#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from gensim import corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
import nltk
import re

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and parse publication dates
def clean_and_parse_dates(date_str):
    try:
        date_str = date_str.replace('Politics of ', '')
        return datetime.strptime(date_str, '%A, %d %B %Y')
    except ValueError:
        return None

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    new_text = re.sub(r'[^\w\s]', '', text.lower())
    word_tokens = word_tokenize(new_text)
    tokens_without_stopwords = [word for word in word_tokens if word not in stop_words]
    return ' '.join(tokens_without_stopwords)

# Function to get document topics
def get_document_topics(model, corpus):
    doc_topics = []
    for doc in corpus:
        topic_probs = model.get_document_topics(doc, minimum_probability=0.0)
        most_probable_topic = max(topic_probs, key=lambda x: x[1])[0]
        doc_topics.append(most_probable_topic)
    return doc_topics

# Load the data and model
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Ghana_Political_News.csv')
        # Clean and parse 'publication_date'
        df['publication_date'] = df['publication_date'].apply(clean_and_parse_dates)
        df = df.dropna(subset=['publication_date'])  # Drop rows where 'publication_date' could not be parsed
        df['Cleaned_contents'] = df['contents'].apply(preprocess_text)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

@st.cache_resource
def load_model():
    try:
        model = joblib.load('ExponentialSmoothing.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None  # Return None on error

@st.cache_resource
def train_lda_model(df):
    tokenized_contents = df['Cleaned_contents'].apply(word_tokenize)
    dictionary = corpora.Dictionary(tokenized_contents)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_contents]
    num_topics = 5
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
    topic_labels = [' '.join([word for word, _ in topic[1]]) for topic in lda_model.show_topics(num_words=10, formatted=False)]
    df['Topic_Index'] = get_document_topics(lda_model, corpus)
    df['Topic'] = df['Topic_Index'].map(lambda idx: topic_labels[idx])
    return df

def plot_topic_trends(df):
    if df.empty:
        st.error("No data to plot.")
        return

    new_df = df.set_index('publication_date')
    topic_trends = new_df.groupby('Topic_Index').resample('D').size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 7))
    for topic in topic_trends.columns:
        plt.plot(topic_trends.index, topic_trends[topic], label=f'Topic {topic}')
    plt.title('Topic Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()  # Clear the figure to avoid overlap

def plot_forecast(topic_trends, model):
    if model is None:
        st.error("Model not loaded.")
        return

    forecast_period = 30
    for topic in topic_trends.columns:
        if len(topic_trends[topic]) > 2:
            train_size = int(len(topic_trends[topic]) * 0.6)
            train, test = topic_trends[topic][:train_size], topic_trends[topic][train_size:]

            model_fit = ExponentialSmoothing(train, trend='add', seasonal=None, seasonal_periods=None).fit()
            forecast = model_fit.forecast(steps=len(test))
            future_forecast = model_fit.forecast(steps=forecast_period)

            plt.figure(figsize=(14, 7))
            plt.plot(topic_trends.index[:train_size], train, label='Train')
            plt.plot(topic_trends.index[train_size:], test, label='Test')
            plt.plot(topic_trends.index[train_size:], forecast, label='Forecast')
            
            # Ensure topic_trends.index is a datetime index
            future_index = pd.date_range(start=pd.to_datetime(topic_trends.index[-1]) + timedelta(days=1), periods=forecast_period, freq='D')
            plt.plot(future_index, future_forecast, label='Future Forecast')

            plt.title(f'Forecast for Topic {topic}')
            plt.xlabel('Date')
            plt.ylabel('Number of Articles')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            plt.clf()  # Clear the figure to avoid overlap

def main():
    st.title('Ghanaian News Analysis')
    st.write('This app visualizes the trends and forecasts of news topics.')

    # Load and display data
    df = load_data()
    if not df.empty:
        df = train_lda_model(df)
        st.write(df.head())

        # Plot topic trends
        st.subheader('Topic Trends Over Time')
        if 'publication_date' in df.columns:
            plot_topic_trends(df)

        # Load model and plot forecast
        model = load_model()
        if model:
            new_df = df.set_index('publication_date')
            st.subheader('Topic Forecast')
            if 'publication_date' in df.columns:
                topic_trends = new_df.groupby('Topic_Index').resample('D').size().unstack(fill_value=0)
                plot_forecast(topic_trends, model)

if __name__ == '__main__':
    main()


# In[ ]:




