import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

def scrape_youtube_comments(video_url, progress_bar):
    if video_url is None:
        st.warning('Please enter a YouTube Video URL.')
        return []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(video_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the element containing comments (you might need to inspect the YouTube page to get the correct selector)
    comment_elems = soup.select('#content #content-text')

    comments = []
    for comment_elem in comment_elems:
        comments.append(comment_elem.text)

    return comments

def classify_comments(comments):
    analyzer = SentimentIntensityAnalyzer()
    labeled_comments = []

    for comment in comments:
        vs = analyzer.polarity_scores(comment)
        if vs['compound'] >= 0.05:
            sentiment = 'positive'
        elif vs['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        labeled_comments.append({'Comment': comment, 'Sentiment': sentiment})

    return labeled_comments

def comments_to_dataframe(labeled_comments):
    df = pd.DataFrame(labeled_comments, columns=['Comment','Sentiment'])
    return df

def visualise(df):
    df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=['positive', 'negative', 'neutral'])
    sentiment_counts = df['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

# Streamlit app
st.title('YouTube Comment Sentiment Analysis')

video_url = st.text_input('Enter YouTube Video URL:')
if st.button('Analyze'):
    if video_url:
        progress_bar = st.progress(0)  # Initialize progress bar
        comments = scrape_youtube_comments(video_url, progress_bar)
        labeled_comments = classify_comments(comments)
        df = comments_to_dataframe(labeled_comments)
        visualise(df)
    else:
        st.warning('Please enter a YouTube Video URL.')
