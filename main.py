import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

def scrape_youtube_comments(video_url, progress_bar):
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--mute-audio")
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(video_url)

    total_iterations = 50  # Total number of scroll iterations
    for i in range(total_iterations):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)
        progress = (i + 1) / total_iterations  # Calculate progress within [0.0, 1.0] range
        progress_bar.progress(progress)  # Update progress bar

    comments = []
    comment_elems = driver.find_elements(By.CSS_SELECTOR, "#content #content-text")

    for comment_elem in comment_elems:
        comments.append(comment_elem.text)

    driver.quit()
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
