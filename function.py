import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import re
import neattext.functions as nfx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from translate import Translator
from langdetect import detect 

import joblib

# def predict_with_model(text):
#     # Load the model
#     model = joblib.load('clf.pkl')
#     # Make predictions
#     predictions = model.predict(text)
#     # Define a mapping between numeric labels and sentiment categories
#     sentiment_mapping = {
#         0: 'Negative',
#         1: 'Neutral',
#         2: 'Positive'
#     }
#     # Convert numeric labels to sentiment categories
#     predicted_sentiments = [sentiment_mapping[label] for label in predictions]
#     return predicted_sentiments

# def predict_with_model(text_list):
#     # Load the model
#     model = joblib.load('clf.pkl')
#     # Make predictions
#     predictions = model.predict(text_list)
#     # Define a mapping between numeric labels and sentiment categories
#     sentiment_mapping = {
#         1: 'Negative',
#         2: 'Neutral',
#         3: 'Positive'
#     }
#     # Convert numeric labels to sentiment categories for each prediction
#     predicted_sentiments = [sentiment_mapping[label] for label in predictions]
#     return predicted_sentiments

import joblib

def predict_with_model(text_list):
    # Load the model pipeline (which includes the vectorizer)
    model = joblib.load('clf.pkl')
    
    # Make predictions
    predictions = model.predict(text_list)
    
    # Define a mapping between numeric labels and sentiment categories
    sentiment_mapping = {
        0: 'Negative',  # Assuming label 0 for Negative
        1: 'Neutral',   # Assuming label 1 for Neutral
        2: 'Positive'   # Assuming label 2 for Positive
    }
    
    # Convert numeric labels to sentiment categories for each prediction
    predicted_sentiments = [sentiment_mapping[label] for label in predictions]
    
    return predicted_sentiments


    
# Create a function to translate the tweets
def translate_text(text):
    try:
        detected_language = detect(text)
        translator = Translator(to_lang="en", from_lang=detected_language)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print("An error occurred during translation:", e)
        return text

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print("An error occurred during language detection:", e)
        return None

# Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text

# Create a function to get the subjectivity
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

def generate_wordcloud(text):
    allWords = ' '.join([twts for twts in text]).strip()
    mywordCloud = WordCloud(width=500, height=300, max_font_size=110).generate(allWords)
    
    # Convert the word cloud to a NumPy array
    wordcloud_array = mywordCloud.to_array()

    # Display the word cloud image using Streamlit
    st.image(wordcloud_array, use_column_width=True)

# def generate_wordcloud(text):
#     # Join all the text from the input list
#     all_words = ' '.join(text)

#     # Specify stopwords for different languages
#     stopwords = set(STOPWORDS)
#     # Add stopwords for other languages as needed
#     stopwords.update(["in", "of", "and", "the", "a", "an", "on", "for", "to", "at"])

#     # Path to Arial font file (.ttf) on your system
#     font_path = "path_to_arial.ttf"  # Change this to the path of Arial font on your system

#     # Create a WordCloud object with specified stopwords, font, and parameters
#     my_wordcloud = WordCloud(width=400, height=200, max_font_size=110, stopwords=stopwords, font_path=none).generate(all_words)

#     # Convert the word cloud to a NumPy array
#     wordcloud_array = my_wordcloud.to_array()

#     # Display the word cloud image using Streamlit
#     st.image(wordcloud_array, use_column_width=True)

def generate_wordcloud_barchart(text):
    allWords = ' '.join([twts for twts in text]).strip()
    mywordCloud = WordCloud(width=500, height=300, max_font_size=110).generate(allWords)
    
    # Get word frequencies
    word_freq = mywordCloud.words_

    # Create a word cloud chart using Plotly
    fig = go.Figure(go.Bar(
        x=list(word_freq.keys()),
        y=list(word_freq.values()),
        marker=dict(color='blue'),  # You can customize the color
        ))
    fig.update_layout(title_text='Word Cloud Frequency', xaxis_title='Words', yaxis_title='Frequency')
    
    return fig

# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'

def getEmoji(score):
  if score < 0:
    return '😔'
  elif score == 0:
    return '😐'
  else:
    return '🙂'

def create_donut_chart(data):
    # Count the occurrences of each detected language
    language_counts = data['Language'].value_counts()

    # Create the donut chart
    fig = go.Figure(data=[go.Pie(labels=language_counts.index, values=language_counts, hole=.5)])

    # Update layout to add title
    fig.update_layout(title_text="Language Distribution")

    return fig

# Create a function to generate bar chart
def generate_bar_chart(data):
    counts = data['Analysis'].value_counts()
    values = counts.index

    fig = go.Figure(data=[go.Bar(x=values, y=counts.values, marker=dict(color=['blue', 'green', 'red', 'purple', 'yellow']), name=values[0])])
    fig.update_layout(title='Sentiment Analysis', xaxis_title='Sentiment', yaxis_title='Frequency')
    return fig


def generate_Comment_plot(data):
    # Assuming df is your DataFrame and column_name is the column you want to plot
    fig = px.scatter(data, x="Count(Tweets)", y=None, title="Comments")
    fig.update_layout(yaxis_title='Tweets', xaxis_title='Comments')
    return fig

def generate_likes_scatter_plot(data):
    fig = px.scatter(data, x="Likes", y=data.index, title="Likes Count")
    fig.update_layout(yaxis_title='Tweets', xaxis_title='Likes')
    return fig

from datetime import datetime
def parse_date(date_str):
    # Split the string to isolate the date part
    date_part = date_str.split('·')[0].strip()
    # Parse the date part to a datetime object
    return datetime.strptime(date_part, '%b %d, %Y').date()



def date_plot(df):
    # Create a dual-axis Plotly chart with data labels (markers)
    fig = px.line(df, x='Extracted_Date', y=['Likes', 'Count(Tweets)'], 
                  title='Dual-Axis Chart with Date and Numeric Columns',
                  labels={'Extracted_Date': 'Date', 'value': 'Value'},
                  line_shape='linear', render_mode='svg')

# Add data labels (markers) to the lines
    fig.update_traces(mode='lines+markers')

# Update layout to display two y-axes
    fig.update_layout(yaxis=dict(title='Likes', side='left', color='blue'),
                      yaxis2=dict(title='Comments', side='right', overlaying='y', color='red'),
                      hovermode="x unified")  # Unified hover mode for both axes
    return fig