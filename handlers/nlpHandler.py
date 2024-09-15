from flask import Flask, request, jsonify
import spacy
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the transformer model for sentiment analysis
sentiment_pipeline = pipeline('sentiment-analysis')

# Load the transformer model
nlp = spacy.load("en_core_web_trf")
analyzer = SentimentIntensityAnalyzer()  # Initialize VADER Sentiment Analyzer


# Text Preprocessing Function
def preprocess_text(text, useTransformers):

    # Remove negations from SpaCy's stop words list
    if useTransformers==False:
        negations = ['not', 'no', 'never']
        for neg in negations:
            if neg in nlp.Defaults.stop_words:
                nlp.vocab[neg].is_stop = False

    # processing text
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    if useTransformers == False:
        return ' '.join(tokens)
    else: 
        return {
            'original_text': text,
            'tokens': [token.text for token in doc],
            'lemmas': [token.lemma_ for token in doc],
            'stop_words': [token.text for token in doc if token.is_stop],
            'punctuations': [token.text for token in doc if token.is_punct],
            'preprocessed_text': ' '.join(tokens)
        }

# Sentiment Analysis Function
def analyze_sentiment(preprocessed_text):
    sentiment_scores = analyzer.polarity_scores(preprocessed_text)
    compound_score = sentiment_scores['compound']

    # Classify sentiment based on compound score
    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment, sentiment_scores



# Using VADER
def predict_sentiment_vader(data):
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess the input text
    preprocessed_text = preprocess_text(text, False)
    
    sentiment, sentiment_scores = analyze_sentiment(preprocessed_text)
    
    return jsonify({
        'text': text,
        'preprocessed_text': preprocessed_text,
        'sentiment': sentiment.upper(),
        'sentiment_scores': sentiment_scores 
    })


# using transformer
def predict_sentiment_transformers(data):
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Preprocess the input text
    preprocessed_text = preprocess_text(text, True)
    

    # Perform sentiment analysis using the transformer model
    result = sentiment_pipeline(preprocessed_text['preprocessed_text'])

    # Extract sentiment from the tranformer result
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']

    return jsonify({
        'text': preprocessed_text['original_text'],
        'preprocessing': preprocessed_text,
        'sentiment': sentiment_label,
        'score': sentiment_score
    })



def predict_sentiment_csv(requestFiles):
    # Check if a file is part of the request
    if 'file' not in requestFiles:
        return jsonify({'error': 'No file uploaded'}), 400

    file = requestFiles['file']
    
    # Check if the file is a valid CSV file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):

        # Read the CSV file into a pandas DataFrame
        df =  df = pd.read_csv(file)

        # Ensure the CSV file has a column named 'review_text' or modify this as needed
        if 'review_text' not in df.columns:
            return jsonify({'error': 'CSV must contain a "review_text" column'}), 400

        # Perform sentiment analysis on each review
        sentiments = []
        for index, row in df.iterrows():
            text = row['review_text']
            preprocessed_text = preprocess_text(text)
            sentiment, sentiment_scores = analyze_sentiment(preprocessed_text)
            sentiments.append({
                'text': text,
                'sentiment': sentiment,
                'sentiment_scores': sentiment_scores
            })

        return jsonify({
            'message': 'Sentiment analysis completed on CSV',
            'sentiments': sentiments
        })

    return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'}), 400


