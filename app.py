from flask import Flask, request, make_response, render_template, jsonify
# import os
from handlers import nlpHandler
import pandas as pd

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route("/")
def index():
    # graph_exists = os.path.exists('static/graph.png')
    return render_template('/index.html')   


# Sentiment Prediction Route Using VADER
@app.route('/predict-sentiment-vader', methods=['POST'])
def predict_sentiment():
    return nlpHandler.predict_sentiment_vader(request.json)

# Sentiment Prediction Route Using Transformer
@app.route('/predict-sentiment-transformer', methods=['POST'])
def predict_sentiment_transformer():
    return nlpHandler.predict_sentiment_transformers(request.json)

# Route for processing a CSV file with reviews
@app.route('/predict-sentiment-csv', methods=['POST'])
def predict_sentiment_csv():
    return nlpHandler.predict_sentiment_csv(request.files)
