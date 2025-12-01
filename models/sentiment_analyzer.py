# models/sentiment_analyzer.py
from transformers import pipeline

# Do NOT force framework="tf"
# Hugging Face will use PyTorch automatically
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_review_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"], float(result["score"])

# pip install hf_xet 