# 🛒 Amazon Reviews Sentiment Analysis with NLTK VADER

This project demonstrates an end-to-end Natural Language Processing (NLP) pipeline for analyzing sentiment in Amazon product reviews using Python, NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner), and Scikit-Learn.

## 🚀 Pipeline & Methodology

1. Text Preprocessing:
   - Tokenization: Splits review strings into individual word tokens using word_tokenize.
   - Case Normalization: Converts all tokens to lowercase to ensure consistency.
   - Stopword Removal: Eliminates common English words (such as "the", "is", "at") that carry minimal semantic sentiment.
   - Lemmatization: Reduces inflected words to their root forms using WordNet Lemmatizer (e.g., "batteries" -> "battery").
2. Sentiment Scoring (VADER):
   - Applies SentimentIntensityAnalyzer to compute polarity scores.
   - Maps scores into binary sentiment classes (1 for positive, 0 for negative).
3. Model Evaluation:
   - Assesses classification performance against ground truth labels using a Confusion Matrix and a detailed Classification Report (Precision, Recall, F1-Score).

## 🛠️ Tech Stack
- Python
- Pandas
- NLTK (Natural Language Toolkit)
- Scikit-Learn

## 💻 Installation & Usage

1. Install required dependencies:
pip install pandas nltk scikit-learn

2. Place sentiment_analysis_amazon_dataset.csv in the root directory.

3. Run the analysis script:
python sentiment_analysis.py

## 📊 Evaluation Metrics

The script outputs model performance metrics:
- Confusion Matrix: Shows True Positives, False Positives, True Negatives, and False Negatives.
- Classification Report: Detailed breakdown of Precision, Recall, and F1-score across sentiment classes.
