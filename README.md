# FeedForward — Internship Feedback Sentiment & Insights
Sentiment analysis & thematic insight mining from intern feedback using TF-IDF, Logistic Regression, and Transformers.

## Overview
FeedForward analyzes a dataset of 100,000 synthetic internship feedback entries, generated using ChatGPT, to:
- Classify feedback into Positive, Neutral, or Negative sentiments
- Extract recurring themes from negative feedback to identify key areas of dissatisfaction

We use both traditional machine learning and state-of-the-art transformer models to gain meaningful insights from feedback data.  
For theme extraction, we exclusively used the fine-tuned BERT model to classify and filter negative feedback before extracting the insufficient aspects.

Note: Since the dataset is synthetic, the models—particularly the deep learning model—can exhibit some signs of overfitting.

Additionally, NLTK's corpora and tokenizer data were downloaded and used during text preprocessing.

## What We Did
1. **Data Preprocessing**
   - Cleaned and preprocessed the synthetic feedback dataset
   - Generated:
     - TF-IDF features for Logistic Regression
     - Tokenized inputs for DistilBERT

2. **Model Training**
   - Logistic Regression for efficient feature-based sentiment classification
   - DistilBERT fine-tuned for deep-learning-based sentiment classification
   - Both models are saved for inference

3. **Sentiment Analysis**
   - Classified all feedback entries into sentiments
   - Identified approximately 33,000 negative reviews

4. **Theme Extraction**
   - Used the fine-tuned BERT model to classify and filter negative feedback
   - Applied TF-IDF vectorization to extract the most common themes representing dissatisfaction

## Purpose
The project provides a comprehensive NLP pipeline to:
- Understand intern satisfaction
- Identify recurring problem areas
- Deliver actionable insights to improve internship programs

## Tools & Libraries
- Python
- pandas
- scikit-learn
- transformers (Hugging Face)
- PyTorch
- NLTK
- Jupyter Notebooks

## Quick Start
1. Run `main.ipynb` to preprocess data, train/load models, and classify sentiments.
2. Run `analyze data/analyze.ipynb` to extract dissatisfaction themes from negative feedback.

Ensure:
- File paths to the project root are correct
- The trained model paths are accurately referenced in the analysis notebook

## Author
Crafted with care by Moazam — NLP enthusiast & ML practitioner in this economy. 😮‍💨
>Transforming feedback into insights.
