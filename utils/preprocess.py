import subprocess
import sys
import pkg_resources
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import torch
import pickle
import os
import shutil

def install_requirements():
    """Check and install dependencies from requirements.txt in the project root."""
    try:
        requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing = []
        
        for req in requirements:
            pkg_name = req.split('>=')[0].split('==')[0].strip()
            try:
                pkg_resources.require(req)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                missing.append(req)
        
        if missing:
            print(f"Installing missing packages: {missing}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        else:
            print("All required packages are already installed.")
            
    except FileNotFoundError:
        print(f"Error: requirements.txt not found at {requirements_path}. Please ensure it exists in the project root.")
        raise
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        raise

def download_nltk_resources(require_wordnet=True):
    """Download required NLTK resources to the project root or default path if not already present."""
    nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
    default_nltk_path = os.path.expanduser('~/nltk_data')
    nltk.data.path = [nltk_data_path, default_nltk_path]
    
    # Clear NLTK cache
    nltk_cache_path = os.path.expanduser('~/.nltk_data')
    if os.path.exists(nltk_cache_path):
        print(f"Clearing NLTK cache at {nltk_cache_path}")
        shutil.rmtree(nltk_cache_path)
    
    resources = [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    for resource, resource_path in resources:
        if resource == 'wordnet' and not require_wordnet:
            print(f"Skipping '{resource}' as it is not required.")
            continue
        
        resource_dir = os.path.join(nltk_data_path, resource_path)
        if os.path.exists(resource_dir):
            print(f"NLTK resource '{resource}' already exists at {resource_dir}.")
            print(f"Contents of {resource_dir}: {os.listdir(resource_dir) if os.path.exists(resource_dir) else 'Empty'}")
            continue
        
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource '{resource}' already available in {nltk.data.path}.")
        except LookupError:
            print(f"Downloading NLTK resource '{resource}' to {nltk_data_path}...")
            print(f"Pre-download contents of {nltk_data_path}: {os.listdir(nltk_data_path) if os.path.exists(nltk_data_path) else 'Empty'}")
            for attempt in range(3):
                try:
                    if os.path.exists(resource_dir):
                        shutil.rmtree(resource_dir)
                    zip_path = os.path.join(nltk_data_path, f"{resource}.zip")
                    if os.path.exists(zip_path):
                        print(f"Removing existing {zip_path}")
                        os.remove(zip_path)
                    os.makedirs(nltk_data_path, exist_ok=True)
                    nltk.download(resource, download_dir=nltk_data_path, quiet=False, force=True)
                    if os.path.exists(resource_dir):
                        print(f"NLTK resource '{resource}' downloaded successfully to {nltk_data_path}.")
                        print(f"Post-download contents of {resource_dir}: {os.listdir(resource_dir)}")
                        break
                    else:
                        print(f"Attempt {attempt + 1}: Failed to verify '{resource}' at {resource_dir}.")
                        if os.path.exists(zip_path):
                            print(f"Zip file found at {zip_path}, indicating extraction failure.")
                except Exception as e:
                    print(f"Attempt {attempt + 1}: Failed to download '{resource}' to {nltk_data_path}: {e}")
                if attempt == 2:
                    print(f"Exhausted retries for '{resource}' in {nltk_data_path}. Trying default path...")
                    resource_dir = os.path.join(default_nltk_path, resource_path)
                    zip_path = os.path.join(default_nltk_path, f"{resource}.zip")
                    for default_attempt in range(3):
                        try:
                            if os.path.exists(resource_dir):
                                shutil.rmtree(resource_dir)
                            if os.path.exists(zip_path):
                                print(f"Removing existing {zip_path}")
                                os.remove(zip_path)
                            os.makedirs(default_nltk_path, exist_ok=True)
                            nltk.download(resource, download_dir=default_nltk_path, quiet=False, force=True)
                            if os.path.exists(resource_dir):
                                print(f"NLTK resource '{resource}' downloaded successfully to {default_nltk_path}.")
                                print(f"Post-download contents of {resource_dir}: {os.listdir(resource_dir)}")
                                break
                            else:
                                print(f"Default attempt {default_attempt + 1}: Failed to verify '{resource}' at {resource_dir}.")
                                if os.path.exists(zip_path):
                                    print(f"Zip file found at {zip_path}, indicating extraction failure.")
                        except Exception as e2:
                            print(f"Default attempt {default_attempt + 1}: Failed to download '{resource}' to {default_nltk_path}: {e2}")
                        if default_attempt == 2:
                            if resource == 'wordnet' and not require_wordnet:
                                print(f"Warning: Failed to download '{resource}'. Continuing without it as it is not required.")
                                break
                            else:
                                print(f"Error: Failed to download '{resource}' after retries. Continuing without it to avoid crashing.")
                                break
                    else:
                        continue
            else:
                continue

def load_and_clean_data(file_path):
    """Load CSV and remove duplicate entries."""
    df = pd.read_csv(file_path)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['Feedback', 'Sentiment'], keep='first')
    print(f"Removed {initial_len - len(df)} duplicate entries. {len(df)} entries remain.")
    df = df.dropna(subset=['Feedback', 'Sentiment'])
    return df

def clean_text(text):
    """Clean text by removing punctuation, normalizing case, and handling special characters."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_for_logistic_regression(df):
    """Preprocess text for Logistic Regression using TF-IDF."""
    df['Cleaned_Feedback'] = df['Feedback'].apply(clean_text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    has_wordnet = any(os.path.exists(os.path.join(p, 'corpora/wordnet')) for p in nltk.data.path)
    if not has_wordnet:
        print("Warning: WordNet not available, skipping lemmatization.")
    
    def process_text(text):
        tokens = word_tokenize(text)
        if has_wordnet:
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        else:
            tokens = [token for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    df['Processed_Feedback'] = df['Cleaned_Feedback'].apply(process_text)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['Processed_Feedback'])
    sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    y = df['Sentiment'].map(sentiment_map)
    
    # Save to the utils folder
    output_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data_tfidf.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({'X_tfidf': X_tfidf, 'y': y, 'tfidf': tfidf}, f)
    print(f"TF-IDF preprocessed data saved to {output_path}")
    
    return X_tfidf, y, tfidf

def preprocess_for_transformers(df, max_length=128):
    """Preprocess text for Transformer-based models using BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df['Cleaned_Feedback'] = df['Feedback'].apply(clean_text)
    encodings = tokenizer(
        df['Cleaned_Feedback'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    labels = df['Sentiment'].map(sentiment_map).values
    
    # Save to the utils folder
    output_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data_transformer.pt')
    torch.save({'input_ids': encodings['input_ids'], 
                'attention_mask': encodings['attention_mask'], 
                'labels': labels}, 
               output_path)
    print(f"Transformer preprocessed data saved to {output_path}")
    
    return encodings, labels, tokenizer