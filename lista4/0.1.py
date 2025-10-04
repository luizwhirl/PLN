# CountVectorizer e TF-IDF

# !pip install -q transformers sentence-transformers scikit-learn tensorflow

import pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('/content/dataset.csv')
df = df[['Comment','sentiment']].dropna().rename(columns={'Comment':'text','sentiment':'label'})

def preprocess(text):
    text = str(text).replace('\n',' ')
    text = re.sub(r'http\S+',' ', text)
    text = re.sub(r'\[.*?\]\(.*?\)',' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]',' ', text)
    return text.lower()

df['text_pp'] = df['text'].apply(preprocess)
df.head()
