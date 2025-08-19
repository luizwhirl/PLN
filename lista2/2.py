import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('rslp')


df = pd.read_csv('../dataset.csv')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([word for word in text.split() if word not in stop_words])

df['processed_comment'] = df['Comment'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf = vectorizer.fit_transform(df['processed_comment'])
feature_names = vectorizer.get_feature_names_out()

num_topics = 10
nmf = NMF(n_components=num_topics, random_state=42)
nmf_matrix = nmf.fit_transform(tfidf)

def get_top_words(model, feature_names, n_top_words):
    top_words = {}
    for topic_idx, topic in enumerate(model.components_):
        top_words[f'Tópico {topic_idx + 1}'] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return top_words

top_words_nmf = get_top_words(nmf, feature_names, 5)

print("as 5 palavras mais relevantes por tópico (NMF):")
for topic, words in top_words_nmf.items():
    print(f"{topic}: {', '.join(words)}")

print("\nas 5 palavras mais relevantes por documento (NMF):")
for i in range(5):
    doc_topic_dist = nmf_matrix[i]
    dominant_topic = np.argmax(doc_topic_dist)
    print(f"Documento {i}:")
    print(f"  texto original: {df['Comment'].iloc[i][:100]}...")
    print(f"  tópico mais relevante: {dominant_topic + 1}")
    print("-" * 20)