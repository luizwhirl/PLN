import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt


df = pd.read_csv('../dataset.csv', on_bad_lines='skip')
df_sample = df.sample(5000, random_state=42)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>+', '', text)  
    text = re.sub(r'[^a-z\s]', '', text) 
    return text

df_sample['processed_comment'] = df_sample['Comment'].apply(preprocess_text)

print(f"{len(df_sample['processed_comment'])} documentos para vetorizar")


# a) vetorização com tfidf
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english', use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sample['processed_comment'])

inertia_tfidf = []
silhueta_scores_tfidf = []
K_range = range(2, 11)

print("calculando kmeans para TFIDF")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)
    inertia_tfidf.append(kmeans.inertia_)
    silhueta_scores_tfidf.append(silhouette_score(tfidf_matrix, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia_tfidf, marker='o')
plt.title('Elbow Method for TF-IDF')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_tfidf.png')

plt.figure(figsize=(10, 5))
plt.plot(K_range, silhueta_scores_tfidf, marker='o')
plt.title('Silhouette Score for TF-IDF')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('silhouette_tfidf.png')

print("gráficos salvos")

# b) 
n_topics = 20  
nmf = NMF(n_components=n_topics, random_state=42, max_iter=500, init='nndsvda')
nmf_topic_matrix = nmf.fit_transform(tfidf_matrix)

inertia_nmf = []
silhueta_scores_nmf = []

print("calculando kmeans para NMF")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(nmf_topic_matrix)
    inertia_nmf.append(kmeans.inertia_)
    silhueta_scores_nmf.append(silhouette_score(nmf_topic_matrix, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia_nmf, marker='o')
plt.title('Elbow Method for NMF Topic Distribution')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_nmf.png')

plt.figure(figsize=(10, 5))
plt.plot(K_range, silhueta_scores_nmf, marker='o')
plt.title('Silhouette Score for NMF Topic Distribution')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('silhouette_nmf.png')

print("graficos salvos")