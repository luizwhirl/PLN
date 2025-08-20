# --- Imports e Configurações Iniciais ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Garante que as stopwords estão disponíveis
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Carregamento e Pré-processamento (Reutilizando do script anterior) ---
# Assumindo que 'df' e 'processed_comment' já foram criados.
# Se estiver rodando este script de forma isolada, descomente as linhas abaixo.
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


# --- Geração das Representações Vetoriais (Reutilizando do script anterior) ---
# 1. Representação TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000)
tfidf = vectorizer.fit_transform(df['processed_comment'])

# 2. Representação de Tópicos NMF
num_topics = 10
nmf = NMF(n_components=num_topics, random_state=42)
# A matriz de distribuição de tópicos por documento
nmf_matrix = nmf.fit_transform(tfidf)


# --- Função para encontrar o K ideal ---
def find_optimal_k(data, max_k=15):
    """
    Testa valores de k de 2 a max_k usando o Método do Cotovelo e a Pontuação de Silhueta.
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plotando o Método do Cotovelo
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo')

    # Plotando a Pontuação de Silhueta
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'go-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Pontuação de Silhueta')
    plt.title('Pontuação de Silhueta Média')
    
    plt.tight_layout()
    plt.show()

# --- Função para mostrar exemplos dos clusters ---
def show_cluster_examples(df, cluster_column_name, n_examples=2):
    """
    Mostra alguns comentários de exemplo para cada cluster.
    """
    print(f"\nExemplos de comentários por cluster ({cluster_column_name}):")
    for cluster_id in sorted(df[cluster_column_name].unique()):
        print(f"\n--- Cluster {cluster_id} ---")
        sample_comments = df[df[cluster_column_name] == cluster_id]['Comment'].sample(n_examples, random_state=42).tolist()
        for comment in sample_comments:
            print(f"  - {comment[:150]}...")
    print("\n" + "="*50)


# ==============================================================================
# 3. a) Agrupamento com Representação TF-IDF
# ==============================================================================
print("Analisando o K ideal para a representação TF-IDF...")
find_optimal_k(tfidf, max_k=15)

# --- Análise e Execução Final ---
# Com base nos gráficos, escolha um valor de `k`.
# O "cotovelo" pode ser sutil. A pontuação de silhueta pode ter um pico claro.
# Vamos supor que k=8 seja uma boa escolha com base nos gráficos.
K_TFIDF = 8
print(f"\nExecutando K-Means com k={K_TFIDF} para TF-IDF...")
kmeans_tfidf = KMeans(n_clusters=K_TFIDF, n_init='auto', random_state=42)
df['tfidf_cluster'] = kmeans_tfidf.fit_predict(tfidf)

print("\nDistribuição dos documentos nos clusters (TF-IDF):")
print(df['tfidf_cluster'].value_counts().sort_index())

# Mostrar exemplos
show_cluster_examples(df, 'tfidf_cluster')


# ==============================================================================
# 3. b) Agrupamento com Distribuição de Tópicos (NMF)
# ==============================================================================
print("\nAnalisando o K ideal para a representação de Tópicos NMF...")
# A matriz nmf_matrix tem dimensionalidade muito menor (num_topics),
# tornando o agrupamento mais rápido e, muitas vezes, mais robusto.
find_optimal_k(nmf_matrix, max_k=10) # max_k não precisa ser maior que num_topics

# --- Análise e Execução Final ---
# Os gráficos para a matriz NMF são geralmente mais claros.
# Vamos supor que o pico da silhueta ou o cotovelo indiquem k=4.
K_NMF = 4
print(f"\nExecutando K-Means com k={K_NMF} para NMF...")
kmeans_nmf = KMeans(n_clusters=K_NMF, n_init='auto', random_state=42)
df['nmf_cluster'] = kmeans_nmf.fit_predict(nmf_matrix)

print("\nDistribuição dos documentos nos clusters (NMF):")
print(df['nmf_cluster'].value_counts().sort_index())

# Mostrar exemplos
show_cluster_examples(df, 'nmf_cluster')