import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import umap 


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

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english', use_idf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_sample['processed_comment'])

n_topics = 20
nmf = NMF(n_components=n_topics, random_state=42, max_iter=500, init='nndsvda')
nmf_topic_matrix = nmf.fit_transform(tfidf_matrix)

K_TFIDF = 3 
K_NMF = 4   

print(f"executando k-means com k={K_TFIDF} para tf-idf")
kmeans_tfidf = KMeans(n_clusters=K_TFIDF, random_state=42, n_init=10)
labels_tfidf = kmeans_tfidf.fit_predict(tfidf_matrix)

print(f"executando k-means com k={K_NMF} para nmf")
kmeans_nmf = KMeans(n_clusters=K_NMF, random_state=42, n_init=10)
labels_nmf = kmeans_nmf.fit_predict(nmf_topic_matrix)


# questao 4
# a)
print("\n questao a)")

print("calculando projeção t-sne para if-idf")
tsne_tfidf = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
projection_tsne_tfidf = tsne_tfidf.fit_transform(tfidf_matrix.toarray()) 

print("calculando projeção UMAP para tf-idf")
umap_reducer_tfidf = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
projection_umap_tfidf = umap_reducer_tfidf.fit_transform(tfidf_matrix)

plt.figure(figsize=(16, 7))
palette = sns.color_palette("deep", K_TFIDF)

plt.subplot(1, 2, 1)
sns.scatterplot(x=projection_tsne_tfidf[:, 0], y=projection_tsne_tfidf[:, 1], hue=labels_tfidf, palette=palette, legend='full')
plt.title('projeção t-SNE da matriz TF-IDF')
plt.xlabel('componente t-SNE 1')
plt.ylabel('componente t-SNE 2')

plt.subplot(1, 2, 2)
sns.scatterplot(x=projection_umap_tfidf[:, 0], y=projection_umap_tfidf[:, 1], hue=labels_tfidf, palette=palette, legend='full')
plt.title('projeção UMAP da matriz TF-IDF')
plt.xlabel('componente UMAP 1')
plt.ylabel('componente UMAP 2')

plt.suptitle('a): visualização de clusters TF-IDF com t-SNE e UMAP (Valores Padrão)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('questao_4a_projecoes_tfidf.png')
plt.show()
print("graficos salvo")

# b)
print("\n b)")

perplexities = [5, 30, 100] 
n_neighbors_list = [5, 15, 100] 

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
print("testando diferentes valores de 'perplexity' para t-SNE")
for i, p in enumerate(perplexities):
    tsne_temp = TSNE(n_components=2, perplexity=p, random_state=42, n_iter=1000)
    projection_temp = tsne_temp.fit_transform(tfidf_matrix.toarray())
    sns.scatterplot(x=projection_temp[:, 0], y=projection_temp[:, 1], hue=labels_tfidf, palette=palette, ax=axes[i], legend=False)
    axes[i].set_title(f't-SNE com Perplexity = {p}')
fig.suptitle('b): efeito do hiperparâmetro perplexity (t-SNE)', fontsize=16)
plt.savefig('questao_4b_tsne_perplexity.png')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
print("testando diferentes valores de 'n_neighbors' para UMAP")
for i, n in enumerate(n_neighbors_list):
    umap_temp = umap.UMAP(n_components=2, n_neighbors=n, random_state=42)
    projection_temp = umap_temp.fit_transform(tfidf_matrix)
    sns.scatterplot(x=projection_temp[:, 0], y=projection_temp[:, 1], hue=labels_tfidf, palette=palette, ax=axes[i], legend=False)
    axes[i].set_title(f'UMAP com n_neighbors = {n}')
fig.suptitle('b) efeito do hiperparâmetro n_neighbors (UMAP)', fontsize=16)
plt.savefig('questao_4b_umap_n_neighbors.png')
plt.show()
print("graficos salvo")


# c
print("\nquestão c)")

BEST_PERPLEXITY = 40
BEST_N_NEIGHBORS = 25

print(f"calculando projeção t-sne para nmf com perplexity={BEST_PERPLEXITY}...")
tsne_nmf = TSNE(n_components=2, perplexity=BEST_PERPLEXITY, random_state=42, n_iter=1000)
projection_tsne_nmf = tsne_nmf.fit_transform(nmf_topic_matrix)

print(f"calculando projeção umap para nmf com n_neighbors={BEST_N_NEIGHBORS}...")
umap_reducer_nmf = umap.UMAP(n_components=2, n_neighbors=BEST_N_NEIGHBORS, random_state=42)
projection_umap_nmf = umap_reducer_nmf.fit_transform(nmf_topic_matrix)

plt.figure(figsize=(16, 7))
palette_nmf = sns.color_palette("viridis", K_NMF)

# t-SNE
plt.subplot(1, 2, 1)
sns.scatterplot(x=projection_tsne_nmf[:, 0], y=projection_tsne_nmf[:, 1], hue=labels_nmf, palette=palette_nmf, legend='full')
plt.title(f'melhor projeção t-SNE (Perplexity={BEST_PERPLEXITY})')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')

# UMAP
plt.subplot(1, 2, 2)
sns.scatterplot(x=projection_umap_nmf[:, 0], y=projection_umap_nmf[:, 1], hue=labels_nmf, palette=palette_nmf, legend='full')
plt.title(f'melhor projeção UMAP (n_neighbors={BEST_N_NEIGHBORS})')
plt.xlabel('Componente UMAP 1')
plt.ylabel('Componente UMAP 2')

plt.suptitle('visualização clusters NMF com hiperparâmetros ajustados', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('questao_4c_projecoes_nmf_otimizadas.png')
plt.show()
print("graficos salvo")
