import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import networkx as nx

# --- Configurações Iniciais ---
# Carregar o modelo pequeno em inglês da spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Modelo 'en_core_web_sm' não encontrado. Por favor, execute:")
    print("python -m spacy download en_core_web_sm")
    exit()

# Carregar o dataset
try:
    df = pd.read_csv("../dataset.csv")
except FileNotFoundError:
    print("Arquivo 'dataset.csv' não encontrado. Faça o download de:")
    print("https://www.kaggle.com/datasets/sainitishmitta04/23k-reddit-gaming-comments-with-sentiments-dataset")
    exit()

# Para verificar as colunas disponíveis, você pode descomentar a linha abaixo:
# print("Colunas disponíveis no dataset:", df.columns)

# *** CORREÇÃO APLICADA AQUI ***
# A coluna com os textos chama-se 'Comment', e não 'body'.
TEXT_COLUMN = 'Comment'

# Limpeza e amostragem dos dados
df.dropna(subset=[TEXT_COLUMN], inplace=True)
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
sample_df = df.sample(n=2000, random_state=42)
texts = sample_df[TEXT_COLUMN].tolist()


# ==============================================================================
# a) Extraia as etiquetas gramaticais (POS) de cada token do seu textos.
# ==============================================================================

print("Iniciando a Tarefa a) e b): Extração e Contagem de Etiquetas POS...")

all_pos_tags = []
docs = list(nlp.pipe(texts))

for doc in docs:
    for token in doc:
        all_pos_tags.append(token.pos_)

print(f"Tarefa a) concluída. Total de {len(all_pos_tags)} tokens etiquetados.")


# ==============================================================================
# b) Calcule e plote um gráfico com as frequências de cada tipo gramatical.
# ==============================================================================

pos_counts = Counter(all_pos_tags)
most_common_pos = pos_counts.most_common(15)
pos_labels, pos_frequencies = zip(*most_common_pos)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
sns.barplot(x=list(pos_labels), y=list(pos_frequencies), palette="viridis")
plt.title('Frequência das 15 Principais Etiquetas Gramaticais (POS)', fontsize=16)
plt.xlabel('Etiqueta Gramatical (POS)', fontsize=12)
plt.ylabel('Frequência Absoluta', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
print("Tarefa b) concluída. Exibindo o gráfico de frequências POS...")
plt.show()


# ==============================================================================
# c) Reconhecimento de Entidades Nomeadas (NER).
# ==============================================================================

print("\nIniciando a Tarefa c): Reconhecimento de Entidades Nomeadas (ORG)...")

entities_per_doc = []
for doc in docs:
    current_doc_entities = set(
        ent.text.strip().lower() for ent in doc.ents if ent.label_ == 'ORG'
    )
    if current_doc_entities:
        entities_per_doc.append(list(current_doc_entities))

print(f"Tarefa c) concluída. Entidades 'ORG' encontradas em {len(entities_per_doc)} documentos.")
print("Exemplos de documentos com entidades:", entities_per_doc[:5])


# ==============================================================================
# d) Gere um grafo com pesos onde os nós representam cada entidade reconhecida.
# ==============================================================================

print("\nIniciando a Tarefa d): Geração e Plotagem do Grafo de Coocorrência...")

edge_weights = Counter()

for entities in entities_per_doc:
    for pair in combinations(sorted(entities), 2):
        edge_weights[pair] += 1

G = nx.Graph()

for (entity1, entity2), weight in edge_weights.items():
    G.add_edge(entity1, entity2, weight=weight)

if G.number_of_edges() > 0:
    G_filtered = nx.Graph()
    for u, v, data in G.edges(data=True):
        if data['weight'] > 1:
            G_filtered.add_edge(u, v, weight=data['weight'])

    if G_filtered.number_of_edges() > 0:
        G_to_plot = G_filtered
        print(f"O grafo foi filtrado. Mostrando as {G_to_plot.number_of_edges()} conexões mais fortes (peso > 1).")
    else:
        G_to_plot = G
        print("Nenhuma conexão com peso > 1 encontrada. Mostrando o grafo completo.")
else:
    G_to_plot = G
    print("Nenhuma coocorrência encontrada para gerar o grafo.")

if G_to_plot.number_of_nodes() > 0:
    plt.figure(figsize=(16, 16))
    pos = nx.kamada_kawai_layout(G_to_plot)
    weights = [G_to_plot[u][v]['weight'] * 1.5 for u, v in G_to_plot.edges()]

    nx.draw_networkx_nodes(G_to_plot, pos, node_size=1500, node_color='skyblue', alpha=0.9)
    nx.draw_networkx_edges(G_to_plot, pos, width=weights, edge_color='gray', alpha=0.7)
    nx.draw_networkx_labels(G_to_plot, pos, font_size=10, font_family='sans-serif')

    plt.title("Grafo de Coocorrência de Organizações (Empresas de Games)", fontsize=20)
    plt.axis('off')
    print("Tarefa d) concluída. Exibindo o grafo...")
    plt.show()
else:
    print("Não há nós ou arestas suficientes para desenhar o grafo.")