import pandas as pd
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: 
    print("baixando pacotes nltk")
    nltk.download('punkt')
    nltk.download('punkt_tab')

csv_file = 'da.csv'

# a) (via Downloader)
print("--- etapa a) ---")

try:
    df = pd.read_csv(csv_file)
    print(f"dataset carregado com sucesso. Colunas: {df.columns.tolist()}")
    print(f"total de documentos (linhas): {len(df)}")

    model_name = 'glove-wiki-gigaword-300'
    print(f"\n Carregando o modelo '{model_name}'....")
    model = api.load(model_name)
    print("modelo Word Embedding carregado com sucesso")

except FileNotFoundError:
    print(f"ERRO: o arquivo  do dataset '{csv_file}' não encontrado.")
    exit()
except Exception as e:
    print(f"erro ao carregar o {e}")
    exit()

print("\ncriando vocabulário do corpus a partir dos textos")
df['processed_comment'] = df['processed_comment'].fillna('') 
corpus_text = ' '.join(df['processed_comment'])
corpus_vocabulary = set(word_tokenize(corpus_text))
print(f"vocabulário criado com {len(corpus_vocabulary)} palavras únicas")


print("\n--- etapa b)---")

query_words_b = ['happiness', 'investment', 'planet', 'algorithm', 'recipe']

for query_word in query_words_b:
    print(f"\n> palavra de consulta: '{query_word}'")
    
    try:
        similar_words_in_model = model.most_similar(query_word, topn=50)
        similar_in_corpus = [word for word, score in similar_words_in_model if word in corpus_vocabulary]
        top_3_similar = similar_in_corpus[:3]
        
        if not top_3_similar:
            print(f"  nenhuma palavra parecida com '{query_word}' foi encontrada no seu conjunto de textos")
            continue

        print(f"  as 3 palavras mais parecidas nos seus textos são:")
        for similar_word in top_3_similar:
            score = model.similarity(query_word, similar_word)
            print(f"    - '{similar_word}' (similaridade: {score:.4f})")
            
            docs_with_word = df[df['processed_comment'].str.contains(r'\b{}\b'.format(similar_word), regex=True)]
            indices = docs_with_word.index.tolist()
            print(f"      > Aparece nos documentos (índices): {indices[:5]}")

    except KeyError:
        print(f"  - a palavra '{query_word}' não existe no vocabulário do modelo")


print("\n--- etapac) ---")

def find_top_documents(query_word, dataframe, w2v_model, k_similar=5, n_docs=3):
    doc_distances = []

    if query_word not in w2v_model:
        print(f" a palavra de consulta '{query_word}' não está no modelo")
        return []

    for index, row in dataframe.iterrows():
        doc_words = set(word_tokenize(row['processed_comment']))
        word_similarities = []
        
        for word in doc_words:
            if word in w2v_model:
                similarity = w2v_model.similarity(query_word, word)
                word_similarities.append((word, similarity))
        
        if not word_similarities:
            continue
            
        word_similarities.sort(key=lambda item: item[1], reverse=True)
        l_k = word_similarities[:k_similar]
        
        if not l_k:
            avg_distance = float('inf')
        else:
            distances = [1 - sim for word, sim in l_k]
            avg_distance = np.mean(distances)
        
        doc_distances.append({'index': index, 'distance': avg_distance, 'comment': row['Comment']})
        
    sorted_docs = sorted(doc_distances, key=lambda item: item['distance'])
    return sorted_docs[:n_docs]



print("\n--- etapa d):  ---")

query_words_d = ['negative', 'company', 'client', 'product', 'service']

for query_word in query_words_d:
    print(f"\n--- buscando documentos para a palavra  '{query_word}' ---")
    
    top_3_docs = find_top_documents(query_word, df, model, k_similar=5, n_docs=3)
    
    if top_3_docs:
        print(f"os 3 documentos mais próximos de '{query_word}' são:")
        for i, doc in enumerate(top_3_docs):
            print(f"  {i+1}. documento (Índice: {doc['index']}) - Distância Média: {doc['distance']:.4f}")
            print(f"     comentário Original: \"{doc['comment'][:150]}...\"") 
    else:
        print(f"não foi possível encontrar documentos para a consulta '{query_word}'")

print("\nfim do program,a")