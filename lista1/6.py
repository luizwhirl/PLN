import pandas as pd
import re
from collections import Counter
import nltk

df = pd.read_csv('dataset.csv')
df.dropna(subset=['Comment'], inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\w*\d\w*', '', text) # mata palavras com números
    text = re.sub(r'[^\w\s]', '', text)   # mata pontuação
    return text

# pré-processamento inicial
df['processed_comment'] = df['Comment'].apply(preprocess_text)

df['tokens'] = df['processed_comment'].apply(lambda x: x.split())


stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
df['tokens_no_stopwords'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
df['stemmed_tokens'] = df['tokens_no_stopwords'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

posTags = False
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk import pos_tag
    # aplicando a rotulação em uma amostra por problemas de performance
    amostra_df = df.head(5000)
    amostra_df['pos_tags'] = amostra_df['tokens_no_stopwords'].apply(pos_tag)
    df = df.merge(amostra_df[['pos_tags']], left_index=True, right_index=True, how='left')
    posTags = True
except Exception as e:
    print("AVISO: Não foi possível realizar a rotulação de POS (Part-of-Speech).", e)
    df['pos_tags'] = [[] for _ in range(len(df))]


print("a)")
# selecionando colunas para exibição, tratando o caso do POS
exibir = ['processed_comment', 'tokens_no_stopwords', 'stemmed_tokens']
if posTags:
    exibir.append('pos_tags')
print(df[exibir].head())
print("\n" + "="*50 + "\n")

print("b)")
print("\antes da remoção de stopwords:")
todosTk_antes = [token for listaTk in df['tokens'] for token in listaTk]
contagem_antes = Counter(todosTk_antes)
top10_antes = contagem_antes.most_common(10)
for word, frequencia in top10_antes:
    print(f"{word}: {frequencia}")

print("\ndepois da remoção de stopwords:")
todosTk_semStpWord = [token for listaTks in df['tokens_no_stopwords'] for token in listaTks]
contagem_semStpWord = Counter(todosTk_semStpWord)
top10_semStpWord = contagem_semStpWord.most_common(10)
for word, frequencia in top10_semStpWord:
    print(f"{word}: {frequencia}")
print("\n" + "="*50 + "\n")

print("c)")
allStmTok = [token for listaTks in df['stemmed_tokens'] for token in listaTks]
stemmedCount = Counter(allStmTok)
top10stm = stemmedCount.most_common(10)
for word, frequencia in top10stm:
    print(f"{word}: {frequencia}")
print("\n" + "="*50 + "\n")

print("d)")
if posTags:
    todasTagspos = [tag for tags_list in df['pos_tags'].dropna() for _, tag in tags_list]
    pos_counts = Counter(todasTagspos)
    top10novo = pos_counts.most_common(10)
    for tag, frequencia in top10novo:
        print(f"{tag}: {frequencia} ({nltk.help.upenn_tagset(tag)[0][1]})")
else:
    print("erro no download dos recursos💔")