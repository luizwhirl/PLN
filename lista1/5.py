import pandas as pd
import re
from collections import Counter

df = pd.read_csv('dataset.csv')
df.dropna(subset=['Comment'], inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\w*\d\w*', '', text) # mata palavras com números

    text = re.sub(r'[^\w\s]', '', text)    # mata pontuação
    return text

#  pre-processamento
df['processed_comment'] = df['Comment'].apply(preprocess_text)

# fica comentário original, comentário processado e o sentimento
processed_df = df[['Comment', 'processed_comment', 'sentiment']]
processed_df.to_csv('lista1/processed_dataset.csv', index=False)

df['tokens'] = df['processed_comment'].apply(lambda x: x.split())
todosTk = [token for listaTk in df['tokens'] for token in listaTk]
contagem = Counter(todosTk)
top10 = contagem.most_common(10)

for word, frequency in top10:
    print(f"{word}: {frequency}")