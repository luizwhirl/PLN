import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('dataset.csv')
    print(df.columns)

    coluna_texto = 'Comment'

    df['contagem_palavras'] = df[coluna_texto].astype(str).str.split().str.len()

    print(df[[coluna_texto, 'contagem_palavras']])

    plt.figure(figsize=(10, 6))
    plt.hist(df['contagem_palavras'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribuição do Comprimento dos Comentários')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')
    plt.grid(axis='y', linestyle='--', linewidth=0.7)

    plt.savefig('histograma.png')

except FileNotFoundError:
    print("sem csv")
except KeyError:
    print(f"sem {coluna_texto}")