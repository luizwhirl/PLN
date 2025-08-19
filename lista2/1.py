import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

df = pd.read_csv('dataset.csv')

if df.columns[0] == 'Unnamed: 0':
    df = df.drop(df.columns[0], axis=1)

print("\informações do dataframe:")
df.info()

print("\primerias 5 linhas:")
print(df.head())

print("\nvalores nulos por coluna: ")
print(df.isnull().sum())

df.dropna(subset=['Comment', 'sentiment'], inplace=True)

print("\ndistribuição das classes de sentimento:")
print(df['sentiment'].value_counts())

# a)
df['sentiment_encoded'] = df['sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})
if df['sentiment_encoded'].isnull().any():
    print("\nerro: valroes na coluna fora do mapeamento")
    df.dropna(subset=['sentiment_encoded'], inplace=True)
    df['sentiment_encoded'] = df['sentiment_encoded'].astype(int)

print("\nclasses definidas e codificadas:")
print(df[['sentiment', 'sentiment_encoded']].value_counts())


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower() 
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\n', '', text) 
    text = ' '.join(word for word in text.split() if word not in stop_words) 
    return text

df['processed_comment'] = df['Comment'].apply(preprocess_text)

print("\ncomentario exemplo:")
print("Original:", df['Comment'].iloc[0])
print("Processado:", df['processed_comment'].iloc[0])

X = df['processed_comment']
y = df['sentiment_encoded']
class_names = ['positive', 'negative', 'neutral']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\n {len(X_train)} para treino, {len(X_test)} para teste.")

# b)
print("\nextraindo")
count_vectorizer = CountVectorizer(max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)
print("shape da matriz do CountVectorizer:", X_train_count.shape)

print("\nextraindo (TfidfVectorizer)...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Shape da matriz TfidfVectorizer:", X_train_tfidf.shape)

# c)

classifiers = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": LinearSVC(max_iter=2000)
}

vectorizers = {
    "CountVectorizer": (X_train_count, X_test_count),
    "TF-IDF": (X_train_tfidf, X_test_tfidf)
}

def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

for vec_name, (X_train_vec, X_test_vec) in vectorizers.items():
    print(f"\n{'='*20} USANDO {vec_name.upper()} {'='*20}")
    for clf_name, clf in classifiers.items():
        print(f"\ntreinando e avaliando o  {clf_name} ---")

        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        print(f"acurácia: {accuracy:.4f}")
        print("eelatório de classificação:")
        print(report)
        print("matriz de confusão:")
        print(cm)

        fig_filename = f"lista2/cm_{vec_name.lower()}_{clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_confusion_matrix(cm, class_names, f'matriz de Confusão - {clf_name} com {vec_name}', fig_filename)
