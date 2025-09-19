import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("preparando o ambiente e os dados..")

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    df = pd.read_csv('../dataset.csv')
except FileNotFoundError:
    print("dataset.csv não encontrado")
    exit()

if df.columns[0] == 'Unnamed: 0':
    df = df.drop(df.columns[0], axis=1)

df.dropna(subset=['Comment', 'sentiment'], inplace=True)
df['sentiment_encoded'] = df['sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})
df.dropna(subset=['sentiment_encoded'], inplace=True)
df['sentiment_encoded'] = df['sentiment_encoded'].astype(int)

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

df['tokenized_comment'] = df['processed_comment'].apply(lambda x: x.split())

X_text = df['processed_comment'] 
X_tokenized = df['tokenized_comment'] 
y = df['sentiment_encoded']
class_names = ['positive', 'negative', 'neutral']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42, stratify=y)
X_train_tok, X_test_tok = train_test_split(X_tokenized, test_size=0.3, random_state=42, stratify=y)

print(f"Dados divididos: {len(y_train)} para treino, {len(y_test)} para teste.\n")


# resultados anteriores
print(" 2. executando classificadores com CountVectorizer e TF-IDF ")

count_vectorizer = CountVectorizer(max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

classifiers_original = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": LinearSVC(max_iter=2000)
}
vectorizers = {
    "CountVectorizer": (X_train_count, X_test_count),
    "TF-IDF": (X_train_tfidf, X_test_tfidf)
}

for vec_name, (X_train_vec, X_test_vec) in vectorizers.items():
    print(f"\n{'='*20} USANDO {vec_name.upper()} {'='*20}")
    for clf_name, clf in classifiers_original.items():
        print(f"\n--- Treinando e avaliando o {clf_name} ---")
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
        print(f"Acurácia: {accuracy:.4f}")
        print("Relatório de Classificação:")
        print(report)


# a) DOC2VEC + classificadores classicos
print("\n\n  a) Doc2Vec + classificadores classicos ---")

tagged_train_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(X_train_tok)]

print("\ntreinando o Doc2Vec...")
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20, dm=1)
doc2vec_model.build_vocab(tagged_train_data)
doc2vec_model.train(tagged_train_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
print(" doc2vec treinado")

X_train_doc2vec = np.array([doc2vec_model.infer_vector(doc) for doc in X_train_tok])
X_test_doc2vec = np.array([doc2vec_model.infer_vector(doc) for doc in X_test_tok])

classifiers_doc2vec = {
    "Regressão Logística": LogisticRegression(max_iter=1000),
    "Naive Bayes (Gaussiano)": GaussianNB(),
    "SVM (Linear)": LinearSVC(max_iter=2000, dual=False) 
}

print(f"\n{'='*20} AVALIANDO CLASSIFICADORES COM DOC2VEC {'='*20}")
for clf_name, clf in classifiers_doc2vec.items():
    print(f"\n treinando e avaliando o {clf_name} ")
    clf.fit(X_train_doc2vec, y_train)
    y_pred = clf.predict(X_test_doc2vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"Acurácia: {accuracy:.4f}")
    print("Relatório de Classificação:")
    print(report)


# b) WORD2VEC + rede neural (LSTM)
print("\n\n b) Word2Vec + rede neural ")

# Treinar modelo Word2Vec com os dados de treino
print("\ntreinando  word2vec...")
w2v_model = Word2Vec(sentences=X_train_tok, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.train(X_train_tok, total_examples=len(X_train_tok), epochs=10)
print(" word2vec treinado")

VOCAB_SIZE = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train_text) 
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

word_index = tokenizer.word_index
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < VOCAB_SIZE:
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

model_lstm = Sequential([
    Embedding(input_dim=VOCAB_SIZE,
              output_dim=EMBEDDING_DIM,
              weights=[embedding_matrix],
              input_length=MAX_LEN,
              trainable=False),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nestrutura do LSTM:")
model_lstm.summary()

print("\nrreinando o LSTM...")
history = model_lstm.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

print(f"\n{'='*20} AVALIANDO O  LSTM {'='*20}")
loss, accuracy = model_lstm.evaluate(X_test_pad, y_test, verbose=0)
print(f"\nAcurácia no Teste: {accuracy:.4f}")

y_pred_probs = model_lstm.predict(X_test_pad)
y_pred_lstm = np.argmax(y_pred_probs, axis=1)
report_lstm = classification_report(y_test, y_pred_lstm, target_names=class_names, digits=4)
print("Relatório de Classificação:")
print(report_lstm)


print("\n\nexecução concluida")