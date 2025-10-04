from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-distilroberta-v1')

embeddings = model.encode(df['text_pp'].tolist(), show_progress_bar=True, batch_size=64)

X_train, X_test, y_train, y_test = train_test_split(embeddings, y_enc, test_size=0.30, random_state=42, stratify=y_enc)

classifiers_bert = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'GaussianNB': GaussianNB(),
    'LinearSVC': LinearSVC(max_iter=5000)
}

for cname, clf in classifiers_bert.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    print("=== DistilBERT +", cname, "===")
    print("accuracy:",acc,"f1_macro:",f1_macro)
    print(classification_report(y_test, preds, target_names=le.classes_))
    print("Confusion:\n", confusion_matrix(y_test, preds))
