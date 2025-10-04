# count/TF-IDF + classificadores

X = df['text_pp'].values
y = df['label'].values
le = LabelEncoder(); y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.30, random_state=42, stratify=y_enc)

vectorizers = {
    'count': CountVectorizer(max_features=5000, ngram_range=(1,2)),
    'tfidf': TfidfVectorizer(max_features=5000, ngram_range=(1,2))
}
classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1),
    'MultinomialNB': MultinomialNB(),
    'LinearSVC': LinearSVC(max_iter=5000)
}

results = {}
for vname, vec in vectorizers.items():
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    for cname, clf in classifiers.items():
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        acc = accuracy_score(y_test, preds)
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
        results[f"{vname}+{cname}"] = {
            'accuracy': acc, 'f1_macro': f1_macro,
            'report': classification_report(y_test, preds, target_names=le.classes_),
            'cm': confusion_matrix(y_test, preds)
        }

for k,v in results.items():
    print("===",k,"===")
    print("accuracy:",v['accuracy'],"f1_macro:",v['f1_macro'])
    print(v['report'])
    print("Confusion matrix:\n",v['cm'])
