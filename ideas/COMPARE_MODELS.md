```python
def compare_models(X_train, X_test, y_train, y_test):
    # Define all pipelines
    pipelines = {
        'Logistic Regression': Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', LogisticRegression(max_iter=1000))
        ]),
        'SVM': Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', OneVsRestClassifier(LinearSVC(random_state=42)))
        ]),
        'Random Forest': Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'XGBoost': Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', XGBClassifier(random_state=42))
        ]),
        'Naive Bayes': Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
    }

    # Results storage
    results = {}

    # Train and evaluate each model
    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Store results
        results[name] = {
            'train_score': pipeline.score(X_train, y_train),
            'test_score': pipeline.score(X_test, y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"Training accuracy: {results[name]['train_score']:.3f}")
        print(f"Testing accuracy: {results[name]['test_score']:.3f}")
        print("\nClassification Report:")
        print(results[name]['classification_report'])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = results[name]['confusion_matrix']
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm_percentage,
                   annot=True,
                   fmt='.1f',
                   cmap='Blues')
        plt.title(f'{name} - Confusion Matrix (%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    return results

# Compare all models
results = compare_models(X_train, X_test, y_train, y_test)

# Plot comparative performance
def plot_model_comparison(results):
    models = list(results.keys())
    train_scores = [results[m]['train_score'] for m in models]
    test_scores = [results[m]['test_score'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_scores, width, label='Train')
    ax.bar(x + width/2, test_scores, width, label='Test')

    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_model_comparison(results)
```
