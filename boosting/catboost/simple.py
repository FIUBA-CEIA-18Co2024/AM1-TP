import os
import uuid
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, save_path):
    """Plot confusion matrices for train and test sets side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Training confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_train_percent = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_train_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
    ax1.set_title('Training Confusion Matrix (%)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Testing confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_test_percent = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_test_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Testing Confusion Matrix (%)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def split_data_stratified(X, y, test_size=0.2, random_state=42):
    """Split data ensuring proportional representation of all classes"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_idx, test_idx in sss.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

    print("Class distribution in splits:")
    print("\nTraining set:")
    print(y_train.value_counts().sort_index())
    print("\nTest set:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test

def objective(trial, X_train, X_test, y_train, y_test, batch_size, threads):
    params = {
        'vectorizer__max_features': trial.suggest_int('vectorizer__max_features', 1000, 5000),
        'vectorizer__ngram_range': (1, 2),
        'classifier__iterations': trial.suggest_int('classifier__iterations', 100, 1000),
        'classifier__depth': trial.suggest_int('classifier__depth', 4, 8),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.3),
        'classifier__l2_leaf_reg': trial.suggest_float('classifier__l2_leaf_reg', 1.0, 10.0),
        'classifier__thread_count': threads  # Limit to 4 CPU cores
    }

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=params['vectorizer__max_features'], ngram_range=params['vectorizer__ngram_range'])),
        ('classifier', CatBoostClassifier(
            iterations=params['classifier__iterations'],
            depth=params['classifier__depth'],
            learning_rate=params['classifier__learning_rate'],
            l2_leaf_reg=params['classifier__l2_leaf_reg'],
            thread_count=params['classifier__thread_count'],
            task_type='GPU' if use_gpu else 'CPU',
            devices='0' if use_gpu else None,
            verbose=False
        ))
    ])

    # Implement batching
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        pipeline.fit(X_train[start:end], y_train[start:end])

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train(X_train, X_test, y_train, y_test, trials, run_dir, batch_size=32, threads=4):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test, batch_size, threads), n_trials=trials, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train final model with best parameters
    final_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=trial.params['vectorizer__max_features'],
            ngram_range=(1, 2)
        )),
        ('classifier', CatBoostClassifier(
            iterations=trial.params['classifier__iterations'],
            depth=trial.params['classifier__depth'],
            learning_rate=trial.params['classifier__learning_rate'],
            l2_leaf_reg=trial.params['classifier__l2_leaf_reg'],
            thread_count=threads,
            task_type='GPU' if use_gpu else 'CPU',
            devices='0' if use_gpu else None,
            verbose=False
        ))
    ])

    # Implement batching
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        final_pipeline.fit(X_train[start:end], y_train[start:end])

    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    # Plot and save confusion matrices
    plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, os.path.join(run_dir, 'confusion_matrices.png'))
   
    
if __name__ == '__main__':
    print("Starting CatBoost optimization...")
    use_gpu = False  # Set to True to use GPU
    hyperparameters_trials = 3
    batch_size = 32
    threads = 4

    # Create a unique directory for this run
    run_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"runs/run_{timestamp}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Load and prepare data
    dataset = pd.read_csv('../../data/cleaned_dataset_processed_balanced.csv').sample(n=1000, random_state=42)
    X = dataset['cleaned_review']
    y = dataset['rating']
    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    
    # Train and optimize model hyperparameters
    train(X_train, X_test, y_train, y_test, 
          trials=hyperparameters_trials, 
          run_dir=run_dir, 
          batch_size=batch_size,
          threads=threads)