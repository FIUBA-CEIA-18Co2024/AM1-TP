import os
import json
import uuid
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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

def plot_hyperparameter_evolution(study, save_path, metric):
    """Plot evolution of hyperparameters vs metric (accuracy or f1-score)"""
    trials_df = pd.DataFrame([
        {**t.params, metric: t.value if metric == 'accuracy' else t.user_attrs.get('f1')}
        for t in study.trials if t.value is not None and (metric == 'accuracy' or 'f1' in t.user_attrs)
    ])

    num_params = trials_df.select_dtypes(include=[np.number]).columns
    num_params = [col for col in num_params if col != metric]

    n_params = len(num_params)
    n_cols = 2
    n_rows = (n_params + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()

    best_value = trials_df[metric].max()

    for i, param in enumerate(num_params):
        ax = axes[i]
        sorted_data = trials_df.sort_values(param)
        ax.plot(sorted_data[param], sorted_data[metric], 'b-', alpha=0.3)
        ax.scatter(sorted_data[param], sorted_data[metric], alpha=0.5)
        ax.axhline(y=best_value, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel(param)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} vs {param}')
        ax.grid(True, linestyle='--', alpha=0.7)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(vectorizer, classifier, save_path, top_n=20):
    """Plot top features based on their importance"""
    try:
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Get feature importance scores
        importance_scores = classifier.get_feature_importance()

        # Create DataFrame with features and their importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })

        # Sort by importance and get top N features
        top_features = feature_importance.nlargest(top_n, 'importance')

        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                     f'{width:.4f}',
                     ha='left', va='center', fontweight='bold')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")

def plot_hyperparameter_importance(study, save_path):
    """Plot hyperparameter importance vs accuracy"""
    importance = optuna.importance.get_param_importances(study)

    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame(
        importance.items(),
        columns=['Parameter', 'Importance']
    ).sort_values('Importance', ascending=True)

    plt.barh(range(len(importance_df)), importance_df['Importance'])
    plt.yticks(range(len(importance_df)), importance_df['Parameter'])
    plt.xlabel('Importance')
    plt.title('Hyperparameter Importance')
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

def objective(trial, X_train, X_test, y_train, y_test, threads):
    params = {
        'vectorizer__max_features': trial.suggest_int('vectorizer__max_features', 1000, 5500),
        'vectorizer__ngram_range': (1, 2),
        'classifier__iterations': trial.suggest_int('classifier__iterations', 100, 1200),
        'classifier__depth': trial.suggest_int('classifier__depth', 4, 8),
        'classifier__learning_rate': trial.suggest_float('classifier__learning_rate', 0.01, 0.7),
        'classifier__l2_leaf_reg': trial.suggest_float('classifier__l2_leaf_reg', 1.0, 10.0),
        'classifier__bagging_temperature': trial.suggest_float('classifier__bagging_temperature', 0.0, 1.0),
        'classifier__thread_count': threads  # Limit to 4 CPU cores
    }

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=params['vectorizer__max_features'], ngram_range=params['vectorizer__ngram_range'])),
        ('classifier', CatBoostClassifier(
            iterations=params['classifier__iterations'],
            depth=params['classifier__depth'],
            learning_rate=params['classifier__learning_rate'],
            l2_leaf_reg=params['classifier__l2_leaf_reg'],
            bootstrap_type='Bayesian',  # Specify Bayesian bootstrap type
            bagging_temperature=params['classifier__bagging_temperature'],
            thread_count=threads,
            task_type='GPU' if use_gpu else 'CPU',
            devices='0' if use_gpu else None,
            verbose=False
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    trial.set_user_attr('f1', f1)
    return accuracy

def train(X_train, X_test, y_train, y_test, trials, run_dir, threads):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test, threads), n_trials=trials, show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best parameters to JSON file
    with open(os.path.join(run_dir, 'best_params.json'), 'w') as f:
        json.dump(trial.params, f, indent=4)

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

    final_pipeline.fit(X_train, y_train)
    
    # Save the trained model
    model_path = os.path.join(run_dir, 'best_model.cbm')
    final_pipeline.named_steps['classifier'].save_model(model_path)

    y_train_pred = final_pipeline.predict(X_train)
    y_test_pred = final_pipeline.predict(X_test)

    # Plot and save confusion matrices
    plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, os.path.join(run_dir, 'confusion_matrices.png'))

    # Plot and save hyperparameter evolution vs accuracy
    plot_hyperparameter_evolution(study, os.path.join(run_dir, 'hyperparameter_evolution_accuracy.png'), metric='accuracy')

    # Plot and save hyperparameter evolution vs f1-score
    plot_hyperparameter_evolution(study, os.path.join(run_dir, 'hyperparameter_evolution_f1.png'), metric='f1')

    # Plot and save feature importance
    plot_feature_importance(final_pipeline.named_steps['vectorizer'], final_pipeline.named_steps['classifier'], os.path.join(run_dir, 'feature_importance.png'))

    # Plot and save hyperparameter importance
    plot_hyperparameter_importance(study, os.path.join(run_dir, 'hyperparameter_importance.png'))

if __name__ == '__main__':
    print("Starting CatBoost optimization...")
    use_gpu = True  # Set to True to use GPU
    hyperparameters_trials = 50 # 300
    threads = 10  # Number of CPU threads for CatBoost

    # Create a unique directory for this run
    run_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"runs/run_{timestamp}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Load and prepare data
    dataset = pd.read_csv('../../data/cleaned_dataset_processed_balanced.csv').sample(n=2500, random_state=42)
    X = dataset['cleaned_review']
    y = dataset['rating']
    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    
    # Train and optimize model hyperparameters
    train(X_train, X_test, y_train, y_test, trials=hyperparameters_trials, run_dir=run_dir, threads=threads)