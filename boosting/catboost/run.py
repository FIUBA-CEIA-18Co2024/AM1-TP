from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Full imports section should look like:
import nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Added accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.sparse
from catboost import CatBoostClassifier
import optuna
import warnings
import GPUtil
from threading import Thread
import uuid
import json
import os
from datetime import datetime


warnings.filterwarnings('ignore')

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

def monitor_gpu():
    """Monitor GPU usage and memory"""
    while True:
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            print(f'\rGPU {gpu.id} - Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB, Load: {gpu.load*100}%', end='')
        time.sleep(1)

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

def plot_accuracy_vs_params(study, save_path):
    """Plot accuracy vs different hyperparameters"""
    # Get trials data
    trials_df = pd.DataFrame([
        {**t.params, 'accuracy': t.value}
        for t in study.trials if t.value is not None
    ])

    # Select numerical parameters
    num_params = trials_df.select_dtypes(include=[np.number]).columns
    num_params = [col for col in num_params if col != 'accuracy']

    # Create subplots for each parameter
    n_params = len(num_params)
    n_cols = 2
    n_rows = (n_params + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()

    for i, param in enumerate(num_params):
        ax = axes[i]
        # Sort by parameter value
        sorted_data = trials_df.sort_values(param)
        # Plot line and points
        ax.plot(sorted_data[param], sorted_data['accuracy'], 'b-', alpha=0.3)  # Line
        ax.scatter(sorted_data[param], sorted_data['accuracy'], alpha=0.5)      # Points
        ax.set_xlabel(param)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Accuracy vs {param}')
        ax.grid(True, linestyle='--', alpha=0.7)

    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(vectorizer, classifier, save_path):
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

        # Sort by importance and get top 20 features
        top_features = feature_importance.nlargest(20, 'importance')

        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')

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

def plot_accuracy_history(study, save_path):
    """Plot accuracy per iteration"""
    # Get trials data
    trials_df = pd.DataFrame([
        {'number': t.number, 'accuracy': t.value}
        for t in study.trials if t.value is not None
    ])

    # Plot accuracy history
    plt.figure(figsize=(10, 6))
    plt.plot(trials_df['number'], trials_df['accuracy'], 'b-', label='Trial accuracy')
    plt.plot(trials_df['number'], trials_df['accuracy'].cummax(), 'r-', label='Best accuracy')
    plt.xlabel('Trial number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_run_directory():
    """Create a unique directory for this run"""
    run_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"runs/run_{timestamp}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_run_results(study, final_model, y_train, y_train_pred, y_test, y_test_pred, run_dir):
    """Save all results and plots for a specific run"""
    # Create subdirectories
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plots with correct paths
    plot_confusion_matrices(
        y_train, y_train_pred, y_test, y_test_pred,
        os.path.join(plots_dir, 'confusion_matrices.png')
    )

    plot_hyperparameter_importance(
        study,
        os.path.join(plots_dir, 'hyperparameter_importance.png')
    )

    plot_accuracy_vs_params(
        study,
        os.path.join(plots_dir, 'accuracy_vs_params.png')
    )

    plot_accuracy_history(
        study,
        os.path.join(plots_dir, 'accuracy_history.png')
    )

    # Get vectorizer and classifier from final model
    vectorizer = final_model.named_steps['vectorizer']
    classifier = final_model.named_steps['classifier']

    # Plot feature importance
    plot_feature_importance(
        vectorizer, classifier,
        os.path.join(plots_dir, 'feature_importance.png')
    )

    # Save study results
    study_results = {
        'run_id': os.path.basename(run_dir),
        'timestamp': datetime.now().isoformat(),
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_report': classification_report(y_train, y_train_pred, output_dict=True),
        'test_report': classification_report(y_test, y_test_pred, output_dict=True),
        'trial_history': [
            {
                'trial_number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials if t.value is not None
        ]
    }

    # Save results to JSON
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(study_results, f, indent=4)

    # Save run summary
    summary = f"""
        Run Summary
        ==========
        Run ID: {study_results['run_id']}
        Timestamp: {study_results['timestamp']}
        Best Accuracy: {study_results['best_value']:.4f}
        Train Accuracy: {study_results['train_accuracy']:.4f}
        Test Accuracy: {study_results['test_accuracy']:.4f}
        Number of Trials: {study_results['n_trials']}

        Best Parameters:
        {json.dumps(study_results['best_params'], indent=2)}
            """

    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write(summary)

    return study_results

def train_with_optuna(X_train, X_test, y_train, y_test, n_trials=15):  # More trials
    """Train model with focus on quality"""
    run_dir = create_run_directory()
    print(f"\nStarting new run: {os.path.basename(run_dir)}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,  # More random trials for exploration
            seed=42
        )
    )

    try:
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
            show_progress_bar=True
        )

        print("\nBest trial:")
        trial = study.best_trial
        print(f"Value: {trial.value:.3f}")
        print("\nBest parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Train final model with best parameters
        final_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=trial.params['max_features'],
                ngram_range=trial.params['ngram_range']
            )),
            ('classifier', create_catboost_classifier(params=trial.params))
        ])

        print("\nTraining final model with best parameters...")
        final_pipeline.fit(X_train, y_train)

        # Generate predictions
        y_train_pred = final_pipeline.predict(X_train)
        y_test_pred = final_pipeline.predict(X_test)

        # Save all results
        results = save_run_results(
            study, final_pipeline,
            y_train, y_train_pred,
            y_test, y_test_pred,
            run_dir
        )

        print(f"\nResults saved in: {run_dir}")
        return final_pipeline, study, results, run_dir

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        # Save error information
        error_info = {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(run_dir, 'error.json'), 'w') as f:
            json.dump(error_info, f, indent=4)
        return None, study, None, run_dir

def create_catboost_classifier(params=None):
    """Create a CatBoost classifier optimized for GPU with focus on model quality"""
    try:
        # Base GPU parameters optimized for quality
        gpu_params = {
            'task_type': 'GPU',
            'devices': '0',
            'gpu_ram_part': 0.3,
            'verbose': True,
            'random_seed': 42,
            'bootstrap_type': 'Bayesian',  # Changed from Bernoulli to Bayesian
            'sampling_frequency': 'PerTree',
            'min_data_in_leaf': 20,
            'grow_policy': 'SymmetricTree',  # More stable tree growing
            'random_strength': 1,            # Add randomness to splits
            'eval_metric': 'Accuracy'
        }

        if params is not None:
            model_params = {
                'iterations': params.get('iterations', 1000),      # More iterations
                'depth': params.get('depth', 8),                   # Deeper trees
                'learning_rate': params.get('learning_rate', 0.1),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3.0),
                'border_count': params.get('border_count', 128),
                'bagging_temperature': params.get('bagging_temperature', 1.0)
            }
        else:
            model_params = {}

        all_params = {**gpu_params, **model_params}
        return CatBoostClassifier(**all_params)

    except Exception as e:
        print(f"Error creating CatBoost classifier: {str(e)}")
        return CatBoostClassifier(task_type='CPU', **model_params if params is not None else {})

def objective(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function optimized for model quality"""
    params = {
        # TF-IDF parameters to ensure variance
        'vectorizer__max_features': trial.suggest_int('max_features', 1000, 5000),
        'vectorizer__ngram_range': (1,2),
        'vectorizer__min_df': 5,                # Ignore terms that appear in less than 5 documents
        'vectorizer__max_df': 0.95,             # Ignore terms that appear in more than 95% of documents
        'vectorizer__use_idf': True,            # Enable IDF
        'vectorizer__norm': 'l2',               # Enable normalization

        # CatBoost parameters adjusted for stability
        'classifier__iterations': trial.suggest_int('iterations', 100, 1000),
        'classifier__depth': trial.suggest_int('depth', 4, 8),
        'classifier__learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3),
        'classifier__l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'classifier__min_data_in_leaf': 20,     # Ensure enough samples in leaves
        'classifier__bootstrap_type': 'Bayesian',
        'classifier__bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'classifier__random_strength': trial.suggest_float('random_strength', 1, 20)
    }

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', CatBoostClassifier(
            task_type='GPU',
            devices='0',
            gpu_ram_part=0.3,
            verbose=True,
            random_seed=42,
            early_stopping_rounds=50,
            bootstrap_type='Bayesian'  # Consistent bootstrap type
        ))
    ])

    pipeline.set_params(**params)

    try:
        # Add basic data validation
        if len(set(y_train)) < 2:
            print("Not enough classes in training data")
            return 0.0

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Verify predictions have variance
        y_pred = pipeline.predict(X_test)
        if len(set(y_pred)) < 2:
            print("No variance in predictions")
            return 0.0

        accuracy = pipeline.score(X_test, y_test)
        return accuracy
    except Exception as e:
        print(f"Trial failed: {str(e)}")
        return 0.0

def main():
    print("Starting CatBoost optimization...")

    # Load and prepare data
    dataset = pd.read_csv('../../data/cleaned_dataset_processed_balanced.csv').sample(n=20500, random_state=42)
    X = dataset['cleaned_review']
    y = dataset['rating']

    print(f"\nTotal samples: {len(dataset)}")
    print("\nClass distribution:")
    print(y.value_counts().sort_index())
    print(f"\nAverage review length: {X.str.len().mean():.1f} characters")


    X_train, X_test, y_train, y_test = split_data_stratified(X, y)

    try:
        start_time = time.time()
        final_model, study, results, run_dir = train_with_optuna(
            X_train, X_test, y_train, y_test, n_trials=25
        )
        end_time = time.time()

        if final_model is not None:
            print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
            print(f"\nRun completed successfully. Results saved in: {run_dir}")
            print("\nGenerated files:")
            for root, dirs, files in os.walk(run_dir):
                level = root.replace(run_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")
        else:
            print(f"Training failed. Error information saved in: {run_dir}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()
