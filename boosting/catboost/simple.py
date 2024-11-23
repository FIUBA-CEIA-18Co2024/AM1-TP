import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit



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

def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        'vectorizer__max_features': trial.suggest_int('max_features', 1000, 5000),
        'vectorizer__ngram_range': (1, 2),
        'classifier__iterations': trial.suggest_int('iterations', 100, 1000),
        'classifier__depth': trial.suggest_int('depth', 4, 8),
        'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'classifier__l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'classifier__thread_count': 6  # Limit to 4 CPU cores
    }

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', CatBoostClassifier(
            task_type='GPU' if use_gpu else 'CPU',
            devices='0' if use_gpu else None,
            verbose=False
        ))
    ])

    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def train(X_train, X_test, y_train, y_test, trials):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(
        trial, X_train, X_test, y_train, y_test), 
        n_trials=trials,
        show_progress_bar=True
        )

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    
if __name__ == '__main__':
    print("Starting CatBoost optimization...")
    use_gpu = False  # Set to True to use GPU
    hyperparameters_trials = 3

    # Load and prepare data
    dataset = pd.read_csv('../../data/cleaned_dataset_processed_balanced.csv').sample(n=1000, random_state=42)
    X = dataset['cleaned_review']
    y = dataset['rating']
    X_train, X_test, y_train, y_test = split_data_stratified(X, y)
    
    # Train and optimize model hyperparameters
    train(X_train, X_test, y_train, y_test, trials=hyperparameters_trials)