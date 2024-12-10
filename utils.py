import pandas as pd
import os

from sklearn.metrics import (
                                accuracy_score,
                                precision_score,
                                recall_score,
                                confusion_matrix,
                                f1_score,
                                classification_report, 
                                roc_auc_score,
                                auc, 
                                roc_curve
)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC, LinearSVC
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold

import optuna
from optuna.terminator.callback import TerminatorCallback 
import sqlite3
from optuna.storages import RDBStorage
from optuna.integration import CatBoostPruningCallback

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

random_state = 42
max_iter = 20000
# Etapas del pipeline para distintos modelos
pipeline_steps = {
    "linear_svc" : [
        ("vectorizer", TfidfVectorizer(min_df=10, max_df=0.9, ngram_range=(1, 2))),
        ("scaler", MaxAbsScaler()),
        ("classifier", OneVsOneClassifier(LinearSVC(class_weight="balanced", max_iter=max_iter, random_state=random_state)))
    ],
    "svc_linear": [
        ("vectorizer", TfidfVectorizer(min_df=10, max_df=0.9, ngram_range=(1, 2), max_features=10000)),
        ("scaler", MaxAbsScaler()),
        ("classifier", SVC(kernel="linear", class_weight="balanced", decision_function_shape="ovo", probability = True, max_iter=max_iter, random_state=random_state))
    ],
    "svc_rbf": [
        ("vectorizer", TfidfVectorizer(min_df=10, max_df=0.9, ngram_range=(1, 2))),
        ("dim_reduction", TruncatedSVD(random_state=random_state)),
        ("scaler", MinMaxScaler()),
        ("classifier", SVC(kernel="rbf", class_weight="balanced", decision_function_shape="ovo", probability = True, max_iter=max_iter, random_state=random_state))
    ],
    "catboost": [
        ("vectorizer", TfidfVectorizer(min_df=10, max_df=0.9, ngram_range=(1, 2))),
        ("dim_reduction", TruncatedSVD(random_state=random_state)),
        ("scaler", MinMaxScaler()),
        ("classifier", CatBoostClassifier(learning_rate= 0.02, task_type="GPU", random_state=random_state))
    ],
}

def display_cm_normalized(cm, model=None):
    """
    Display a normalized confusion matrix with percentages.

    Parameters:
        cm (numpy.ndarray): Confusion matrix to normalize and display.
        model (str, optional): Model name to include in the title. Defaults to None.
    """
    # Normalize the confusion matrix
    cm_normalized = 100 * cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Generate class labels dynamically based on the confusion matrix size
    num_classes = cm.shape[0]
    class_labels = [str(i + 1) for i in range(num_classes)]  # '1', '2', ..., 'n'

    # Plot the confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and their labels
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_labels, yticklabels=class_labels,
           ylabel='True label',
           xlabel='Predicted label')

    # Add a title if a model name is provided
    if model is not None:
        ax.set(title=f'Normalized Confusion Matrix {model}')

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations in the cells
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_normalized[i, j]:{fmt}}%",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

def test_scoring (model, description, y_train, y_train_pred, y_test, y_test_pred):
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    display_cm_normalized(cm_train, f"{description} (train)")
    display_cm_normalized(cm_test, f"{description} (test)")

    print(f"Score for {description}")
    print(f"Precisión: {precision_score(y_test, y_test_pred, zero_division=0, pos_label=1, average=None)}")
    print(f"Recuperación: {recall_score(y_test, y_test_pred, zero_division=0, pos_label=1, average=None)}")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"F1-score: {f1_score(y_test, y_test_pred, average=None)}")

    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["Modelo"] = model
    display(report_df)

    return(cm_test, report_df)

# No funciona con GPU
def catboost_cross_val_score(trial, pipeline, X, y, param, cv=3, eval_metric="Accuracy", early_stopping_rounds=100):
    """
    Validación cruzada manual para un pipeline que contiene CatBoostClassifier.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        # Dividir los datos
        train_x, valid_x = X.iloc[train_idx], X.iloc[valid_idx]
        train_y, valid_y = y.iloc[train_idx], y.iloc[valid_idx]

        # Resetear índices
        train_x = train_x.reset_index(drop=True)
        valid_x = valid_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=True)
        valid_y = valid_y.reset_index(drop=True)

        # Ajustar los hiperparámetros del modelo CatBoost dentro del pipeline
        steps = []
        for name, step in pipeline.steps:
            if isinstance(step, CatBoostClassifier):
                step.set_params(**param)
                steps.append((name, step))
            else:
                steps.append((name, step))

        pipeline_trained = Pipeline(steps)

        # Crear el callback de pruning para CatBoost
        pruning_callback = CatBoostPruningCallback(trial, eval_metric)

        # Entrenar el pipeline completo
        try:
            pipeline_trained.fit(
                train_x,
                train_y,
                classifier__eval_set=[(valid_x, valid_y)],
                classifier__verbose=0,
                classifier__early_stopping_rounds=early_stopping_rounds,
                classifier__callbacks=[pruning_callback],
            )
        except optuna.exceptions.TrialPruned:
            raise

        # Evaluar el pipeline en el conjunto de validación
        preds = pipeline_trained.predict(valid_x)
        accuracy = accuracy_score(valid_y, preds)
        scores.append(accuracy)

    return np.mean(scores)

CATBOOST_GPU = True
def objective(trial, pipeline_steps, param_distributions, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1):
    # Generar los hiperparámetros dinámicamente
    params = {}
    for step_name, step_params in param_distributions.items():
        if isinstance(step_params, dict): 
            params[step_name] = {}
            for param_name, (suggest_type, suggest_args) in step_params.items():
                if suggest_type == "suggest_int":
                    if len(suggest_args) == 3:
                        params[step_name][param_name] = trial.suggest_int(param_name, suggest_args[0], suggest_args[1], step = suggest_args[2])
                    else:
                        params[step_name][param_name] = trial.suggest_int(param_name, suggest_args)
                elif suggest_type == "suggest_float":
                    if len(suggest_args) == 3:
                        params[step_name][param_name] = trial.suggest_float(param_name, suggest_args[0], suggest_args[1], step = suggest_args[2])
                    else:
                        params[step_name][param_name] = trial.suggest_float(param_name, suggest_args)
                elif suggest_type == "suggest_categorical":
                    params[step_name][param_name] = trial.suggest_categorical(param_name, suggest_args)
                else:
                    raise ValueError(f"Unsupported suggestion type: {suggest_type} -> ({param_name} {suggest_args})")
        else:
            display("Error con los parametros")
    
    # Construir el pipeline dinámico con los parámetros
    dynamic_pipeline_steps = []
    for step_name, step in pipeline_steps:
        if step_name in params:  # Configurar solo si hay parámetros para este paso
            if hasattr(step, 'set_params'):  # Solo ajustar si el step tiene método 'set_params'
                # Revisar si el paso es un OneVsOneClassifier o un modelo con un estimador anidado
                if isinstance(step, OneVsOneClassifier):
                    # Ajustar los parámetros del estimador anidado (ej. LinearSVC)
                    step.set_params(estimator__C=params[step_name]['C'])
                else:
                    # Ajustar parámetros normalmente
                    step.set_params(**params[step_name])
                dynamic_pipeline_steps.append((step_name, step))
            else:
                dynamic_pipeline_steps.append((step_name, step))
        else:
            dynamic_pipeline_steps.append((step_name, step))
    pipeline = Pipeline(dynamic_pipeline_steps, verbose=True)
    
    if (any(isinstance(step[1], CatBoostClassifier) for step in dynamic_pipeline_steps)) and not CATBOOST_GPU:
        # Extraer los parámetros del clasificador
        param = params["classifier"]
        param.update({"eval_metric": scoring})  # Agregar la métrica de evaluación
        try:
            score = catboost_cross_val_score(trial, pipeline, X_train, y_train, param, cv=cv)
        except Exception as e:
            print(f"Error durante la validación cruzada de catboost: {e}")
            return float("-inf")  
    else:
        try:
            score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs).mean() #n_jobs=n_jobs
        except Exception as e:
            print(f"Error durante la validación cruzada: {e}")
            return float("-inf")  
    
    return score

# Función para actualizar parámetros de un paso
def update_step_params(pipeline, **new_params):
    for step in pipeline:
        if hasattr(step, 'set_params'):
            if isinstance(step, OneVsOneClassifier) and hasattr(step.estimator, 'set_params'):
                valid_params = step.estimator.get_params().keys()
                params_to_update = {key: value for key, value in new_params.items() if key in valid_params}
                step.estimator.set_params(**params_to_update) # Actualiza parámetros del estimador subyacente (por ejemplo, LinearSVC)
            else:
                valid_params = step.get_params().keys()
                params_to_update = {key: value for key, value in new_params.items() if key in valid_params}
                step.set_params(**params_to_update)  # Actualiza el modelo directamente
    return pipeline

def pipeline_config(model, pipeline_steps, storage=None):
    # Crear un estudio y guardarlo en un archivo SQLite
    storage = storage  # Archivo donde se guardará el progreso
    study_name = model         # Nombre del estudio
    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True)
    # Recrear el Pipeline con los mejores hiperparámetros
    best_params = study.best_params
    display(best_params)
    pipeline = Pipeline(pipeline_steps)
    pipeline = update_step_params(pipeline, **best_params)
    display(pipeline.get_params)
    return pipeline

def create_db(name, number=5, type="sqlite"):
    db_dir = "./optuna-dbs"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, f"{name}_{number}clases.db")

    if os.path.exists(db_path):
        if type == "sqlite":
            return f"sqlite:///{db_path}"
        elif type == "postgres":
            return f"postgresql://optuna_user:pass1234@localhost:5432/optuna"
        else:
            return None
    else:
        if type == "sqlite":
            db = f"sqlite:///{db_path}"
            db = RDBStorage(url=db, heartbeat_interval=60, grace_period=120)
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA max_page_count = 2147483646;")  # Incrementa el límite
            conn.close()
            return db
        elif type == "postgres":
            db = "postgresql://optuna_user:pass1234@localhost:5432/optuna"
            db = RDBStorage(url=db, heartbeat_interval=60, grace_period=120)
            return db
        else:
            return None        

def plot_tsne(X_tsne, y, title="Visualización t-SNE"):
    # Crear un mapeo de colores discretos
    unique_classes = sorted(y.unique())
    colors = plt.cm.tab10(range(len(unique_classes)))  # Usar tab10 para hasta 10 colores
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    labels = y.map(color_map)
    n_components = X_tsne.shape[1]
    
    if isinstance(X_tsne, pd.DataFrame):
        x_vals = X_tsne.iloc[:, 0]
        y_vals = X_tsne.iloc[:, 1]
    elif isinstance(X_tsne, np.ndarray):
        x_vals = X_tsne[:, 0]
        y_vals = X_tsne[:, 1]

    # Crear la figura
    fig = plt.figure(figsize=(10, 10))
    if n_components == 3:
        if isinstance(X_tsne, pd.DataFrame):
            z_vals = X_tsne.iloc[:, 2] if X_tsne.shape[1] == 3 else None
        elif isinstance(X_tsne, np.ndarray):
            z_vals = X_tsne[:, 2] if X_tsne.shape[1] == 3 else None
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            x_vals, 
            y_vals, 
            z_vals, 
            c=labels, 
            edgecolors='w', 
            alpha=0.6
        )
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
    elif n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            x_vals, 
            y_vals, 
            c=labels, 
            edgecolors='w', 
            alpha=0.6
        )
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=color, label=f'Clase {cls}', markersize=10, linestyle='')
        for cls, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='best', title='Clases')

    ax.set_title(title)
    plt.show()