import os
import base64
import optuna
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
                                accuracy_score,
                                precision_score,
                                recall_score,
                                confusion_matrix,
                                f1_score,
                                classification_report, 
                                roc_auc_score,
                                auc, 
                                roc_curve,
                                ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler 

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from IPython.display import display, Markdown
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, learning_curve

from catboost import CatBoostClassifier
from catboost import CatBoostError

from optuna.storages import RDBStorage
from optuna.integration import CatBoostPruningCallback
from optuna.terminator.callback import TerminatorCallback 

from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from wordcloud import WordCloud
from IPython.display import display, HTML



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

class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).ravel()
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

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

def test_scoring(model, description, y_train, y_train_pred, y_test, y_test_pred, time_train = None):
    # Confusion matrices (normalizadas)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    display_cm_normalized(cm_train, f"{description} (train)")
    display_cm_normalized(cm_test, f"{description} (test)")

    # Métricas globales y por clase
    display("Y Test",np.unique(y_test, return_counts=True))
    display("Y Test Predictions", np.unique(y_test_pred, return_counts=True))
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Reestructurar el DataFrame
    report_df = report_df.reset_index().rename(columns={"index": "Clase"})
    report_df["accuracy"] = np.nan  # Añadir columna para "accuracy"
    report_df["time_train"] = np.nan 
    report_df.loc[report_df["Clase"] == "accuracy", ["precision", "recall", "f1-score", "support"]] = np.nan
    report_df.loc[report_df["Clase"] == "accuracy", "accuracy"] = report_dict["accuracy"]

    time_row = {"Clase": "time_train", "precision": np.nan, "recall": np.nan, "f1-score": np.nan, 
            "support": np.nan, "accuracy": np.nan, "Modelo": model, "time_train": time_train}
    report_df = pd.concat([report_df, pd.DataFrame([time_row])], ignore_index=True)

    report_df["Modelo"] = model

    # Guardar corrida
    db_dir = "db/model-dbs/"
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "model_reports.db")
    conn = sqlite3.connect(db_path)
    table_name = description  # Nombre de la tabla en SQLite
    report_df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    # Mostrar métricas generales
    print(f"Score for {description}")
    display(report_df)

    # Devolver la matriz de confusión y el reporte reestructurado
    return cm_test, report_df

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
                try:
                    step.set_params(**params_to_update)  # Actualiza el modelo directamente
                except CatBoostError as e:
                    display(step)
                    print("Recreating catboost classifier due to error: {e}")
                    step = CatBoostClassifier(learning_rate= 0.02, task_type="GPU", random_state=random_state)
                    step.set_params(**params_to_update)
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
    db_dir = "./db/optuna-dbs"
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

### Funciones para plot de pre procesamiento de datos ###    
def highlight_differences(text1, text2):
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    different_words = words1.symmetric_difference(words2)
    
    html = f"""
    <div style="display: flex; gap: 20px;">
        <div style="flex: 1;">
            <h4>Original:</h4>
            <p>{''.join([f'<span style="background-color: orange">{w}</span> ' if w in different_words else f'{w} ' for w in text1.split()])}</p>
        </div>
        <div style="flex: 1;">
            <h4>Cleaned:</h4>
            <p>{''.join([f'<span style="background-color: orange">{w}</span> ' if w in different_words else f'{w} ' for w in text2.split()])}</p>
        </div>
    </div>
    """
    return HTML(html)

def create_rating_wordcloud(df, rating=None, 
                          title=None,
                          width=800, 
                          height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=100):
    """
    Create word cloud from reviews with specific rating(s)
    
    Parameters:
    - df: pandas DataFrame with 'rating' and 'cleaned_review' columns
    - rating: int or list of ints (1-5), if None shows all reviews
    - title: custom title, if None auto-generates based on rating
    """
    
    # Filter by rating if specified
    if rating is not None:
        if isinstance(rating, (int, float)):
            df_filtered = df[df['rating'] == rating]
            default_title = f'Word Cloud - Rating {rating}'
        elif isinstance(rating, list):
            df_filtered = df[df['rating'].isin(rating)]
            default_title = f'Word Cloud - Ratings {rating}'
    else:
        df_filtered = df
        default_title = 'Word Cloud - All Ratings'
    
    # Check if we have reviews
    if len(df_filtered) == 0:
        print(f"No reviews found for rating(s): {rating}")
        return None
    
    # Combine all filtered reviews
    text = ' '.join(df_filtered['cleaned_review'].astype(str))
    
    # Create and configure WordCloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        random_state=42
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title or default_title, fontsize=16, pad=20)
    plt.tight_layout(pad=0)
    plt.show()
    
    # Return word frequencies
    return dict(sorted(wordcloud.words_.items(), 
                      key=lambda x: x[1], 
                      reverse=True))

def create_rating_wordcloud_comparison(df, rating=None, rating_bis=None,
                          title=None,
                          title_bis=None,
                          width=800, 
                          height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=100):
    """
    Create word cloud from reviews with specific rating(s)
    
    Parameters:
    - df: pandas DataFrame with 'rating' and 'cleaned_review' columns
    - rating: int or list of ints (1-5), if None shows all reviews
    - rating_bis: int or list of ints (1-5), if None shows all reviews
    - title: custom title, if None auto-generates based on rating
    """
    
    # Filter by rating if specified
    if rating is not None:
        if isinstance(rating, (int, float)):
            df_filtered = df[df['rating'] == rating]
            default_title = f'Word Cloud - Rating {rating}'
        elif isinstance(rating, list):
            df_filtered = df[df['rating'].isin(rating)]
            default_title = f'Word Cloud - Ratings {rating}'
    else:
        df_filtered = df
        default_title = 'Word Cloud - All Ratings'
    
    # Filter by rating_bis if specified
    if rating_bis is not None:
        if isinstance(rating_bis, (int, float)):
            df_filtered_bis = df[df['rating'] == rating_bis]
            default_title_bis = f'Word Cloud - Rating {rating_bis}'
        elif isinstance(rating_bis, list):
            df_filtered_bis = df[df['rating'].isin(rating_bis)]
            default_title_bis = f'Word Cloud - Ratings {rating_bis}'
    else:
        df_filtered_bis = df
        default_title_bis = 'Word Cloud - All Ratings'
    
    # Check if we have reviews
    if len(df_filtered) == 0:
        print(f"No reviews found for rating(s): {rating}")
        return None
    
    if len(df_filtered_bis) == 0:
        print(f"No reviews found for rating(s): {rating_bis}")
        return None
    
    # Combine all filtered reviews
    text = ' '.join(df_filtered['cleaned_review'].astype(str))
    text_bis = ' '.join(df_filtered_bis['cleaned_review'].astype(str))
    
    # Create and configure WordCloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        random_state=42
    ).generate(text)
    
    wordcloud_bis = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        random_state=42
    ).generate(text_bis)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(width/100*2, height/100))
    
    axes[0].imshow(wordcloud, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title(title or default_title, fontsize=16, pad=20)
    
    axes[1].imshow(wordcloud_bis, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title(title_bis or default_title_bis, fontsize=16, pad=20)
    
    plt.tight_layout(pad=0)
    plt.show()
    
    # Return word frequencies
    return {
        'rating': dict(sorted(wordcloud.words_.items(), key=lambda x: x[1], reverse=True)),
        'rating_bis': dict(sorted(wordcloud_bis.words_.items(), key=lambda x: x[1], reverse=True))
    }

def plot_rating_distribution(dataset, figsize=(15, 6), palette='viridis'):
    """
    Creates a visualization of rating distribution with bar plot and pie chart.
    
    Parameters:
    -----------
    dataset : pandas DataFrame
        DataFrame containing a 'rating' column
    figsize : tuple, optional
        Figure size for the plots (width, height)
    palette : str, optional
        Color palette to use for the plots
    
    Returns:
    --------
    dict
        Dictionary containing the numerical summaries
    """
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate total records
    total_records = len(dataset)
    
    # 1. Bar plot with seaborn
    sns.countplot(data=dataset, x='rating', ax=ax1, palette=palette)
    ax1.set_title(f'Rating Distribution (Bar Plot)\nTotal Records: {total_records:,}')
    ax1.set_xlabel('Rating')
    ax1.set_ylabel('Count')
    
    # 2. Pie chart with matplotlib
    rating_counts = dataset['rating'].value_counts()
    colors = sns.color_palette(palette, n_colors=len(rating_counts))
    ax2.pie(rating_counts, 
            labels=rating_counts.index, 
            autopct='%1.1f%%', 
            colors=colors)
    ax2.set_title('Rating Distribution (Pie Chart)')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Calculate numerical summaries
    counts = dataset['rating'].value_counts().sort_index()
    percentages = dataset['rating'].value_counts(normalize=True).sort_index().mul(100).round(1)
    
    # Print numerical summary
    print("\nNumerical Summary:")
    print(counts)
    print("\nPercentage Distribution:")
    print(percentages)
    
    # Return the summaries as a dictionary
    return {
        'counts': counts,
        'percentages': percentages
    }

def plot_rating_distribution_comparison(dataset, dataset_original, column='rating', figsize=(15, 12), palette='viridis'):
    """
    Creates a visualization of rating distribution with bar plot and pie chart for two datasets.
    
    Parameters:
    -----------
    dataset : pandas DataFrame
        DataFrame containing the specified column
    dataset_original : pandas DataFrame
        Original DataFrame containing the specified column
    column : str, optional
        Column name to use for the rating distribution (default is 'rating')
    figsize : tuple, optional
        Figure size for the plots (width, height)
    palette : str, optional
        Color palette to use for the plots
    
    Returns:
    --------
    dict
        Dictionary containing the numerical summaries for both datasets
    """
    
    # Create a figure with four subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate total records for both datasets
    total_records = len(dataset)
    total_records_original = len(dataset_original)
    
    # 1. Bar plot with seaborn for the first dataset
    sns.countplot(data=dataset, x=column, ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title(f'{column.capitalize()} Distribution (Bar Plot) - Cleaned\nTotal Records: {total_records:,}')
    axes[0, 0].set_xlabel(column.capitalize())
    axes[0, 0].set_ylabel('Count')
    
    # 2. Pie chart with matplotlib for the first dataset
    rating_counts = dataset[column].value_counts()
    colors = sns.color_palette(palette, n_colors=len(rating_counts))
    axes[0, 1].pie(rating_counts, 
                   labels=rating_counts.index, 
                   autopct='%1.1f%%', 
                   colors=colors)
    axes[0, 1].set_title(f'{column.capitalize()} Distribution (Pie Chart) - Cleaned')
    
    # 3. Bar plot with seaborn for the original dataset
    sns.countplot(data=dataset_original, x=column, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title(f'{column.capitalize()} Distribution (Bar Plot) - Original\nTotal Records: {total_records_original:,}')
    axes[1, 0].set_xlabel(column.capitalize())
    axes[1, 0].set_ylabel('Count')
    
    # 4. Pie chart with matplotlib for the original dataset
    rating_counts_original = dataset_original[column].value_counts()
    colors_original = sns.color_palette(palette, n_colors=len(rating_counts_original))
    axes[1, 1].pie(rating_counts_original, 
                   labels=rating_counts_original.index, 
                   autopct='%1.1f%%', 
                   colors=colors_original)
    axes[1, 1].set_title(f'{column.capitalize()} Distribution (Pie Chart) - Original')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Calculate numerical summaries for both datasets
    counts = dataset[column].value_counts().sort_index()
    percentages = dataset[column].value_counts(normalize=True).sort_index().mul(100).round(1)
    
    counts_original = dataset_original[column].value_counts().sort_index()
    percentages_original = dataset_original[column].value_counts(normalize=True).sort_index().mul(100).round(1)
    
    # Print numerical summary for both datasets
    print(f"\nNumerical Summary - Cleaned ({column.capitalize()}):")
    print(counts)
    print(f"\nPercentage Distribution - Cleaned ({column.capitalize()}):")
    print(percentages)
    
    print(f"\nNumerical Summary - Original ({column.capitalize()}):")
    print(counts_original)
    print(f"\nPercentage Distribution - Original ({column.capitalize()}):")
    print(percentages_original)
    
    # Return the summaries as a dictionary
    return {
        'cleaned': {
            'counts': counts,
            'percentages': percentages
        },
        'original': {
            'counts': counts_original,
            'percentages': percentages_original
        }
    }
    
def encode_image_to_base64(image_path):
    """
    Encode a PNG image to a base64 string.

    Args:
        image_path (str): Path to the PNG image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def load_model_info (models, model_reports="db/model-dbs/model_reports.db"):
    combined_reports = pd.DataFrame()
    if os.path.exists(model_reports):
        try:
            with sqlite3.connect(model_reports) as conn:
                for model in models:
                    table_name = model
                    query = f"SELECT * FROM {table_name}"
                    try:
                        df = pd.read_sql(query, conn)
                        # Concatenar el DataFrame con los reportes combinados
                        combined_reports = pd.concat([combined_reports, df], ignore_index=True)
                    except Exception as e:
                        print(f"Error leyendo la tabla {table_name}: {e}")
        except Exception as e:
            print(f"Error al conectar con la base de datos: {e}")
    else:
        print(f"La base de datos {model_reports} no existe.")
    display(combined_reports)
    return combined_reports

def best_model (weighted_avg_report):
    # Seleccionar las métricas clave
    metrics_to_compare = ["precision", "recall", "f1-score", "f1-score_normalized"]
    weighted_metrics_comparison = weighted_avg_report[["Modelo"] + metrics_to_compare].set_index("Modelo")

    # Mostrar resultados numéricos
    print("\nMétricas ponderadas por modelo:")
    display(weighted_metrics_comparison)

    # Visualización de métricas ponderadas
    weighted_metrics_comparison.plot(kind="bar", figsize=(12, 8), alpha=0.85)
    plt.title("Comparación de Métricas Ponderadas por Modelo")
    plt.xlabel("Modelo")
    plt.ylabel("Métrica Ponderada")
    plt.legend(title="Métricas", loc="best")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Determinar el mejor modelo (según F1-score ponderado)
    best_model = weighted_metrics_comparison["f1-score"].idxmax()
    best_f1_score = weighted_metrics_comparison.loc[best_model, "f1-score"]
    print(f"\nEl mejor modelo por f1-score es {best_model} con un F1-Score ponderado de {best_f1_score:.4f}.")

    best_model = weighted_metrics_comparison["f1-score_normalized"].idxmax()
    best_f1_time = weighted_metrics_comparison.loc[best_model, "f1-score_normalized"]
    print(f"\nEl mejor modelo por relación score/tiempo es {best_model} con un F1-Score normalizado de {best_f1_time:.4f}.")
    return best_model, best_f1_time

def compare_models(models, cms=None):
    combined_reports = load_model_info(models)
    # Filtrar para quitar las filas no relevantes (eliminamos 'accuracy' de las métricas por clase)
    class_only_reports = combined_reports[~combined_reports["Clase"].isin(["accuracy", "macro avg", "weighted avg"])]
    # Convertir "Clase" a tipo numérico para ordenar adecuadamente en el gráfico
    class_only_reports.loc[:, "Clase"] = pd.to_numeric(class_only_reports["Clase"], errors="coerce")
    # Crear un dataframe solo con las filas de 'accuracy' para mostrarlo aparte
    accuracy_report = combined_reports[combined_reports["Clase"] == "accuracy"]
    # Crear un dataframe con los puntajes ponderados
    weighted_avg_report = combined_reports[combined_reports["Clase"] == "weighted avg"]
    # Filtrar la fila con 'time_train' en el DataFrame combinado
    time_train_value = combined_reports[['time_train', 'Modelo']][combined_reports['time_train'].notna()]
    # Agregar la columna 'time_train' al DataFrame 'weighted_avg_report'
    weighted_avg_report = pd.merge(weighted_avg_report.drop('time_train', axis=1), time_train_value, on='Modelo', how='left')

    # Crear los gráficos para las métricas por clase (sin incluir accuracy)
    metrics = ['precision', 'recall', 'f1-score']
    plt.figure(figsize=(18, 14))  # Aumentar el ancho y alto del diseño para mejor espacio

    # Graficar cada métrica (sin accuracy)
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 1, i)  # 3 filas, 2 columnas
        sns.barplot(data=class_only_reports, x="Clase", y=metric, hue="Modelo")
        plt.title(f"Comparación de {metric.capitalize()} por Clase", fontsize=14)
        plt.xlabel("Clase", fontsize=12)
        plt.ylabel(f"{metric.capitalize()}", fontsize=12)

        # Colocar la leyenda fuera del gráfico
        plt.legend(title="Modelo", fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

        # Agregar líneas punteadas para los valores mínimo y máximo
        min_val = class_only_reports[metric].min()
        max_val = class_only_reports[metric].max()
        plt.axhline(min_val, color='red', linestyle='--', label=f"Min {metric.capitalize()}")
        plt.axhline(max_val, color='green', linestyle='--', label=f"Max {metric.capitalize()}")

        # Etiquetas para las líneas mínimas y máximas (fuera del gráfico para no solaparse)
        plt.text(class_only_reports["Clase"].max() + 0.5, min_val, f'{min_val:.2f}', 
                 color='red', ha='center', va='bottom', fontsize=10)
        plt.text(class_only_reports["Clase"].max() + 0.5, max_val, f'{max_val:.2f}', 
                 color='green', ha='center', va='bottom', fontsize=10)
    plt.show()

    plt.figure(figsize=(18, 6))

    # Gráfico del f1-score ponderado por modelo
    plt.subplot(1, 2, 1)
    sns.barplot(data=weighted_avg_report, x="Modelo", y="f1-score", color="lightblue")
    plt.title("F1-Score Ponderado por Modelo", fontsize=14)
    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Etiquetas en diagonal con mayor tamaño
    plt.yticks(fontsize=10)  # Aumentar el tamaño de las etiquetas del eje Y
    plt.yscale('log')

    # Líneas horizontales para valores máximo y mínimo
    min_val = weighted_avg_report["f1-score"].min()
    max_val = weighted_avg_report["f1-score"].max()
    plt.axhline(min_val, color='red', linestyle='--', label=f"Min F1-Score: {min_val:.2f}")
    plt.axhline(max_val, color='green', linestyle='--', label=f"Max F1-Score: {max_val:.2f}")

    # Etiquetas para las líneas
    plt.text(0, min_val, f'{min_val:.2f}', color='black', ha='left', va='top', fontsize=10)
    plt.text(0, max_val, f'{max_val:.2f}', color='black', ha='left', va='bottom', fontsize=10)

    # Gráfico del f1-score normalizado por tiempo de entrenamiento
    plt.subplot(1, 2, 2)
    weighted_avg_report["f1-score_normalized"] = weighted_avg_report["f1-score"] / np.log(weighted_avg_report['time_train'])
    sns.barplot(data=weighted_avg_report, x="Modelo", y="f1-score_normalized", color="orange")
    plt.title("F1-Score Normalizado por Tiempo de Entrenamiento", fontsize=14)
    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("F1-Score Normalizado", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Etiquetas en diagonal
    plt.yticks(fontsize=10)  # Aumentar el tamaño de las etiquetas del eje Y
    plt.yscale('log')
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

    if cms is not None:
        # Graficar las matrices de confusión
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Aumentar el tamaño de la figura
        # Iterar sobre cada matriz y su subplot
        for ax, cm, name in zip(axes.ravel(), cms, models):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(name)

    # Ajustar diseño de las matrices de confusión
    plt.tight_layout()
    plt.show()

    best_model(weighted_avg_report)


### Regresion logistica  
def split_data_stratified(X, y, test_size=0.2, random_state=42):
    """
    Split data ensuring proportional representation of all classes
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Get indices for train and test splits
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

def plot_learning_curves(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """Plot learning curves for a given estimator"""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

def plot_grid_search_results(grid_search, param_name):
    """Plot grid search results for a specific parameter"""
    # Clear any existing plots
    plt.clf()
    
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Get the parameter values
    param_values = [params[param_name] for params in results['params']]
    
    # Calculate mean scores for each parameter value
    mean_scores = {}
    std_scores = {}
    for value, mean, std in zip(param_values, 
                               results['mean_test_score'],
                               results['std_test_score']):
        if value not in mean_scores:
            mean_scores[value] = []
            std_scores[value] = []
        mean_scores[value].append(mean)
        std_scores[value].append(std)
    
    # Improved sorting function
    def sort_key(x):
        if x is None:
            return float('inf')  # Place None at the end
        return float(x) if isinstance(x, (int, float)) else str(x)
    
    # Separate numeric and None values
    numeric_values = [v for v in set(param_values) if v is not None]
    has_none = None in set(param_values)
    
    # Sort numeric values
    unique_values = sorted(numeric_values, key=float)
    if has_none:
        unique_values.append(None)
    
    # Calculate average for each parameter value
    mean_scores = [np.mean(mean_scores[value]) for value in unique_values]
    std_scores = [np.mean(std_scores[value]) for value in unique_values]
    
    # Convert values to strings for display
    x_labels = ['None' if v is None else str(v) for v in unique_values]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(unique_values)), mean_scores, 
                yerr=std_scores, fmt='o-')
    
    # Set the x-axis labels
    plt.xticks(range(len(unique_values)), x_labels, rotation=45)
    
    # Labels and title
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Grid Search Results for {param_name}')
    plt.grid(True)
    
    # Add value labels
    for i, score in enumerate(mean_scores):
        plt.text(i, score, f'{score:.3f}', 
                horizontalalignment='center',
                verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_all_param_results(grid_search, param_grid):
    """Plot grid search results for all parameters"""
    n_params = len(param_grid)
    if n_params == 0:
        return
        
    # Calculate subplot layout
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    plt.figure(figsize=(6*n_cols, 4*n_rows))
    
    for i, param_name in enumerate(param_grid.keys(), 1):
        plt.subplot(n_rows, n_cols, i)
        
        results = pd.DataFrame(grid_search.cv_results_)
        param_values = [params[param_name] for params in results['params']]
        
        # Custom sorting function that handles None
        def sort_key(x):
            if x is None:
                return float('inf')  # Place None at the end
            return x
        
        # Calculate mean scores for each parameter value
        mean_scores = {}
        std_scores = {}
        for value, mean, std in zip(param_values, 
                                   results['mean_test_score'],
                                   results['std_test_score']):
            if value not in mean_scores:
                mean_scores[value] = []
                std_scores[value] = []
            mean_scores[value].append(mean)
            std_scores[value].append(std)
        
        # Sort values handling None
        unique_values = sorted(set(param_values), key=sort_key)
        mean_scores = [np.mean(mean_scores[value]) for value in unique_values]
        std_scores = [np.mean(std_scores[value]) for value in unique_values]
        
        # Convert values to strings for display
        x_labels = ['None' if v is None else str(v) for v in unique_values]
        
        # Plot
        plt.errorbar(range(len(unique_values)), mean_scores, 
                    yerr=std_scores, fmt='o-')
        plt.xticks(range(len(unique_values)), x_labels, rotation=45)
        plt.xlabel(param_name)
        plt.ylabel('Mean CV Score')
        plt.title(f'Grid Search Results - {param_name}')
        plt.grid(True)
        
        # Add value labels
        for j, score in enumerate(mean_scores):
            plt.text(j, score, f'{score:.3f}', 
                    horizontalalignment='center',
                    verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

def create_param_grids():
    """Define parameter grids for each model"""
    param_grids = {
        'Logistic Regression': {
            'vectorizer__max_features': [1000, 3000, 8000, 10000, 15000], # [1000, 3000, 8000, 10000, 15000]
            'vectorizer__ngram_range': [(1,1), (1,2)],
            'classifier__C': [0.1, 1.0, 10.0, 12.0],
            'classifier__class_weight': ['balanced'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['saga', 'liblinear']
        },
        'With Scaling': {
            'vectorizer__max_features': [1000, 3000, 8000, 10000, 15000],
            'vectorizer__ngram_range': [(1,1), (1,2)],
            'scaler__with_mean': [False],  # Correct way to parameterize scaler
            'classifier__C': [0.1, 1.0, 10.0, 12.0],
            'classifier__class_weight': ['balanced'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['saga', 'liblinear']
        },
        'With PCA': {
            'vectorizer__max_features': [1000, 3000, 8000, 10000, 15000],
            'vectorizer__ngram_range': [(1,1), (1,2)],
            'dim_reduction__n_components': [50, 100, 200],
            'classifier__C': [0.1, 1.0, 10.0, 12.0],
            'classifier__class_weight': ['balanced'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['saga', 'liblinear']
        },
        'PCA + Scaling': {
            'vectorizer__max_features': [1000, 3000, 8000, 10000, 15000],
            'vectorizer__ngram_range': [(1,1), (1,2)],
            'dim_reduction__n_components': [50, 100, 200],
            'scaler__with_mean': [False],  # Correct way to parameterize scaler
            'classifier__C': [0.1, 1.0, 10.0, 12.0],
            'classifier__class_weight': ['balanced'],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['saga', 'liblinear']
        },

    }
    return param_grids

def compare_models_with_grid_search(X_train, X_test, y_train, y_test, n_jobs=10, cv=3, classes="3"):
    # Transform labels from 1-5 to 0-4
    y_train_transformed = y_train - 1
    y_test_transformed = y_test - 1

    # Get number of unique classes
    n_classes = len(np.unique(np.concatenate([y_train_transformed, y_test_transformed])))
    
    # Define pipelines
    pipelines = {
        'Logistic Regression': Pipeline([
            ('vectorizer', TfidfVectorizer(
                min_df=10,
                max_df=0.95,)
             ),
            ('classifier', LogisticRegression(max_iter=4000))
        ]),      
        'With Scaling': Pipeline([
            ('vectorizer', TfidfVectorizer(
                min_df=10,
                max_df=0.95,)
             ),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=4000))
        ]), 
        'With PCA': Pipeline([
            ('vectorizer', TfidfVectorizer(
                min_df=10,
                max_df=0.95,)
             ),
            ('dim_reduction', TruncatedSVD(n_components=100)),
            ('classifier', LogisticRegression(max_iter=4000))
        ]),
        
        'PCA + Scaling': Pipeline([
            ('vectorizer', TfidfVectorizer(
                min_df=10,
                max_df=0.95,)
             ),
            ('dim_reduction', TruncatedSVD(n_components=100)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=4000))
        ])
    }

    param_grids = create_param_grids()
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name} with Grid Search...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_transformed)
        
        # Calculate fit time
        fit_time = grid_search.refit_time_

        # Store best model results
        y_pred = grid_search.predict(X_test)
        y_pred_original = y_pred + 1
        y_test_original = y_test_transformed + 1
        
        y_pred_train = grid_search.predict(X_train)
        y_pred_original_train = y_pred_train + 1
        y_train_original = y_train_transformed + 1

        results[name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'train_score': grid_search.score(X_train, y_train_transformed),
            'test_score': grid_search.score(X_test, y_test_transformed),
            'classification_report': classification_report(y_test_original, y_pred_original),
            'confusion_matrix': confusion_matrix(y_test_original, y_pred_original),
            'grid_search': grid_search,
            'fit_time': fit_time,
            'f1_score': f1_score(y_test_original, y_pred_original, average='weighted'),
            'f1_score_train': f1_score(y_train_original, y_pred_original_train, average='weighted')
        }

        # Print results
        print(f"\n{name} Results:")
        print(f"Best parameters: {results[name]['best_params']}")
        print(f"Best CV score: {results[name]['best_score']:.3f}")
        print(f"Training accuracy: {results[name]['train_score']:.3f}")
        print(f"Testing accuracy: {results[name]['test_score']:.3f}")
        print(f"Fit time: {results[name]['fit_time']:.3f} seconds")
        print("\nClassification Report:")
        print(results[name]['classification_report'])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = results[name]['confusion_matrix']
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix (%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        test_scoring(f'logistic_regression_{name}', f'logistic_regression_{name}'+str(classes), y_train, y_pred_original_train, y_test, y_pred_original, time_train=fit_time)

        # Plot learning curves for best model
        plot_learning_curves(
            grid_search.best_estimator_,
            f'Learning Curves - {name}',
            X_train, y_train_transformed
        )

        # Plot grid search results for all parameters
        print(f"\nGrid Search Results for {name}:")
        plot_all_param_results(grid_search, param_grids[name])

        # Add this after other plots
        print(f"\nTop Features for {name}:")
        plot_top_features(grid_search.best_estimator_, name)
        print(f"\nClass-specific Features for {name}:")
        plot_top_features_by_class(grid_search.best_estimator_, name)

    return results

def plot_model_comparison(results):
    """Plot comparison of model performances"""
    try:
        # Prepare data
        models = list(results.keys())
        train_scores = [results[m]['train_score'] for m in models]
        test_scores = [results[m]['test_score'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        f1_scores_train = [results[m]['f1_score_train'] for m in models]

        # Create plot for accuracy
        x = np.arange(len(models))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        rects1 = ax1.bar(x - width/2, train_scores, width, label='Train')
        rects2 = ax1.bar(x + width/2, test_scores, width, label='Test')

        # Customize plot for accuracy
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison - Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45)
        ax1.legend()

        # Add value labels for accuracy
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')

        autolabel(rects1, ax1)
        autolabel(rects2, ax1)

        # Create plot for F1-score
        rects3 = ax2.bar(x - width/2, f1_scores_train, width, label='Train')
        rects4 = ax2.bar(x + width/2, f1_scores, width, label='Test')

        # Customize plot for F1-score
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Model Performance Comparison - F1 Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.legend()

        # Add value labels for F1-score
        autolabel(rects3, ax2)
        autolabel(rects4, ax2)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in model comparison plot: {str(e)}")

def plot_top_features_by_class(model, model_name, n_features=10):
    """
    Plot top features for each class separately (for models that support it)
    """
    try:
        vectorizer = model.named_steps['vectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        if model_name in ['Logistic Regression', 'SVM']:
            coefficients = model.named_steps['classifier'].coef_
            n_classes = coefficients.shape[0]
            
            # Create subplot grid
            n_cols = 2
            n_rows = (n_classes + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.ravel()
            
            for i in range(n_classes):
                # Get top features for this class
                class_coef = coefficients[i]
                top_positive_idx = np.argsort(class_coef)[-n_features:]
                top_negative_idx = np.argsort(class_coef)[:n_features]
                
                # Plot
                ax = axes[i]
                y_pos = np.arange(n_features * 2)
                
                # Combine positive and negative features
                features = np.concatenate([
                    feature_names[top_negative_idx],
                    feature_names[top_positive_idx]
                ])
                importances = np.concatenate([
                    class_coef[top_negative_idx],
                    class_coef[top_positive_idx]
                ])
                
                ax.barh(y_pos, importances)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_title(f'Class {i+1} Important Features')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Could not plot class-specific features for {model_name}: {str(e)}")

def plot_top_features(model, model_name, n_features=20):
    """
    Plot top n_features for each model
    """
    try:
        # Get vectorizer and feature names
        vectorizer = model.named_steps['vectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importance based on model type
        if model_name in ['Random Forest', 'XGBoost']:
            # Tree-based models
            importances = model.named_steps['classifier'].feature_importances_
            importance_type = 'Feature Importance'
        
        elif model_name == 'Logistic Regression':
            # Get absolute values of coefficients (average across classes for multiclass)
            importances = np.abs(model.named_steps['classifier'].coef_).mean(axis=0)
            importance_type = 'Coefficient Magnitude'
        
        elif model_name == 'SVM':
            # For LinearSVC, get coefficients from the underlying estimator
            importances = np.abs(model.named_steps['classifier'].estimator.coef_).mean(axis=0)
            importance_type = 'Coefficient Magnitude'
        
        elif model_name == 'Naive Bayes':
            # For Naive Bayes, use feature log probabilities
            importances = np.exp(model.named_steps['classifier'].feature_log_prob_).mean(axis=0)
            importance_type = 'Feature Probability'
        
        # Get top n features
        top_indices = np.argsort(importances)[-n_features:]
        top_features = feature_names[top_indices]
        top_importances = importances[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(n_features), top_importances[::-1])
        plt.yticks(range(n_features), top_features[::-1])
        plt.xlabel(importance_type)
        plt.title(f'Top {n_features} Most Important Features - {model_name}')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not plot feature importance for {model_name}: {str(e)}")