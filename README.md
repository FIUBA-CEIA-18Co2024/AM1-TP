# AM-TP
Trabajo Práctico para la materia Aprendizaje de Máquia I de la Carrera de Especialización en Inteligencia Artificial (CEIA)

Tema: Analisis de sentimiento

Objetivo: Entrenar modelo para identificar analisis de sentimiento de tweets o titulares de diarios. Comparar el modelo con LLMs SOTA en la tarea e implementarlo en API REST con el objetivo de poder ser consumido via web chat UI (streamlit)

Modelos: SVM, XGBOOST, otros


Datasets:
- https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis
- https://huggingface.co/VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis
  - http://tass.sepln.org/tass_data/download.php?auth=Ot4aysCsd49ewvWesr6
- https://github.com/Observatorio-de-Redes/politicosentwitter?tab=readme-ov-file
- https://huggingface.co/datasets/pysentimiento/spanish-targeted-sentiment-headlines (!!)
- https://github.com/taljuk01/political_sentiment_analysis/tree/master (!!)
  - https://drive.google.com/file/d/1qHfM_APQ4BTpzUdQc7X_51RGU5Mpi5De/view
- https://huggingface.co/datasets/NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1
- https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews (!!)
- http://www.t4sa.it
  - user: t4sa
    pass: U4Cm_dUa
- https://www.kaggle.com/datasets/kazanova/sentiment140 (!!)

Tal vez se puedan mergear spanish targeted con political sentiment analysis dando un dataset con al menos 10k muestras.