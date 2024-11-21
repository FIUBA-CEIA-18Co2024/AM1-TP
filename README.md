# AM-TP
Trabajo Práctico para la materia Aprendizaje de Máquia I de la Carrera de Especialización en Inteligencia Artificial (CEIA)

Tema: Analisis de sentimiento

Objetivo: Entrenar modelo para identificar analisis de sentimiento de tweets o titulares de diarios. Comparar el modelo con LLMs SOTA en la tarea e implementarlo en API REST con el objetivo de poder ser consumido via web chat UI (streamlit)

Modelos: SVM, XGBOOST, otros

### Evaluación
La evaluación de los conocimientos impartidos durante las clases será a modo de entrega de trabajo práctico final. La entrega de trabajo final de la cursada comprende de una investigación, desarrollo y resultados finales basados en un set de datos a elección por el grupo (máximo 6 personas).
Criterios de aprobación

### Criterios de aprobación
- Obligación de trabajar en grupo mínimo de 2 y máximo de 6. Excepciones se pueden hacer mediante un correcto justificativo.
- Cada TP debe citar la fuente de información de evaluación, es decir, citar de donde se obtuvieron los datos.
-  Puede ser entregado en cualquier formato, preferentemente en notebook de ipython (formato ipynb). También puede entregarse mediante un documento en Google Colab.
- Contendrá la propuesta de investigación en los datos, citando el porqué de la evaluación y que se pretende encontrar o descubrir con dicha investigación
- Deberá explicar el porqué de la elección del algoritmo empleado para la resolución del problema y cómo llegó a esa elección por sobre otras opciones
- Deberá expresar de manera clara el resultado de la investigación aportando las métricas necesarias para comprender cómo se desempeña el algoritmo elegido.
- El entregable debe venir con el código acompañado para su reproducibilidad.
- En su cierre debe dar una reflexión de su investigación y proponer nuevos caminos de resolución en el caso de que sea necesario.
- Como última fecha de entrega del trabajo, estará limitado a 7 días posteriores a la última clase.
- La entrega es por el aula virtual, ya sea el envío del contenido o el link a repositorio (de GitHub o GitLab) con el trabajo.

### Estrategia de trabajo:
- GridSearch (hyper param optimization)
- PCA
- Regularizacion
- Balanceo de clases (SMOTE, otros)
- Cross Validation
- Metrics (f1-score, accuracy, recall, precision)
- Modelos: logistic regression, SVM, xgboost
- Agrupar en 3 clases
- Comparativo con modelos basado en LLM/NN SOTA

### Datasets:
Se utiliza dataset de reviews de estadía en hoteles de la empresa TripAdvisor. Se generan filas sinteticas extras con Llama3-1B-Instruct para balancear las clases.
Como se toman estos datos, no como augmentation, sino como reseñas nuevas, las muestras se utilizaran tanto para splits de train como de test.

- tripadvisor_hotel_reviews-balanced.csv: Dataset original Kaggle + datos sinteticos generados con LLM
- cleaned_dataset_processed_balanced.csv: Dataset pre-procesado usando tripadvisor_hotel_reviews-balanced.csv
- cleaned_dataset_100.csv: Dataset pre-procesado usando tripadvisor_hotel_reviews-balanced.csv con solo 100 muestras
- dataset.csv: Dataset original Kaggle

*Original*: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews


Tal vez se puedan mergear spanish targeted con political sentiment analysis dando un dataset con al menos 10k muestras.
