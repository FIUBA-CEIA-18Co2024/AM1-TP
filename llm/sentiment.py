import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit



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

class ModelComparison:
    def __init__(self, run_dir="comparison_results"):
        load_dotenv()
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {
            "roberta-sentiment": "cardiffnlp/twitter-roberta-base-sentiment",
            "distilbert-sentiment": "distilbert-base-uncased-finetuned-sst-2-english"
        }
        self.run_dir = self._create_run_dir(run_dir)
        self.results = {}

    def _create_run_dir(self, base_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = f"{base_dir}/run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def prepare_dataset(self, random_state=42):
        """
        Prepare dataset using the exact split_data_stratified function from run.py
        """
        # Load dataset
        df = pd.read_csv('../data/cleaned_dataset_processed_balanced.csv')

        # Prepare X and y
        X = df[['cleaned_review']]
        y = df['three_classes']

        # Split dataset using the same function as run.py
        X_train, X_test, y_train, y_test = split_data_stratified(
            X,
            y,
            test_size=0.2,
            random_state=random_state
        )

        # Convert to list of strings for the text data
        test_texts = X_test['cleaned_review'].tolist()
        test_labels = y_test.tolist()

        return X_test['cleaned_review'].values, y_test.values

    def plot_performance_comparison(self):
        """
        Create bar plot comparing accuracy and F1 scores for all models
        """
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='lightcoral')

        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/performance_comparison.png")
        plt.close()

    def _load_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            self.models[model_name],
            token=self.hf_token,
            model_max_length=512  # Set maximum length
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.models[model_name],
            token=self.hf_token
        )
        return pipeline("sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1)

    def evaluate_model(self, model_name, test_texts, true_labels, batch_size=32):
        print(f"\nEvaluating {model_name}...")

        try:
            classifier = self._load_llm(model_name)
            tokenizer = classifier.tokenizer
            max_length = 512
            predictions = []

            # Process in batches
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                batch_texts = [str(text) for text in batch_texts]

                # Tokenize and truncate
                encoded_batch = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt"
                )

                truncated_batch = tokenizer.batch_decode(
                    encoded_batch['input_ids'],
                    skip_special_tokens=True
                )

                results = classifier(truncated_batch)

                # Convert model outputs to labels
                batch_preds = []
                for r in results:
                    if model_name == "roberta-sentiment":
                        # RoBERTa specific mapping
                        if r['label'] == 'LABEL_0':  # negative
                            label = 0
                        elif r['label'] == 'LABEL_1':  # neutral
                            label = 1
                        else:  # LABEL_2, positive
                            label = 2

                    elif model_name == "distilbert-sentiment":
                        # DistilBERT specific mapping
                        # Assuming output is binary (negative/positive)
                        score = float(r['score'])
                        label_text = r['label']

                        if label_text == 'NEGATIVE':
                            if score > 0.8:
                                label = 0  # Strong negative
                            else:
                                label = 1  # Weak negative/neutral
                        else:  # POSITIVE
                            if score > 0.8:
                                label = 2  # Strong positive
                            else:
                                label = 1  # Weak positive/neutral

                    batch_preds.append(label)

                predictions.extend(batch_preds)

            # Calculate metrics with better handling
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)

            # Detailed classification report with zero_division handling
            classification_rep = classification_report(
                true_labels,
                predictions,
                output_dict=True,
                zero_division=1
            )

            # Print detailed distribution of predictions
            print("\nPrediction distribution:")
            pred_dist = pd.Series(predictions).value_counts().sort_index()
            print(pred_dist)

            print("\nTrue label distribution:")
            true_dist = pd.Series(true_labels).value_counts().sort_index()
            print(true_dist)

            results = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'detailed_metrics': classification_rep,
                'predictions': predictions,
                'prediction_distribution': pred_dist.to_dict(),
                'true_distribution': true_dist.to_dict()
            }

            self.results[model_name] = results

            # Save results
            with open(f"{self.run_dir}/{model_name}_results.json", 'w') as f:
                results_to_save = results.copy()
                results_to_save['predictions'] = predictions
                json.dump(results_to_save, f, indent=4)

            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Print per-class metrics
            print("\nPer-class metrics:")
            for label in sorted(set(true_labels)):
                print(f"\nClass {label}:")
                print(f"Precision: {classification_rep[str(label)]['precision']:.4f}")
                print(f"Recall: {classification_rep[str(label)]['recall']:.4f}")
                print(f"F1-score: {classification_rep[str(label)]['f1-score']:.4f}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results[model_name] = {'error': str(e)}

    def plot_confusion_matrices(self, true_labels, predictions_dict):
        """
        Plot confusion matrices for all models with improved visualization
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]

        labels = ['Negative', 'Neutral', 'Positive']

        for ax, (model_name, preds) in zip(axes, predictions_dict.items()):
            cm = confusion_matrix(true_labels, preds)
            # Normalize confusion matrix to percentages
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            sns.heatmap(cm_normalized, annot=True, fmt='.2f', ax=ax, cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            ax.set_title(f'{model_name}\nConfusion Matrix (%)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

        plt.tight_layout()
        plt.savefig(f"{self.run_dir}/confusion_matrices.png")
        plt.close()


def main():
    # Initialize comparison
    comparison = ModelComparison()

    # Prepare dataset using split_data_stratified
    test_texts, test_labels = comparison.prepare_dataset()

    # Store predictions for confusion matrices
    predictions_dict = {}

    # Evaluate each model
    for model_name in comparison.models.keys():
        comparison.evaluate_model(model_name, test_texts, test_labels)
        if model_name in comparison.results and 'predictions' in comparison.results[model_name]:
            predictions_dict[model_name] = comparison.results[model_name]['predictions']

    # Plot confusion matrices
    comparison.plot_confusion_matrices(test_labels, predictions_dict)

    # Plot performance comparison
    comparison.plot_performance_comparison()

    print(f"\nResults and visualizations saved in: {comparison.run_dir}")

if __name__ == "__main__":
    main()
