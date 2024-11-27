I'll help you create a simple comparison script that evaluates multiple models on your sentiment analysis task. Here's a suggested implementation:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from time import time
import json
import os
from datetime import datetime

class ModelComparison:
    def __init__(self, run_dir="comparison_results"):
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

    def _load_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(self.models[model_name])
        model = AutoModelForSequenceClassification.from_pretrained(self.models[model_name])
        return pipeline("sentiment-analysis",
                       model=model,
                       tokenizer=tokenizer,
                       device=0 if self.device == "cuda" else -1)

    def _map_labels(self, score, model_name):
        """Map model outputs to your three classes (negative, neutral, positive)"""
        if model_name == "roberta-sentiment":
            # RoBERTa outputs: 0 (negative), 1 (neutral), 2 (positive)
            return score
        elif model_name == "distilbert-sentiment":
            # DistilBERT outputs: NEGATIVE (0) or POSITIVE (1)
            # You might want to adjust these thresholds
            if score < 0.4:
                return 0  # negative
            elif score > 0.6:
                return 2  # positive
            else:
                return 1  # neutral

    def evaluate_model(self, model_name, test_texts, true_labels, batch_size=32):
        print(f"\nEvaluating {model_name}...")

        start_time = time()

        try:
            classifier = self._load_llm(model_name)
            predictions = []

            # Process in batches
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]
                results = classifier(batch_texts)

                # Convert model outputs to your three classes
                batch_preds = [self._map_labels(r['score'], model_name) for r in results]
                predictions.extend(batch_preds)

            end_time = time()

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            classification_rep = classification_report(true_labels, predictions, output_dict=True)

            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'detailed_metrics': classification_rep,
                'inference_time': end_time - start_time,
                'device': self.device
            }

            # Save results
            self.results[model_name] = results

            # Save to file
            with open(f"{self.run_dir}/{model_name}_results.json", 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Inference time: {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            self.results[model_name] = {'error': str(e)}

    def compare_with_catboost(self, catboost_results_path):
        """Compare LLM results with CatBoost results"""
        try:
            with open(catboost_results_path, 'r') as f:
                catboost_results = json.load(f)

            self.results['catboost'] = catboost_results

            # Create comparison summary
            comparison = {
                'models': {},
                'best_model': {
                    'name': '',
                    'accuracy': 0,
                    'f1_score': 0
                }
            }

            for model_name, results in self.results.items():
                if 'error' not in results:
                    comparison['models'][model_name] = {
                        'accuracy': results['accuracy'],
                        'f1_score': results['f1_score'],
                        'inference_time': results['inference_time']
                    }

                    if results['f1_score'] > comparison['best_model']['f1_score']:
                        comparison['best_model'] = {
                            'name': model_name,
                            'accuracy': results['accuracy'],
                            'f1_score': results['f1_score']
                        }

            # Save comparison results
            with open(f"{self.run_dir}/model_comparison.json", 'w') as f:
                json.dump(comparison, f, indent=4)

            return comparison

        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            return None

def main():
    # Load your test data
    dataset = pd.read_csv('data/cleaned_dataset_processed_balanced.csv')
    test_texts = dataset['cleaned_review'].values
    test_labels = dataset['three_classes'].values  # Assuming you have three classes (0,1,2)

    # Initialize comparison
    comparison = ModelComparison()

    # Evaluate each model
    for model_name in comparison.models.keys():
        comparison.evaluate_model(model_name, test_texts, test_labels)

    # Compare with CatBoost results
    catboost_results_path = 'path/to/your/catboost/results.json'
    final_comparison = comparison.compare_with_catboost(catboost_results_path)

    if final_comparison:
        print("\nFinal Comparison Results:")
        print(json.dumps(final_comparison, indent=2))
        print(f"\nBest Model: {final_comparison['best_model']['name']}")
        print(f"Best F1 Score: {final_comparison['best_model']['f1_score']:.4f}")

if __name__ == "__main__":
    main()
```

To use this script:

1. Make sure you have the required dependencies:
```bash
pip install transformers torch pandas numpy scikit-learn
```

2. The script:
- Creates a unique results directory for each run
- Evaluates multiple LLM models
- Compares results with your CatBoost implementation
- Saves detailed metrics and comparisons to JSON files
- Handles batching for efficient processing
- Works with both CPU and GPU
- Provides timing information

3. You'll need to:
- Adjust the path to your dataset
- Provide the path to your CatBoost results
- Potentially adjust the label mapping in `_map_labels()` to match your classes
- Maybe modify batch_size based on your available memory

4. The script will output:
- Individual model performance metrics
- Comparison between all models
- Best performing model
- Detailed results saved in JSON format

Would you like me to explain any part in more detail or help with specific modifications?
