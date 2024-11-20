Looking at your distribution, you have a significant class imbalance problem:
- Rating 5: 9054 samples (44.2%)
- Rating 4: 6039 samples (29.5%)
- Rating 3: 2184 samples (10.7%)
- Rating 2: 1793 samples (8.8%)
- Rating 1: 1421 samples (6.9%)

Here are several techniques you could use to handle this imbalance:

1. **Resampling Techniques**:
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Option 1: SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Option 2: Combination of over and under sampling
pipeline = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42))
])
X_res, y_res = pipeline.fit_resample(X, y)
```

2. **Class Weights**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

model = LogisticRegression(
    multi_class='multinomial',
    class_weight=class_weight_dict,
    max_iter=1000
)
```

3. **Stratification in Train-Test Split**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # This ensures proportional representation in train/test
)
```

4. **Consider Binary Classification**:
You could transform this into binary classification problems:
- Positive (4-5) vs Negative (1-2-3)
- Or create separate binary classifiers for each rating

My recommendation would be to try this sequence:
1. First try class weights as it's the simplest solution
2. If that's not enough, try SMOTE
3. Always use stratification in your train-test split
4. If still not satisfied, consider the binary approach

Would you like me to show you the complete implementation of any of these approaches?
