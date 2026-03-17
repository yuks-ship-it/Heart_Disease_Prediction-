# API Reference

Complete documentation of all modules, functions, and classes in the Heart Disease Prediction project.

## Table of Contents

- [data_loader Module](#data_loader-module)
- [preprocessing Module](#preprocessing-module)
- [train_models Module](#train_models-module)
- [evaluate Module](#evaluate-module)
- [save_results Module](#save_results-module)

---

## `data_loader` Module

**Location:** `src/data_loader.py`  
**Purpose:** Load and retrieve datasets  
**Dependencies:** pandas

### `load_data(path)`

Load a dataset from a CSV file.

**Signature:**
```python
def load_data(path: str) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File path to CSV dataset |

**Returns:**

| Type | Description |
|------|-------------|
| `pandas.DataFrame` | Loaded dataset with all rows and columns |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | If file doesn't exist at specified path |
| `pandas.errors.EmptyDataError` | If CSV file is empty |
| `pandas.errors.ParserError` | If CSV format is invalid |

**Example:**

```python
from src.data_loader import load_data

# Load dataset
df = load_data("data/raw/heart_disease_uci.csv")

# Check dataset shape
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# View first few rows
print(df.head())
```

**Dataset Structure:**

| Column | Type | Description |
|--------|------|-------------|
| age | int | Patient age in years |
| sex | int | Gender (0=female, 1=male) |
| cp | int | Chest pain type (0-3) |
| trestbps | float | Resting blood pressure (mmHg) |
| chol | float | Serum cholesterol (mg/dl) |
| fbs | int | Fasting blood sugar > 120 (0/1) |
| restecg | int | Resting electrocardiogram (0-2) |
| thalach | int | Maximum heart rate achieved |
| exang | int | Exercise-induced angina (0/1) |
| oldpeak | float | ST depression |
| slope | int | Slope of ST segment (0-2) |
| ca | int | Coronary arteries (0-4) |
| thal | int | Thalassemia type (0-3) |
| num | int | Heart disease (0=absent, >0=present) |

---

## `preprocessing` Module

**Location:** `src/preprocessing.py`  
**Purpose:** Clean, transform, and prepare data for modeling  
**Dependencies:** pandas, numpy, scikit-learn

### `preprocess_data(df)`

Complete preprocessing pipeline for the heart disease dataset.

**Signature:**
```python
def preprocess_data(df: pd.DataFrame) -> tuple
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | pandas.DataFrame | Raw heart disease dataset |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| [0] | numpy.ndarray | Training features (scaled) |
| [1] | numpy.ndarray | Test features (scaled) |
| [2] | numpy.ndarray or Series | Training labels |
| [3] | numpy.ndarray or Series | Test labels |

**Return Tuple:** `(X_train, X_test, y_train, y_test)`

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If required column 'num' is missing |
| `ValueError` | If data contains invalid types |

**Processing Pipeline:**

```
1. Binary Conversion (Multi-class → Binary)
   └─ Target: "num" column
      Old: 0, 1, 2, 3, 4
      New: 0, 1 (0=no disease, 1=present)

2. Categorical Encoding
   └─ One-hot encoding for categorical features
   └─ Drops first category to avoid dummy variable trap

3. Feature-Target Separation
   └─ X: All features except target
   └─ y: Target variable ("num")

4. Missing Value Imputation
   └─ Strategy: Mean value
   └─ Handles NaN values in features

5. Feature Scaling
   └─ StandardScaler normalization
   └─ μ = 0, σ = 1

6. Train-Test Split
   └─ 80% training, 20% testing
   └─ Stratified split (preserves class distribution)
   └─ random_state=42 (reproducibility)
```

**Example:**

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Load data
df = load_data("data/raw/heart_disease_uci.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(df)

# Check shapes
print(f"X_train shape: {X_train.shape}")  # (240, n_features)
print(f"X_test shape: {X_test.shape}")    # (60, n_features)
print(f"y_train shape: {y_train.shape}")  # (240,)
print(f"y_test shape: {y_test.shape}")    # (60,)

# Check class distribution
print(f"Training set class distribution:")
print(y_train.value_counts(normalize=True))
```

**Key Implementation Details:**

1. **Binary Conversion:**
   ```python
   df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
   ```

2. **Categorical Encoding:**
   ```python
   df = pd.get_dummies(df, drop_first=True)
   ```

3. **Imputation:**
   ```python
   imputer = SimpleImputer(strategy="mean")
   X_imputed = imputer.fit_transform(X)
   ```

4. **Scaling:**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_imputed)
   ```

5. **Stratified Split:**
   ```python
   train_test_split(
       X_scaled, y,
       test_size=0.2,
       random_state=42,
       stratify=y
   )
   ```

**Performance Considerations:**

- **Time Complexity:** O(n) where n = number of samples
- **Space Complexity:** O(n × m) where m = number of features
- **Typical Runtime:** < 1 second for standard dataset

---

## `train_models` Module

**Location:** `src/train_models.py`  
**Purpose:** Train and serialize machine learning models  
**Dependencies:** scikit-learn, joblib

### `train_all_models(X_train, y_train)`

Train multiple classification models on the provided training data.

**Signature:**
```python
def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X_train` | numpy.ndarray | Training features (m × n) |
| `y_train` | numpy.ndarray | Training labels (m,) |

**Returns:**

| Type | Description |
|------|-------------|
| dict | Dictionary with model names as keys, trained models as values |

**Return Dictionary Keys:**
```python
{
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "decision_tree": DecisionTreeClassifier
}
```

**Side Effects:**

- Creates `models/` directory if it doesn't exist
- Saves 4 pickle files to `models/`:
  - `models/knn.pkl`
  - `models/svm.pkl`
  - `models/random_forest.pkl`
  - `models/decision_tree.pkl`

**Example:**

```python
from src.train_models import train_all_models

# Train models (assuming X_train, y_train are prepared)
trained_models = train_all_models(X_train, y_train)

# Check returned models
print(trained_models.keys())  # dict_keys(['knn', 'svm', 'random_forest', 'decision_tree'])

# Access specific model
knn = trained_models['knn']
predictions = knn.predict(X_test)
```

### Model Details

#### 1. K-Nearest Neighbors (KNN)

**Hyperparameters:**
```python
KNeighborsClassifier(n_neighbors=5)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_neighbors` | 5 | Number of neighbors for voting |

**Characteristics:**
- **Type:** Instance-based learning
- **Training Time:** O(n) - minimal
- **Prediction Time:** O(n×m) - slower
- **Interpretability:** High
- **Scalability:** Low (doesn't scale well with data)

**When to Use:**
- Small to medium datasets
- Fast training required
- Interpretability needed

---

#### 2. Support Vector Machine (SVM)

**Hyperparameters:**
```python
SVC(kernel="rbf", probability=True)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kernel` | "rbf" | Radial basis function (non-linear) |
| `probability` | True | Enable probability estimates |

**Characteristics:**
- **Type:** Boundary-based learning
- **Training Time:** O(n²) to O(n³) - slower
- **Prediction Time:** O(n) - moderate
- **Interpretability:** Low (black box)
- **Scalability:** Medium

**When to Use:**
- Non-linearly separable data
- High-dimensional spaces
- When training time is acceptable

---

#### 3. Random Forest

**Hyperparameters:**
```python
RandomForestClassifier(n_estimators=100)
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 100 | Number of decision trees |

**Characteristics:**
- **Type:** Ensemble learning
- **Training Time:** O(n×log(n)×m) - moderate
- **Prediction Time:** O(n) - fast
- **Interpretability:** Medium (feature importance available)
- **Scalability:** High
- **Advantages:** Handles non-linear relationships, robust to outliers

**When to Use:**
- Large datasets
- Feature importance needed
- Balanced accuracy and performance desired

---

#### 4. Decision Tree

**Hyperparameters:**
```python
DecisionTreeClassifier()  # Default parameters
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `criterion` | "gini" | Split quality measure |
| `max_depth` | None | Maximum tree depth |
| `min_samples_split` | 2 | Minimum samples for split |
| `min_samples_leaf` | 1 | Minimum samples in leaf |

**Characteristics:**
- **Type:** Tree-based learning
- **Training Time:** O(n×log(n)×m) - fast
- **Prediction Time:** O(log(n)) - very fast
- **Interpretability:** Very high (can visualize tree)
- **Scalability:** High
- **Disadvantages:** Prone to overfitting

**When to Use:**
- Interpretability is critical
- Small to medium datasets
- Fast training/prediction needed

---

**Model Comparison Table:**

| Aspect | KNN | SVM | Random Forest | Decision Tree |
|--------|-----|-----|---------------|---------------|
| Training Time | Very Fast | Slow | Moderate | Very Fast |
| Prediction Time | Slow | Moderate | Fast | Very Fast |
| Interpretability | High | Low | Medium | Very High |
| Non-linear Support | Yes | Yes | Yes | Yes |
| Scalability | Low | Medium | High | High |
| Overfitting Risk | Low | Low | Medium | High |

---

## `evaluate` Module

**Location:** `src/evaluate.py`  
**Purpose:** Assess model performance on test data  
**Dependencies:** scikit-learn

### `evaluate_models(models, X_test, y_test)`

Evaluate multiple trained models on test data.

**Signature:**
```python
def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | dict | Dictionary of trained models |
| `X_test` | numpy.ndarray | Test features (n × m) |
| `y_test` | numpy.ndarray | Test labels (n,) |

**Returns:**

| Type | Description |
|------|-------------|
| dict | Model names → accuracy scores (float 0.0-1.0) |

**Return Dictionary Format:**
```python
{
    "knn": 0.8290,
    "svm": 0.8522,
    "random_forest": 0.8411,
    "decision_tree": 0.7923
}
```

**Side Effects:**

- Prints accuracy for each model
- Prints detailed classification reports (precision, recall, F1-score)

**Example:**

```python
from src.evaluate import evaluate_models

# Evaluate models (assuming models dict is populated)
results = evaluate_models(trained_models, X_test, y_test)

# Access results
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

# Find best model
best_model = max(results, key=results.get)
best_accuracy = results[best_model]
print(f"Best Model: {best_model} ({best_accuracy:.4f})")
```

### Output Format

**Console Output:**

```
KNN Accuracy: 0.8291
              precision    recall  f1-score   support

           0       0.85      0.92      0.88       123
           1       0.81      0.68      0.74        76

    accuracy                           0.83       199
   macro avg       0.83      0.80      0.81       199
weighted avg       0.83      0.83      0.83       199
```

### Metrics Explained

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **Definition:** Percentage of correct predictions
- **Range:** 0.0 - 1.0
- **Interpretation:** 0.85 = 85% correct

#### Precision
```
Precision = TP / (TP + FP)
```
- **Definition:** Of predicted positives, how many are actually positive
- **Range:** 0.0 - 1.0
- **Use Case:** When false positives are costly

#### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- **Definition:** Of actual positives, how many are correctly identified
- **Range:** 0.0 - 1.0
- **Use Case:** When false negatives are costly

#### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- **Definition:** Harmonic mean of precision and recall
- **Range:** 0.0 - 1.0
- **Use Case:** Balanced metric for imbalanced classes

#### Support
- **Definition:** Number of actual instances for each class
- **Range:** Integer ≥ 0
- **Interpretation:** How many test samples exist per class

### Confusion Matrix Terms

| Term | Definition |
|------|-----------|
| TP (True Positive) | Predicted disease, actually disease |
| TN (True Negative) | Predicted healthy, actually healthy |
| FP (False Positive) | Predicted disease, actually healthy |
| FN (False Negative) | Predicted healthy, actually disease |

### Interpreting Results

**High Accuracy, Imbalanced Metrics:**
```
Accuracy: 0.92
Precision (Class 0): 0.95
Recall (Class 0): 0.98
Precision (Class 1): 0.60
Recall (Class 1): 0.40
```
👉 Model biased toward majority class 0

**Balanced Metrics:**
```
Precision (Class 0): 0.85
Recall (Class 0): 0.85
Precision (Class 1): 0.80
Recall (Class 1): 0.80
```
👉 Model performs well on both classes

---

## `save_results` Module

**Location:** `src/save_results.py`  
**Purpose:** Export and segment results  
**Dependencies:** pandas, os

### `save_patient_results(df, target_column="num")`

Separate patients by disease status and save to CSV files.

**Signature:**
```python
def save_patient_results(
    df: pd.DataFrame,
    target_column: str = "num"
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | pandas.DataFrame | - | Complete dataset with all patients |
| `target_column` | str | "num" | Name of target column |

**Return Value:**

| Type | Description |
|------|-------------|
| None | Void function (saves files, prints messages) |

**Side Effects:**

- Creates `results/` directory if it doesn't exist
- Generates two CSV files:
  1. `results/heart_disease.csv` - Patients with disease
  2. `results/no_heart_disease.csv` - Patients without disease
- Prints summary statistics to console

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `KeyError` | If target_column doesn't exist |
| `OSError` | If cannot write to results directory |

**Example:**

```python
from src.data_loader import load_data
from src.save_results import save_patient_results

# Load dataset
df = load_data("data/raw/heart_disease_uci.csv")

# Save patient results
save_patient_results(df)

# Output:
# Saved 160 patients with heart disease to 'results/heart_disease.csv'
# Saved 140 patients without heart disease to 'results/no_heart_disease.csv'

# Load results
import pandas as pd
disease_df = pd.read_csv("results/heart_disease.csv")
healthy_df = pd.read_csv("results/no_heart_disease.csv")

print(disease_df.shape)   # (160, 14)
print(healthy_df.shape)   # (140, 14)
```

### Output File Structure

**results/heart_disease.csv:**
```
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
52,1,0,125,212,0,1,168,0,1.0,2,2,3,1
45,0,3,110,264,0,1,132,0,1.2,1,0,3,1
...
```

**results/no_heart_disease.csv:**
```
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num
63,1,3,145,233,1,2,150,0,2.3,0,0,6,0
67,1,0,100,299,0,2,125,1,0.9,1,2,2,0
...
```

### Segmentation Logic

```python
# All patients with disease (num > 0)
heart_disease = df[df[target_column] > 0]

# All patients without disease (num == 0)
no_heart_disease = df[df[target_column] == 0]
```

**Note:** Original dataset may have multi-class target (0, 1, 2, 3, 4)
- This function segments by presence/absence, not severity
- For disease severity, check the `num` column values

---

## Complete Pipeline Example

### Running the Full Pipeline

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_models import train_all_models
from src.evaluate import evaluate_models
from src.save_results import save_patient_results

# 1. Load Data
df = load_data("data/raw/heart_disease_uci.csv")
print(f"Loaded {df.shape[0]} records")

# 2. Save Patient Results
save_patient_results(df)

# 3. Preprocess Data
X_train, X_test, y_train, y_test = preprocess_data(df)
print(f"Training set: {X_train.shape}")

# 4. Train Models
models = train_all_models(X_train, y_train)
print(f"Trained {len(models)} models")

# 5. Evaluate Models
results = evaluate_models(models, X_test, y_test)

# 6. Display Summary
print("\n=== FINAL RESULTS ===")
for model, acc in results.items():
    print(f"{model.upper()}: {acc:.4f}")
```

---

## Exception Handling

### Common Errors and Solutions

**FileNotFoundError: Cannot find data file**
```python
try:
    df = load_data("data/raw/heart_disease_uci.csv")
except FileNotFoundError:
    print("Error: Dataset not found. Check file path.")
```

**KeyError: Missing required column**
```python
try:
    X_train, X_test, y_train, y_test = preprocess_data(df)
except KeyError as e:
    print(f"Error: Missing column {e}")
```

**MemoryError: Dataset too large**
```python
# Load in chunks
for chunk in pd.read_csv("data/raw/heart_disease_uci.csv", chunksize=1000):
    # Process chunk
    pass
```

---

## Performance Benchmarks

**Typical Execution Times (on standard hardware):**

| Operation | Time |
|-----------|------|
| Load data | < 0.1s |
| Preprocess | 0.1-0.2s |
| Train KNN | 0.05s |
| Train SVM | 0.5-1.0s |
| Train Random Forest | 0.2-0.3s |
| Train Decision Tree | 0.05s |
| Evaluate all | 0.1-0.2s |
| **Total** | **~2.0s** |

---

## Related Documentation

- [Project Structure](PROJECT_STRUCTURE.md) - File organization
- [Getting Started](GETTING_STARTED.md) - Setup and usage
- [README](README.md) - Project overview

---

**Version:** 1.0  
**Last Updated:** February 2026
