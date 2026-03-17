# Project Structure

Detailed guide to the Heart Disease Prediction project's file organization and directory layout.

## Directory Tree

```
Heart_disease_prediction/
│
├── 📄 main.py                          Main pipeline orchestrator
├── 📄 requirements.txt                 Python package dependencies
├── 📄 README.md                        Project overview
├── 📄 GETTING_STARTED.md              Setup and usage guide
├── 📄 PROJECT_STRUCTURE.md            This file
├── 📄 API_REFERENCE.md                Module documentation
│
├── 📁 data/                           Data directory
│   ├── 📁 raw/                        Original datasets
│   │   └── 📊 heart_disease_uci.csv   UCI Heart Disease dataset
│   └── 📁 processed/                  Processed data (future use)
│
├── 📁 src/                            Source code modules
│   ├── 📄 __init__.py                 Package initialization
│   ├── 📄 data_loader.py              Data loading utilities
│   ├── 📄 preprocessing.py            Data preprocessing pipeline
│   ├── 📄 train_models.py             Model training functions
│   ├── 📄 evaluate.py                 Model evaluation metrics
│   ├── 📄 save_results.py             Results export utilities
│   └── 📁 __pycache__/                Python cache (auto-generated)
│
├── 📁 models/                         Trained model artifacts
│   ├── 🤖 knn.pkl                     K-Nearest Neighbors model
│   ├── 🤖 svm.pkl                     Support Vector Machine model
│   ├── 🤖 random_forest.pkl           Random Forest Classifier model
│   └── 🤖 decision_tree.pkl           Decision Tree Classifier model
│
├── 📁 notebooks/                      Jupyter notebooks
│   └── 📔 01_eda.ipynb                Exploratory Data Analysis
│
├── 📁 results/                        Output results
│   ├── 📊 heart_disease.csv           Patients with heart disease
│   └── 📊 no_heart_disease.csv        Patients without heart disease
│
├── 📁 reports/                        Reports and documentation
│   └── (To be populated with analysis reports)
│
└── 📁 predict/                        Virtual environment
    ├── pyvenv.cfg                     Environment configuration
    ├── 📁 Scripts/                    Executable scripts
    ├── 📁 Lib/                        Python packages
    └── 📁 Include/                    C header files
```

## File Descriptions

### Root Level Files

#### `main.py`
**Purpose:** Main orchestration script  
**Type:** Python executable  
**Description:** Coordinates the complete machine learning pipeline:
1. Loads dataset from CSV
2. Saves patient segmentation results
3. Preprocesses features
4. Trains all models
5. Evaluates performance
6. Displays accuracy comparison

**Usage:**
```bash
python main.py
```

**Output:** Trained models, segmented results, accuracy metrics

---

#### `requirements.txt`
**Purpose:** Dependency specification  
**Type:** Text file  
**Description:** Lists all required Python packages and versions

**Contents:**
```
pandas          - Data manipulation and analysis
numpy           - Numerical computing
scikit-learn    - Machine learning algorithms
matplotlib      - Data visualization
seaborn         - Statistical data visualization
joblib          - Model serialization/deserialization
```

**Usage:**
```bash
pip install -r requirements.txt
```

---

#### `README.md`
**Purpose:** Project overview  
**Type:** Markdown  
**Description:** High-level project summary, quick start guide, and feature list

---

#### `GETTING_STARTED.md`
**Purpose:** Setup and usage instructions  
**Type:** Markdown  
**Description:** Step-by-step guide for:
- Environment setup
- Installation
- Running the pipeline
- Customization
- Troubleshooting

---

#### `PROJECT_STRUCTURE.md`
**Purpose:** Directory and file documentation  
**Type:** Markdown  
**Description:** This file - explains project organization

---

#### `API_REFERENCE.md`
**Purpose:** Function-level documentation  
**Type:** Markdown  
**Description:** Detailed documentation of all modules and functions

---

## Directory Details

### `data/` - Data Storage

**Purpose:** Store datasets  
**Subdirectories:**

#### `data/raw/`
- **Purpose:** Original, unmodified datasets
- **Files:**
  - `heart_disease_uci.csv` - UCI Heart Disease dataset
    - **Rows:** ~300 patient records
    - **Columns:** 13 medical features + 1 target column
    - **Format:** CSV (comma-separated values)
    - **No modification** to raw data - preprocessing happens in memory

#### `data/processed/`
- **Purpose:** Cleaned, processed datasets (optional expansion)
- **Current:** Empty (reserved for future use)
- **Potential Use:** Store preprocessed features, train/test splits

---

### `src/` - Source Code

**Purpose:** All reusable Python modules  
**Type:** Python package  

#### `src/__init__.py`
- **Purpose:** Package initialization file
- **Description:** Makes `src` a Python package (can be empty or contain imports)

---

#### `src/data_loader.py`
**Purpose:** Data loading utilities  
**Key Function:** `load_data(path)`

```python
def load_data(path):
    """
    Load the dataset from a CSV file
    
    Parameters:
        path (str): Path to CSV file
    
    Returns:
        pandas.DataFrame: Loaded dataset
    
    Example:
        df = load_data("data/raw/heart_disease_uci.csv")
    """
```

**Dependencies:** pandas  
**Used in:** main.py

---

#### `src/preprocessing.py`
**Purpose:** Data preprocessing pipeline  
**Key Function:** `preprocess_data(df)`

**Processing Steps:**
1. **Binary Classification Conversion**
   - Converts multi-class target to binary (0 or 1)
   - Logic: `df["num"] > 0` → disease present

2. **Categorical Encoding**
   - One-hot encoding for categorical features
   - Drops first category to avoid multicollinearity

3. **Feature Extraction**
   - Separates features (X) and target (y)

4. **Missing Value Imputation**
   - Strategy: Mean value
   - Handles NaN values in features

5. **Feature Scaling**
   - StandardScaler normalization
   - Brings all features to similar scale

6. **Train-Test Split**
   - 80% training, 20% testing
   - Stratified to preserve class distribution
   - Random state = 42 for reproducibility

```python
def preprocess_data(df):
    """
    Preprocess heart disease dataset
    
    Parameters:
        df (pandas.DataFrame): Raw dataset
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
```

**Dependencies:** pandas, numpy, scikit-learn  
**Used in:** main.py

---

#### `src/train_models.py`
**Purpose:** Model training and persistence  
**Key Function:** `train_all_models(X_train, y_train)`

**Models Trained:**
1. **K-Nearest Neighbors (KNN)**
   - Parameters: `n_neighbors=5`
   - Algorithm: Instance-based learning

2. **Support Vector Machine (SVM)**
   - Parameters: `kernel="rbf"`, `probability=True`
   - Algorithm: Boundary-based classification

3. **Random Forest**
   - Parameters: `n_estimators=100`
   - Algorithm: Ensemble of decision trees

4. **Decision Tree**
   - Parameters: Default
   - Algorithm: Tree-based recursive partitioning

**Output:** Models saved as .pkl files in `models/` directory

```python
def train_all_models(X_train, y_train):
    """
    Train multiple classification models
    
    Parameters:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        dict: Dictionary of trained models
    """
```

**Dependencies:** scikit-learn, joblib  
**Used in:** main.py

---

#### `src/evaluate.py`
**Purpose:** Model evaluation and metrics  
**Key Function:** `evaluate_models(models, X_test, y_test)`

**Metrics Calculated:**
- Accuracy: Overall correctness percentage
- Precision: True positives / (true positives + false positives)
- Recall: True positives / (true positives + false negatives)
- F1-Score: Harmonic mean of precision and recall

**Output:** Prints classification reports and returns accuracies

```python
def evaluate_models(models, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Model names → accuracy scores
    """
```

**Dependencies:** scikit-learn  
**Used in:** main.py

---

#### `src/save_results.py`
**Purpose:** Results export and patient segmentation  
**Key Function:** `save_patient_results(df, target_column="num")`

**Functionality:**
- Separates patients with disease from healthy patients
- Saves to separate CSV files
- Prints statistics on segmented data

```python
def save_patient_results(df, target_column="num"):
    """
    Separate and save patients by disease status
    
    Parameters:
        df (pandas.DataFrame): Full dataset
        target_column (str): Target column name
    
    Output Files:
        results/heart_disease.csv
        results/no_heart_disease.csv
    """
```

**Dependencies:** pandas  
**Used in:** main.py

---

### `models/` - Trained Models

**Purpose:** Store serialized trained models  
**Format:** Pickle (.pkl) files  
**Auto-created:** Yes (when main.py runs)

#### Model Files

| File | Model | Purpose |
|------|-------|---------|
| `knn.pkl` | K-Nearest Neighbors | Fast, instance-based prediction |
| `svm.pkl` | Support Vector Machine | Non-linear boundary modeling |
| `random_forest.pkl` | Random Forest | Ensemble predictions |
| `decision_tree.pkl` | Decision Tree | Interpretable predictions |

**Loading Models:**
```python
import joblib
model = joblib.load('models/knn.pkl')
predictions = model.predict(X_new)
```

---

### `notebooks/` - Analysis Notebooks

**Purpose:** Interactive data exploration and analysis  
**Format:** Jupyter Notebook (.ipynb)

#### `notebooks/01_eda.ipynb`
**Purpose:** Exploratory Data Analysis  
**Contents:**
- Dataset overview and statistics
- Missing value analysis
- Feature distributions
- Correlation analysis
- Visualizations

**Usage:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

### `results/` - Output Results

**Purpose:** Store segmented patient data  
**Auto-created:** Yes (when main.py runs)

#### `results/heart_disease.csv`
- **Purpose:** Patients with heart disease
- **Rows:** Count varies per dataset
- **Columns:** All original features + disease indicator
- **Generated by:** `save_results.py`

#### `results/no_heart_disease.csv`
- **Purpose:** Patients without heart disease
- **Rows:** Count varies per dataset
- **Columns:** All original features + disease indicator
- **Generated by:** `save_results.py`

---

### `reports/` - Analysis Reports

**Purpose:** Store generated reports and analysis  
**Current Status:** Empty (reserved for future reports)

**Potential Contents:**
- Model performance summaries
- Feature importance analysis
- Statistical reports
- Visualizations and charts

---

### `predict/` - Virtual Environment

**Purpose:** Isolated Python environment  
**Type:** Virtual environment directory  
**Auto-created:** Yes (via `python -m venv predict`)

**Key Subdirectories:**
- `Scripts/` - Executable scripts (Activate, pip, etc.)
- `Lib/` - Installed Python packages
- `Include/` - C header files for compiled extensions

**DO NOT:** Commit this directory to version control

---

## Data Flow Diagram

```
data/raw/
  │
  ├─→ src/data_loader.py
       │
       ├─→ src/save_results.py (saves to results/)
       │
       ├─→ src/preprocessing.py
            │
            ├─→ src/train_models.py (saves to models/)
                 │
                 ├─→ src/evaluate.py
                      │
                      └─→ main.py (prints metrics)
```

## File Dependencies Map

```
main.py
  ├── src/data_loader.py
  │   └── pandas
  ├── src/save_results.py
  │   └── pandas, os
  ├── src/preprocessing.py
  │   ├── pandas, numpy
  │   └── scikit-learn
  ├── src/train_models.py
  │   ├── scikit-learn
  │   └── joblib, os
  └── src/evaluate.py
      └── scikit-learn
```

## Size Information

| Directory | Typical Size | Notes |
|-----------|------------|-------|
| `data/raw/` | 50-100 KB | Depends on dataset |
| `src/` | < 10 KB | Source code |
| `models/` | 100 KB - 1 MB | Depends on model complexity |
| `results/` | 50-100 KB | Depends on dataset |
| `predict/` | 200+ MB | Virtual environment (exclude from git) |

---

## Summary

```
src/           → Reusable code modules
data/          → Input datasets
models/        → Output trained models
results/       → Output segmented data
notebooks/     → Interactive analysis
reports/       → Analysis documentation
main.py        → Pipeline orchestration
```

---

**See Also:**
- [API_REFERENCE.md](API_REFERENCE.md) for function documentation
- [GETTING_STARTED.md](GETTING_STARTED.md) for usage instructions
