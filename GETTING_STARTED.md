# Getting Started with Heart Disease Prediction

A step-by-step guide to set up, run, and extend the Heart Disease Prediction project.

## 🛠️ Initial Setup

### Step 1: Environment Setup

#### Option A: Using venv (Recommended)

```bash
# Navigate to project directory
cd Heart_disease_prediction

# Create virtual environment
python -m venv predict

# Activate virtual environment
# On Windows (PowerShell):
.\predict\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\predict\Scripts\activate.bat

# On macOS/Linux:
source predict/bin/activate
```

#### Option B: Using conda

```bash
conda create -n heart-disease python=3.9
conda activate heart-disease
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3: Verify Installation

Test that all packages are installed:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn, joblib; print('All packages installed successfully!')"
```

## 🏃 Running the Project

### Basic Execution

Run the complete ML pipeline:

```bash
python main.py
```

**Expected output:**
```
Saved X patients with heart disease to 'results/heart_disease.csv'
Saved Y patients without heart disease to 'results/no_heart_disease.csv'

KNN Accuracy: 0.XXXX
classification_report output...

SVM Accuracy: 0.XXXX
classification_report output...

RANDOM_FOREST Accuracy: 0.XXXX
classification_report output...

DECISION_TREE Accuracy: 0.XXXX
classification_report output...

Final Model Accuracy Comparison:
knn: 0.XXXX
svm: 0.XXXX
random_forest: 0.XXXX
decision_tree: 0.XXXX
```

### Understanding the Pipeline Flow

```
Load Data → Save Patient Results → Preprocess Data → Train Models → Evaluate Models → Output Results
```

1. **Load Data** - Reads `data/raw/heart_disease_uci.csv`
2. **Save Patient Results** - Exports patients to separate CSV files
3. **Preprocess Data** - Cleans and scales features
4. **Train Models** - Trains 4 different classifiers
5. **Evaluate Models** - Tests on holdout test set
6. **Output** - Prints accuracy comparisons

## 📊 Working with the Data

### Dataset Location

```
data/raw/heart_disease_uci.csv
```

### Exploring the Dataset

Open the included Jupyter notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This notebook contains:
- Dataset overview and statistics
- Missing value analysis
- Feature distributions
- Correlations and relationships

### Data Flow

```
Raw Data
    ↓
Binary Conversion (multi-class → binary)
    ↓
Categorical Encoding (one-hot)
    ↓
Missing Value Imputation
    ↓
Feature Scaling (StandardScaler)
    ↓
Train/Test Split (80/20 stratified)
    ↓
Model Training & Evaluation
```

## 🤖 Working with Models

### Model Artifacts

All trained models are saved in `models/` directory:

```
models/
├── knn.pkl
├── svm.pkl
├── random_forest.pkl
└── decision_tree.pkl
```

### Loading a Trained Model

```python
import joblib

# Load a specific model
knn_model = joblib.load('models/knn.pkl')

# Make predictions
predictions = knn_model.predict(X_test)
```

### Training Custom Models

Edit `src/train_models.py` to modify model parameters:

```python
models = {
    "knn": KNeighborsClassifier(n_neighbors=7),  # Change n_neighbors
    "svm": SVC(kernel="linear", probability=True),  # Change kernel
    "random_forest": RandomForestClassifier(n_estimators=200),  # Change n_estimators
    "decision_tree": DecisionTreeClassifier(max_depth=10)  # Add max_depth
}
```

Then run:
```bash
python main.py
```

## 📈 Interpreting Results

### Accuracy Metric

Shows the percentage of correct predictions:

```
KNN Accuracy: 0.8290 = 82.90% correct predictions
```

### Classification Report

```
              precision    recall  f1-score   support

           0       0.85      0.92      0.88       123
           1       0.81      0.68      0.74        76

    accuracy                           0.83       199
   macro avg       0.83      0.80      0.81       199
weighted avg       0.83      0.83      0.83       199
```

**Interpreting:**
- **Precision:** Of predicted positives, how many were actually positive
- **Recall:** Of actual positives, how many were correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of actual instances per class

## 📂 Output Files

### Results Directory

After running `main.py`, the `results/` directory contains:

```
results/
├── heart_disease.csv          # Patients with heart disease
└── no_heart_disease.csv       # Patients without heart disease
```

Each CSV contains the original features for segmented patients.

### Accessing Results

```python
import pandas as pd

# Load patient results
disease_patients = pd.read_csv('results/heart_disease.csv')
healthy_patients = pd.read_csv('results/no_heart_disease.csv')

print(f"Disease patients: {len(disease_patients)}")
print(f"Healthy patients: {len(healthy_patients)}")
```

## 🔧 Customizing the Pipeline

### Modifying Preprocessing

Edit `src/preprocessing.py`:

```python
# Change test size
train_test_split(X_scaled, y, test_size=0.25, ...)  # 75/25 split

# Change scaling method
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Change imputation strategy
imputer = SimpleImputer(strategy="median")  # Use median instead of mean
```

### Adding New Models

1. Edit `src/train_models.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

def train_all_models(X_train, y_train):
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "decision_tree": DecisionTreeClassifier(),
        "gradient_boosting": GradientBoostingClassifier()  # NEW
    }
    # ... rest of function
```

2. Run the pipeline:
```bash
python main.py
```

## 🐛 Debugging and Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Ensure you're in the correct directory
cd c:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\Heart_disease_prediction

# Verify virtual environment is activated
.\predict\Scripts\Activate.ps1  # Activate if needed
```

### Issue: "FileNotFoundError: data/raw/heart_disease_uci.csv"

**Solution:**
```bash
# Check that the file exists
dir data\raw\

# If missing, verify the path and filename
```

### Issue: "No module named 'pandas'" or other packages

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Or install individual package
pip install pandas
```

### Issue: Models not saving to `models/` directory

**Solution:**
```python
# Check write permissions and directory existence
import os
os.makedirs("models", exist_ok=True)

# Verify path
print(os.path.abspath("models"))
```

### Issue: Different results on multiple runs

**Note:** This is normal! It's due to:
- Random initialization in some models (SVM, Random Forest)
- Data splitting randomness

**Solution:** Use fixed `random_state` in preprocessing.py (already implemented as `random_state=42`)

## 🚀 Advanced Usage

### Running Individual Components

```python
from src.data_loader import load_data
from src.preprocessing import preprocess_data

# Just load and preprocess
df = load_data("data/raw/heart_disease_uci.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

# Use in your own pipeline
```

### Batch Processing Multiple Datasets

```python
import glob

for csv_file in glob.glob("data/raw/*.csv"):
    print(f"Processing {csv_file}")
    df = load_data(csv_file)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    # ... rest of processing
```

### Performance Profiling

```bash
# Time the main.py execution
time python main.py  # macOS/Linux
Measure-Command { python main.py }  # PowerShell
```

## 📚 Additional Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/)

## ✅ Checklist for First Run

- [ ] Virtual environment created and activated
- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Dataset exists at `data/raw/heart_disease_uci.csv`
- [ ] Successfully ran `python main.py`
- [ ] Reviewed output accuracy metrics
- [ ] Checked `models/` directory for .pkl files
- [ ] Reviewed `results/` CSV files

---

**Next Steps:** Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed file descriptions or [API_REFERENCE.md](API_REFERENCE.md) for function documentation.
