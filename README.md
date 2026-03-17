# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease in patients using multiple classification algorithms.

## 📋 Overview

This project utilizes the UCI Heart Disease dataset to build and evaluate multiple machine learning models for binary classification. It determines whether a patient has heart disease based on various medical and demographic features.

**Models Implemented:**
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest Classifier
- Decision Tree Classifier

## 🎯 Objectives

- Load and explore the UCI Heart Disease dataset
- Preprocess data including handling missing values and feature scaling
- Train multiple classification models
- Evaluate model performance and compare accuracy
- Segment patients into disease vs. non-disease categories

## 📁 Project Structure

```
Heart_disease_prediction/
├── main.py                          # Main pipeline script
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
├── GETTING_STARTED.md              # Setup instructions
├── PROJECT_STRUCTURE.md            # Detailed file structure
├── API_REFERENCE.md                # Module documentation
│
├── data/
│   ├── raw/
│   │   └── heart_disease_uci.csv   # Original dataset
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Dataset loading utilities
│   ├── preprocessing.py            # Data preprocessing pipeline
│   ├── train_models.py             # Model training functions
│   ├── evaluate.py                 # Model evaluation metrics
│   └── save_results.py             # Results export utilities
│
├── models/
│   ├── knn.pkl                     # Trained KNN model
│   ├── svm.pkl                     # Trained SVM model
│   ├── random_forest.pkl           # Trained Random Forest model
│   └── decision_tree.pkl           # Trained Decision Tree model
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
│
├── results/
│   ├── heart_disease.csv           # Patients with heart disease
│   └── no_heart_disease.csv        # Patients without heart disease
│
└── reports/                        # Analysis reports and results
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone/Navigate to the project directory:**
```bash
cd Heart_disease_prediction
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv predict
.\predict\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv predict
source predict/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute the complete machine learning pipeline:
```bash
python main.py
```

This will:
1. Load the heart disease dataset
2. Save patient results (disease vs. non-disease)
3. Preprocess the data
4. Train all four models
5. Evaluate and display model accuracies

## 📊 Dataset

**Source:** UCI Heart Disease Dataset  
**Features:** 13 medical and demographic attributes
**Target:** Binary classification (0 = No disease, 1 = Disease present)

### Data Processing

The preprocessing pipeline includes:
- Binary classification conversion (multi-class → binary)
- Categorical variable encoding (one-hot encoding)
- Missing value imputation (mean strategy)
- Feature scaling (StandardScaler)
- Stratified train-test split (80/20)

## 🤖 Models

### K-Nearest Neighbors (KNN)
- **Parameters:** n_neighbors=5
- **Strengths:** Simple, fast inference
- **Weaknesses:** Sensitive to feature scaling, slow training

### Support Vector Machine (SVM)
- **Parameters:** kernel="rbf", probability=True
- **Strengths:** Effective in high-dimensional spaces
- **Weaknesses:** Longer training time

### Random Forest
- **Parameters:** n_estimators=100
- **Strengths:** Handles non-linear relationships, feature importance
- **Weaknesses:** Prone to overfitting

### Decision Tree
- **Parameters:** Default sklearn parameters
- **Strengths:** Interpretable, fast
- **Weaknesses:** High variance, prone to overfitting

## 📈 Results

Models are evaluated using:
- **Accuracy:** Overall correctness percentage
- **Classification Report:** Precision, Recall, F1-Score

Output example:
```
KNN Accuracy: 0.8250
SVM Accuracy: 0.8290
RANDOM_FOREST Accuracy: 0.8410
DECISION_TREE Accuracy: 0.7920
```

## 📦 Output Files

### Trained Models
Models are saved in the `models/` directory as `.pkl` files:
- `knn.pkl` - Trained KNN classifier
- `svm.pkl` - Trained SVM classifier
- `random_forest.pkl` - Trained Random Forest
- `decision_tree.pkl` - Trained Decision Tree

### Results
Patient segmentation saved in `results/`:
- `heart_disease.csv` - Patients with heart disease
- `no_heart_disease.csv` - Patients without heart disease

## 📚 Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md) - Detailed setup and usage guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - File organization and descriptions
- [API_REFERENCE.md](API_REFERENCE.md) - Module and function documentation

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical operations |
| scikit-learn | Latest | ML algorithms |
| matplotlib | Latest | Visualization |
| seaborn | Latest | Statistical plots |
| joblib | Latest | Model serialization |

See [requirements.txt](requirements.txt) for exact versions.

## 📝 License

This project uses publicly available data from the UCI Machine Learning Repository.

## 🤝 Contributing

For improvements or bug fixes, please refer to [GETTING_STARTED.md](GETTING_STARTED.md#debugging-and-troubleshooting).

## ❓ Troubleshooting

**Issue:** ModuleNotFoundError when running main.py
- **Solution:** Ensure virtual environment is activated and dependencies are installed

**Issue:** Models not saving
- **Solution:** Check that the `models/` directory exists and has write permissions

**Issue:** Data not loading
- **Solution:** Verify that `data/raw/heart_disease_uci.csv` exists and is accessible

For more detailed troubleshooting, see [GETTING_STARTED.md](GETTING_STARTED.md#debugging-and-troubleshooting).

## 📧 Contact & Support

For questions or issues, please review the documentation files or check the project's main source files.

---

**Last Updated:** February 2026
**Project Status:** Active
