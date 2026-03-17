from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train_models import train_all_models
from src.evaluate import evaluate_models
from src.save_results import save_patient_results

# Load dataset
df = load_data("data/raw/heart_disease_uci.csv")

# Save patients into separate CSVs
save_patient_results(df)

# Preprocess dataset for modeling
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train models
models = train_all_models(X_train, y_train)

# Evaluate models
results = evaluate_models(models, X_test, y_test)

# Print final accuracy comparison
print("\nFinal Model Accuracy Comparison:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")   