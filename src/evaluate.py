from sklearn.metrics import accuracy_score, classification_report

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model on test data and print accuracy and classification report.
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n{name.upper()} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

    return results