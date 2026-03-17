from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_all_models(X_train, y_train):
    """
    Train KNN, SVM, Random Forest, and Decision Tree models and save them as .pkl files.
    """
    models = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "decision_tree": DecisionTreeClassifier()
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}.pkl")

    return models