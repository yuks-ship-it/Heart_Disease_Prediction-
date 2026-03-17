from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def preprocess_data(df):

    # 🔥 Convert multi-class target to binary
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("num", axis=1)
    y = df["num"]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Stratified split (VERY IMPORTANT)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test