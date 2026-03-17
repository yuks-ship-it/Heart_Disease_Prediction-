import os
import pandas as pd

def save_patient_results(df, target_column="num"):
    """
    Separates patients with and without heart disease into two CSV files.
    
    Parameters:
        df: pandas.DataFrame
            The full heart disease dataset
        target_column: str
            The column indicating heart disease (0 = no, 1 = yes)
    """
    os.makedirs("results", exist_ok=True)

    # Patients with heart disease
    heart_disease = df[df[target_column] == 1]
    heart_disease.to_csv("results/heart_disease.csv", index=False)

    # Patients without heart disease
    no_heart_disease = df[df[target_column] == 0]
    no_heart_disease.to_csv("results/no_heart_disease.csv", index=False)

    print(f"Saved {len(heart_disease)} patients with heart disease to 'results/heart_disease.csv'")
    print(f"Saved {len(no_heart_disease)} patients without heart disease to 'results/no_heart_disease.csv'")