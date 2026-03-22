import kagglehub
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

def retrain():
    # Download dataset
    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    print("Dataset path:", path)
    
    # Read CSV
    df = pd.read_csv(os.path.join(path, "loan_data.csv"))
    df = df.drop_duplicates()
    df = df.dropna()
    
    target = "loan_status"
    X = df.drop(target, axis=1)
    y = df[target]
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1,
        random_state=42,
        eval_metric="logloss"
    )
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Fitting model...")
    pipeline.fit(X_train, y_train)
    
    # Save model
    joblib.dump(pipeline, "loan_approval_xgboost_pipeline.pkl")
    print("Model retrained and saved to loan_approval_xgboost_pipeline.pkl")
    
if __name__ == "__main__":
    retrain()
