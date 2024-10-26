from .preprocessing import DataPreProcessing
import pickle
import xgboost as xgb
import pandas as pd

def predict(X):
    # Load the scaler and label encoders from the pickle file
    with open("./Model/model.pickle", 'rb') as f:
        print("Loading scaler and label encoders")
        label_encoders, scaler = pickle.load(f)
    
    # Load the model using XGBoost's load_model method
    print("Loading model")
    model = xgb.Booster()
    model.load_model("./Model/model.json")
    
    # Perform data cleaning and feature engineering
    dataprocess = DataPreProcessing()
    dataprocess.cleaning_steps(X)               # Perform Cleaning
    dataprocess.perform_feature_engineering(X)  # Perform Feature Engineering

    # Apply label encoding
    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    # Standardize the features
    X = scaler.transform(X)  

    # Make predictions with XGBoost model
    dmatrix = xgb.DMatrix(X)  # Convert to DMatrix for XGBoost
    pred = model.predict(dmatrix)  # Predict time of delivery
    
    return pred
