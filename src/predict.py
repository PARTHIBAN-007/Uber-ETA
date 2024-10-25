from preprocessing import DataPreProcessing
import pickle
def predict(x):
    with open("Model/model.pkl","wb") as f:
        print("Model Imported")
        model , label_encoders , scaler = pickle.load(f)


    dp = DataPreProcessing()
    dp.cleaning_steps(X)
    dp.perform_feature_engineering(X)

    for column, label_encoder in label_encoders.items():
        X[column] = label_encoder.transform(X[column])

    X = scaler.transform(X)  
    pred = model.predict(X) 
    return pred




