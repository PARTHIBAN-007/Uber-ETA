from preprocessing import *
import pandas as pd
train_df = pd.read_csv("./Data/rawdata.csv")


dp = DataPreProcessing()


dp.cleaning_steps(train_df)
dp.perform_feature_engineering(train_df)
dp.extract_label_value(train_df)



X = train_df.drop('Time_taken(min)', axis=1)   
Y = train_df['Time_taken(min)']
label_encoders = dp.label_encoding(X)
X_train,X_test,Y_train,Y_test =   dp.data_split(X,Y)
X_train , X_test ,Scale = dp.standardize(X_train,X_test)


model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, Y_train)

# Evaluate Model
y_pred = model.predict(X_test)
dp.evaluate_model(Y_test, y_pred)

# Create model.pkl and Save Model
with open("./Model/model.pickle", 'wb') as f:
    pickle.dump((model, label_encoders, Scale), f)
print("Model pickle saved to model folder")