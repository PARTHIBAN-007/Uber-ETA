# Food Delivery Time Prediction using ML
<img src="https://static.vecteezy.com/system/resources/previews/002/001/999/original/food-delivery-service-vector.jpg" width="400">

## Project objective
To develop a predictive machine learning model that accurately estimates delivery times 
for Uber Eats orders, aiming to improve operational efficiency and customer satisfaction.
The model will help optimize
routing, resource allocation, and provide more accurate delivery estimates by analyzing various factors that impact delivery time.


## UBER ETA Prediction
<div style ="grid:">
<img src ="./assets/ETA Prediction.png" height=40% width=50%>
<img src ="./assets/ETA Prediction2.png" height=50% width=50%>
</div>
<img src ="./assets/ETA predictions.png">


## Repository Structure


1. **data**: Stores different versions of data in distinct folders.
    - **data_after_feature_engineering.csv**: Dataset created after feature engineering
    - **data_cleaned.csv**: Dataset created after data cleaning
    - **raw_data.csv**: Raw dataset
    - **test.csv**: Testing dataset (can be optional)

2. **model**: Directory for saving and loading the model.pkl file.

3. **notebooks**: Python notebooks for cleaning, preprocessing, feature engineering for reference
       - **Data Preprocessing**: Preprocessing the raw data by handling null values, updating datatypes
       - **EDA**: performed exploratory data analysis to conclude some trends in the data 
       - **Feature Engineering**: Extract and create some new features from the Existing Data to provide more information to the Model
       - **Ml Model**: Performed cross-validation and hyperparameter tuning on different ml algorithms to attain an optimal algorithm


5. **src**: Main source code directory with the following subfolders:
    - **preprocessing**: Functionality to preprocess, feature engineering, modeling on a raw dataset
    - **build_model**: Creates and saves the model.pkl file from the preprocessed dataset
    - **predict**: Predictions of the saved model.pkl on new user input

6. **app.py**: Streamlit frontend

7. **Dockerfile**: Configuration for setting up the project in a Docker container.


# Setting up the Project
1. Clone the repository
```
git clone https://github.com/PARTHIBAN-007/Uber-ETA.git
```
2. install all the libraries and dependencies
```
pip install -r requirements.txt
```
3. Run the following command to use the app
```
streamlit run app.py
```

