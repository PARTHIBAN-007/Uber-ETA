import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



class DataPreProcessing:
    def __init__(self):
        self.R = 6234


    def update_columns(self,df):
        df.rename(columns ={"Weatherconditions":"Weather_conditions"},inplace =True)    


    def extract_features(self,df):
        df['Weather_conditions'] =df['Weather_conditions'].apply(lambda x: x.split(' ')[1].strip())
        df['city_code'] = df['Delivery_person_ID'].str.split("RES",expand =True)[0]

    def extract_label_value(self,df):
        df['Time_taken(min)'] = df['Time_taken(min)'].apply(lambda x: int(x.split(' ')[1].strip()))




    def drop_columns(self,df):
        df.drop(['ID','Delivery_person_ID'],axis=1,inplace =True)    

    def convert_nan(self,df):
        df.replace('NaN', float(np.nan), regex=True,inplace=True)   

    def handle_null_values(self,df):
        df['Delivery_person_Age'].fillna(np.random.choice(df['Delivery_person_Age']), inplace=True)    # Names chosen randomly
        df['Weather_conditions'].fillna(np.random.choice(df['Weather_conditions']), inplace=True)      # weather chosen randomly

        mode_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')                      # Filled with the most common value
        mode_cols = ["Road_traffic_density",
                "Multiple_deliveries", "Festival", "City_type"]

        for col in mode_cols:
            df[col] = mode_imp.fit_transform(df[col].to_numpy().reshape(-1,1)).ravel()

        # Mean is affected by extreme values, mode gives highest frequency which is inappropriate
        # So, median is appropriate measure for imputing
        df['Delivery_person_Ratings'].fillna(df['Delivery_person_Ratings'].median(), inplace=True)

        df["Time_Ordered"] = df["Time_Ordered"].fillna(df["Time_Order_picked"])     


    def datatype_update(self,df):
        df['Delivery_person_Age'] = df['Delivery_person_Age'].astype('float64')
        df['Delivery_person_Ratings'] =df['Delivery_person_Ratings'].astype('float64')
        df['Multiple_deliveries'] = df['Multiple_deliveries'].astype('float64')
        df['Order_Date']=pd.to_datetime(df['Order_Date'],format="%d-%m-%Y")    

    def extract_date_features(self,df):
        df['Order_Date'] = df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df['is_weekend'] = df['Order_Date'].dt.day_of_week>4
        df['month_interval'] =df['Order_Date'].apply(lambda x: "start" if x.day <=10
                                                    else ("middle" if x.day <= 20 else "end"))    
        

    def time_difference(self,df):
        df['Time_Ordered'] = pd.to_timedelta(df['Time_Ordered'])
        df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])

        df['Time_Order_picked_formatted'] = df['Order_Date'] + pd.to_timedelta(np.where(df['Time_Order_picked'] < df['Time_Ordered'], 1, 0), unit='D') + df['Time_Order_picked']
        # print(df['Time_Order_picked_formatted'])

        # Calculate formatted order time
        df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Ordered']
        # print(df['Time_Ordered_formatted'])

        # Calculate time difference in minutes
        df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60
        # print(df['order_prepare_time'])

        # Handle null values by filling with the median
        df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
        # print(df['order_prepare_time'])

        # Drop all the time & date related columns
        df.drop(['Time_Ordered', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)    
                
    def deg_to_rad(self,degree):
        return degree*(np.pi/180)
    
    
    def distcalculate(self, lat1, lon1, lat2, lon2):
        """
        Calculates the distance between two latitude-longitude coordinates using the Haversine formula.
        """
        
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        d_lat = self.deg_to_rad(lat2-lat1)
        d_lon = self.deg_to_rad(lon2-lon1)
        a1 = np.sin(d_lat/2)**2 + np.cos(self.deg_to_rad(lat1))
        a2 = np.cos(self.deg_to_rad(lat2)) * np.sin(d_lon/2)**2
        a = a1 * a2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return self.R * c

    def calculate_distance(self, df):
        
        df['distance'] = np.nan

        for i in range(len(df)):
            df.loc[i, "distance"] = self.distcalculate(df.loc[i, 'Restaurant_latitude'],
                                                    df.loc[i, 'Restaurant_longitude'],
                                                    df.loc[i, 'Delivery_location_latitude'],
                                                    df.loc[i, 'Delivery_location_longitude'])
        df["distance"] = df["distance"].astype("int64")

    

    def label_encoding(self,df):
    
        
        categorical_columns = df.select_dtypes(include='object').columns
        label_encoders = {}

        for column in categorical_columns:
            df[column] = df[column].str.strip()
            label_encoder = LabelEncoder()
            label_encoder.fit(df[column])
            df[column] = label_encoder.transform(df[column])
            label_encoders[column] = label_encoder
        return label_encoders

    def data_split(self,x,y):
        x_train ,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state =42)
        return  x_train ,X_test,Y_train,Y_test
     

    def standardize(self,X_train,X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train,X_test,scaler
    
    def cleaning_steps(self, df):
        self.update_columns(df)
        self.extract_features(df)
        self.drop_columns(df)
        self.datatype_update(df)
        self.convert_nan(df)
        self.handle_null_values(df)

    def perform_feature_engineering(self, df):
        self.extract_date_features(df)
        self.time_difference(df)
        self.calculate_distance(df)

    def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))    


        