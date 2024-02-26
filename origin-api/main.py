import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt, seaborn as sns
from DataType import DataType, Trend, Year

# df = pd.read_csv('./rental_final.csv')
# df.head(4)

# df.dtypes

# use_columns2 = ['floor', 'room_size', 'unit_type_id',
#        'district_id','sum_near', 'Airport Rail Link', 'BTS Silom Line', 'BTS Sukhumvit Line',
#        'District', 'Gold Line', 'Government', 'Hospital',
#        'International School', 'Market', 'MRT Blue line', 'Popular Areas',
#        'Road', 'School', 'Shopping Mall', 'Soi', 'Super Market',
#        'University/College',]

# from sklearn.metrics import accuracy_score
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# # โหลดข้อมูล

# # เลือกคอลัมน์ที่เป็น feature (X) และ target (y)
# X = df[use_columns2]
# y = df['rental_group_id']

# classes = y.unique()
# # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
# # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # สร้างและฝึกโมเดล XGBoost Classifier โดยใช้คลาสที่มีอยู่ในชุดข้อมูลเป้าหมาย
# xgb_classifier = xgb.XGBClassifier(classes=classes)
# xgb_classifier.fit(X_train, y_train)

# # ทำนายค่า target สำหรับชุดทดสอบ
# y_pred = xgb_classifier.predict(X_test)

# # คำนวณความแม่นยำของโมเดล
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

import pickle

pickle_in = open("rental_module.pkl", "rb")
classifier = pickle.load(pickle_in)

pickle_in2 = open("sale_module.pkl", "rb")
classifier2 = pickle.load(pickle_in2)

pickle_in3 = open("priceIndex_module.pkl", "rb")
classifier3 = pickle.load(pickle_in3)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"accuracy": "Hi, Welcome to Rental Price Prediction"}

@app.post("/predict")
def predict(data: DataType):
    data = data.dict()

    # Extracting data from the request
    features = [
        data['floor'], data['room_size'], data['unit_type_id'], data['district_id'],
        data['Airport_Rail_Link'], data['BTS_Silom_Line'],
        data['BTS_Sukhumvit_Line'], data['District'], data['Gold_Line'],
        data['Government'], data['Hospital'], data['International_School'],
        data['Market'], data['MRT_Blue_line'], data['Popular_Areas'], data['Road'],
        data['School'], data['Shopping_Mall'], data['Soi'], data['Super_Market'],
        data['University_College']
    ]

    # Converting features to a NumPy array
    features_array = np.array(features).reshape(1, -1)
    features_array_sales = np.array(features).reshape(1, -1)

    # Predicting with the classifier
    prediction = classifier.predict(features_array)
    prediction_sale = classifier2.predict(features_array_sales)

    # Converting NumPy int64 to regular Python int
    prediction = prediction.item()
    prediction_sale = prediction_sale.item()

    match prediction:
       case 0:
              prediction = 2000
       case 1:
              prediction = 4000
       case 2:
              prediction = 6000
       case 3:
              prediction = 8000
       case 4:
              prediction = 10000
       case 5:
              prediction = 15000
       case 6:
              prediction = 20000
       case 7:
              prediction = 25000
       case 8:
              prediction = 30000
       case 9:
              prediction = 35000
       case 10:
              prediction = 40000
       case 11:
              prediction = 45000
       case 12:
              prediction = 50000
       case 13:
              prediction = 55000
       case 14:
              prediction = 60000
       case 15:
              prediction = 65000
       case 16:
              prediction = 70000
       case 17:
              prediction = 75000
       case 18:
              prediction = 80000
       case 19:
              prediction = 85000
       case 20:
              prediction = 90000
       case 21:
              prediction = 95000
       case 22:
              prediction = 100000
       case 23:
              prediction = 150000
       case 24:
              prediction = 200000
       case 25:
              prediction = 250000
       case 26:
              prediction = 300000
       case 27:
              prediction = 350000
       case 28:
              prediction = 400000
       case 29:
              prediction = 500000    
       
    match prediction_sale:
       case 0:
              prediction_sale = 500000
       case 1:
              prediction_sale = 1000000
       case 2:
              prediction_sale = 1500000
       case 3:
              prediction_sale = 2000000
       case 4:
              prediction_sale = 2500000
       case 5:
              prediction_sale = 3000000
       case 6:
              prediction_sale = 3500000
       case 7:
              prediction_sale = 4000000
       case 8:
              prediction_sale = 4500000
       case 9:
              prediction_sale = 5000000
       case 10:
              prediction_sale = 5500000
       case 11:
              prediction_sale = 6000000
       case 12:
              prediction_sale = 6500000
       case 13:
              prediction_sale = 7000000
       case 14:
              prediction_sale = 7500000
       case 15:
              prediction_sale = 8000000
       case 16:
              prediction_sale = 8500000
       case 17:
              prediction_sale = 9000000
       case 18:
              prediction_sale = 9500000
       case 19:
              prediction_sale = 10000000
       case 20:
              prediction_sale = 15000000
       case 21:
              prediction_sale = 20000000
       case 22:
              prediction_sale = 25000000
       case 23:
              prediction_sale = 30000000
       case 24:
              prediction_sale = 35000000
       case 25:
              prediction_sale = 40000000
       case 26:
              prediction_sale = 45000000
       case 27:
              prediction_sale = 50000000
       case 28:
              prediction_sale = 65000000
       case 29:
              prediction_sale = 75000000
       case 30:
              prediction_sale = 80000000


    return {"prediction": prediction, "sale": prediction_sale}

# @app.post("/trend")
# def trend(data: Trend):
#        data = data.dict()
#        # print(type(data["district_id"]))
#        # print(data["rental_group"])

#        df = pd.read_csv('./final.csv')    
#        rental_result = data["rental_group"]
#        df2 = df[df['rental_group'] == rental_result]
#        # df2 = df2[df2['district_id'] == data["district_id"]]


#        # print(df2)
#        # return {"data" : df2.to_dict(orient="records")}
@app.post("/trend")
def trend(data: Trend):
       data = data.dict()
       print(data)
       df = pd.read_csv('./final.csv')    
       rental_result = data["rental_group"]
       df2 = df[df['rental_group'] == rental_result]
       df2 = df2[df2['district_id'] == data["district_id"]]
    
       df2['date'] = pd.to_datetime(df2['date'])
       convert_date_rental = df2['date'].dt.date.value_counts()
       sort_date_rental = convert_date_rental.sort_index()
    # Replace NaN values with a placeholder
       sort_date_rental = sort_date_rental.fillna('NaN_placeholder')

       sale_result = data["price_group"]
       df3 = df[df['price_group'] == sale_result]
       df3 = df3[df3['district_id'] == data["district_id"]]

       df3['date'] = pd.to_datetime(df3['date'])
       convert_date_sale = df3['date'].dt.date.value_counts()
       sort_date_sale = convert_date_sale.sort_index()
       # Replace NaN values with a placeholder
       sort_date_sale = sort_date_sale.fillna('NaN_placeholder')
       print("sort_date_sale", sort_date_sale)

       # merged = pd.merge(sort_date_rental, sort_date_sale, on=['date'], how='outer')

       return {"rental": sort_date_rental.to_dict(), "sale": sort_date_sale.to_dict() }

@app.post("/priceRange")
def priceRange(data: Trend):
       data = data.dict()
       print(data)
       df = pd.read_csv('./final.csv')    
       rental_result = data["rental_group"]
       df2 = df[df['rental_group'] == rental_result]
       df2 = df2[df2['price_group'] == data["price_group"]]
       df2 = df2[df2['district_id'] == data["district_id"]]

       result = df2[['project_name_x','rental','price']]
       print(result)

       return {"data" : result.to_dict(orient="records")}

@app.get("/allDataTrend")
def allDataTrend():
       df = pd.read_csv('./final.csv')    
       all_date = pd.to_datetime(df['date'])
       convert_date = all_date.dt.date.value_counts()
       sort_date = convert_date.sort_index()

       return {"data": sort_date.to_dict() }

@app.post("/year")
def predict(data: Year):
    data = data.dict()
    print(data)
    # Extracting data from the request
    features = [data["year"]]

    # Converting features to a NumPy array
    features_array = np.array(features).reshape(1, -1)

    # Predicting with the classifier
    prediction = classifier3.predict(features_array)

    # Converting NumPy int64 to regular Python int
    prediction = prediction.item()

    result = ((prediction-193.3)/193.3)*100

    return {"prediction": round(result, 2)}