import joblib  
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder  
from fastapi import FastAPI  
from pydantic import BaseModel  
import uvicorn  

# Load dataset from a CSV file  
df = pd.read_csv('cars.csv')  

# Encode 'Gender' (Male=1, Female=0)  
le = LabelEncoder()  
df["Gender"] = le.fit_transform(df["Gender"])  

# Define features (X) and target (y)  
X = df[["Gender", "Age"]]  # Features  
y = df["Car_Type"]         # Target variable  

# Encode target variable if categorical  
if y.dtype == 'object':  
    y = le.fit_transform(y)  

# Split the dataset (80% Train, 20% Test)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Train the model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Save the trained model and encoder  
joblib.dump(model, "car-recommender.joblib")  
joblib.dump(le, "label_encoder.joblib")  
print("✅ Model trained and saved as 'car-recommender.joblib'")  

# FastAPI Implementation  
app = FastAPI()  
model = joblib.load("car-recommender.joblib")  
label_encoder = joblib.load("label_encoder.joblib")  

# Define a request model  
class CarUser(BaseModel):  
    gender: str  
    age: int  

@app.get('/')  
def index():  
    return {'message': 'Cars Recommender ML API'}  

@app.post('/car/predict')  
def predict_car_type(data: CarUser):  
    gender = 1 if data.gender.lower() == "male" else 0  
    age = data.age  
    
    prediction = model.predict([[gender, age]])  
    predicted_label = round(prediction[0])  
    predicted_car_type = label_encoder.inverse_transform([predicted_label])[0]  
    
    return {  
        'prediction': predicted_car_type  
    }  

if __name__ == '__main__':  
    uvicorn.run(app, host='127.0.0.1', port=8000)