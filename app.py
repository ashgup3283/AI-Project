import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import logging
import os

# Configure logging
#logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

# Define paths to model artifacts
MODEL_ARTIFACTS_DIR = 'model_artifacts'

# Ensure directory exists
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

# Model A artifacts
MODEL_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'random_forest_model_a.joblib')
FEATURE_SCHEMA_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_schema_model_a.json')

# Model B artifacts
MODEL_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'random_forest_model_b.joblib')
FEATURE_SCHEMA_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_schema_model_b.json')

# Load Model A 
model_a = joblib.load(MODEL_A_PATH)
with open(FEATURE_SCHEMA_A_PATH, 'r') as f:
    feature_schema_a = json.load(f)

# Load Model B 
model_b = joblib.load(MODEL_B_PATH)
with open(FEATURE_SCHEMA_B_PATH, 'r') as f:
    feature_schema_b = json.load(f)

# --- IMPORTANT: SCALER AND LABEL ENCODER ARTIFACTS WERE NOT SAVED IN PREVIOUS STEPS ---
# The following section assumes these artifacts exist. In a real deployment,
# the StandardScaler and LabelEncoder objects fitted during training must be saved
# and loaded here to ensure consistent preprocessing and inverse transformation.

# Define paths to scaler and LabelEncoder artifacts (assuming they will be present)
SCALER_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler_a.joblib')
SCALER_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler_b.joblib')
LABEL_ENCODER_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder_a.joblib')
LABEL_ENCODER_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder_b.joblib')

scaler_a = joblib.load(SCALER_A_PATH)

scaler_b = joblib.load(SCALER_B_PATH)

label_encoder_a = joblib.load(LABEL_ENCODER_A_PATH)

label_encoder_b = joblib.load(LABEL_ENCODER_B_PATH)

# Pydantic models for request body
class ModelAInput(BaseModel):
    age: int
    length_of_stay_hours: float
    billed_amount: float
    approved_amount: float
    payment_days: float
    visit_frequency: float
    avg_length_of_stay_per_patient: float
    provider_rejection_rate: float
    days_since_registration: int
    visit_month: int
    visit_day_of_week: int
    visit_day_of_year: int
    gender: str
    city: str
    insurance_provider: str
    department: str
    visit_type: str
    chronic_flag: int
    weekend_visit: bool

class ModelBInput(BaseModel):
    age: int
    length_of_stay_hours: float
    billed_amount: float
    approved_amount: float
    payment_days: float
    visit_frequency: float
    avg_length_of_stay_per_patient: float
    provider_rejection_rate: float
    days_since_registration: int
    visit_month: int
    visit_day_of_week: int
    visit_day_of_year: int
    gender: str
    city: str
    insurance_provider: str
    department: str
    visit_type: str
    chronic_flag: int
    risk_score: str
    weekend_visit: bool

# Helper function for preprocessing input data
def preprocess_input(data: BaseModel, feature_schema: list, model_type: str):
    df = pd.DataFrame([data.model_dump()])

    #Categorical columns that were one-hot encoded during training
    categorical_cols_model_a = [
        'gender', 'city', 'insurance_provider', 'department', 'visit_type',
        'chronic_flag', 'weekend_visit'
    ]
    categorical_cols_model_b = [
        'gender', 'city', 'insurance_provider', 'department', 'visit_type',
        'chronic_flag', 'risk_score', 'weekend_visit'
    ]

    cols_to_apply = []
    if model_type == 'model_a':
        cols_to_apply = categorical_cols_model_a
    elif model_type == 'model_b':
        cols_to_apply = categorical_cols_model_b

    # Convert columns to string type 
    for col in cols_to_apply:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df_processed = pd.get_dummies(df, columns=cols_to_apply, drop_first=False)

    # Making all columns from the training schema are present, filling the missing with 0
    df_final = pd.DataFrame(0, index=[0], columns=feature_schema)
    for col in df_processed.columns:
        if col in df_final.columns:
            df_final[col] = df_processed[col]

    return df_final

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict_a")
async def predict_model_a(data: ModelAInput):

    if model_a is None or not feature_schema_a:
        return {"error": "Model A model is missing or Feature Schema is missing."}, 500

    try:
        processed_data = preprocess_input(data, feature_schema_a, 'model_a')

        processed_data_scaled = scaler_a.transform(processed_data)
        
        prediction = model_a.predict(processed_data_scaled)

        predicted_class = label_encoder_a.inverse_transform(prediction)[0]
        
        return {"prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/predict_b")
async def predict_model_b(data: ModelBInput):

    if model_b is None or not feature_schema_b:
        return {"error": "Model B is not ready for predictions."}, 500

    try:
        processed_data = preprocess_input(data, feature_schema_b, 'model_b')

        processed_data_scaled = scaler_b.transform(processed_data)
        
        prediction = model_b.predict(processed_data_scaled)

        predicted_class = label_encoder_b.inverse_transform(prediction)[0]
        
        return {"prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
