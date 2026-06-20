import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import Literal
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Paths to model artifacts
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

# Paths for Scaler and LabelEncoder
SCALER_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler_a.joblib')
SCALER_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler_b.joblib')
LABEL_ENCODER_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder_a.joblib')
LABEL_ENCODER_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'label_encoder_b.joblib')

scaler_a = joblib.load(SCALER_A_PATH)

scaler_b = joblib.load(SCALER_B_PATH)

label_encoder_a = joblib.load(LABEL_ENCODER_A_PATH)

label_encoder_b = joblib.load(LABEL_ENCODER_B_PATH)

# Load reference data for drift detection
DATASET_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'modeling_dataset.parquet')

# Pydantic models
class ModelAInput(BaseModel):
    age: int = Field(..., gt=0, le=120)
    length_of_stay_hours: float = Field(..., gt=0)
    billed_amount: float = Field(..., gt=0)
    approved_amount: float = Field(..., gt=0)
    payment_days: float = Field(..., ge=0)
    visit_frequency: float = Field(..., gt=0)
    avg_length_of_stay_per_patient: float = Field(..., gt=0)
    provider_rejection_rate: float = Field(..., ge=0, le=1)
    days_since_registration: int = Field(..., ge=0)
    visit_month: int = Field(..., ge=1, le=12)
    visit_day_of_week: int = Field(..., ge=0, le=6)
    visit_day_of_year: int = Field(..., ge=1, le=366)
    gender: Literal['M', 'F']
    city: Literal['Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'Delhi', 'Chennai']
    insurance_provider: Literal['SecureLife', 'HealthPlus', 'CareOne', 'MediCareX']
    department: Literal['General', 'ER', 'Neurology', 'Orthopedics', 'Cardiology', 'ICU']
    visit_type: Literal['ER', 'OPD', 'ICU']
    chronic_flag: Literal[0, 1]
    weekend_visit: bool

class ModelBInput(BaseModel):
    age: int = Field(..., gt=0, le=120)
    length_of_stay_hours: float = Field(..., gt=0)
    billed_amount: float = Field(..., gt=0)
    approved_amount: float = Field(..., gt=0)
    payment_days: float = Field(..., ge=0)
    visit_frequency: float = Field(..., gt=0)
    avg_length_of_stay_per_patient: float = Field(..., gt=0)
    provider_rejection_rate: float = Field(..., ge=0, le=1)
    days_since_registration: int = Field(..., ge=0)
    visit_month: int = Field(..., ge=1, le=12)
    visit_day_of_week: int = Field(..., ge=0, le=6)
    visit_day_of_year: int = Field(..., ge=1, le=366)
    gender: Literal['M', 'F']
    city: Literal['Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'Delhi', 'Chennai']
    insurance_provider: Literal['SecureLife', 'HealthPlus', 'CareOne', 'MediCareX']
    department: Literal['General', 'ER', 'Neurology', 'Orthopedics', 'Cardiology', 'ICU']
    visit_type: Literal['ER', 'OPD', 'ICU']
    chronic_flag: Literal[0, 1]
    risk_score: Literal['High', 'Low', 'Medium']
    weekend_visit: bool

# Categorical columns for one-hot encoding (extracted from training phase logic)
CATEGORICAL_COLS_MODEL_A = [
    'gender', 'city', 'insurance_provider', 'department', 'visit_type',
    'chronic_flag', 'weekend_visit'
]
CATEGORICAL_COLS_MODEL_B = [
    'gender', 'city', 'insurance_provider', 'department', 'visit_type',
    'chronic_flag', 'risk_score', 'weekend_visit'
]

def align_dataframe_with_schema(df: pd.DataFrame, categorical_cols_to_encode: list, target_schema: list):

    df_processed = df.copy()

    # Ensure all listed categorical columns are handled correctly for one-hot encoding
    for col in categorical_cols_to_encode:
        if col in df_processed.columns:
            if df_processed[col].dtype == 'bool':
                # Convert boolean to integer (0 or 1) for consistent one-hot encoding behavior
                df_processed[col] = df_processed[col].astype(int)
            else:
                # Convert all other categorical-designated columns to string
                df_processed[col] = df_processed[col].astype(str)

    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols_to_encode, drop_first=False)

    # Explicitly convert all one-hot encoded columns (uint8 or bool) to int64 for Evidently compatibility
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'uint8' or df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype('int64')

    # Create a new DataFrame with all columns from target_schema, filled with zeros
    final_df = pd.DataFrame(0, index=df_encoded.index, columns=target_schema)

    # Fill in the columns that exist in both df_encoded and target_schema
    for col in df_encoded.columns:
        if col in final_df.columns:
            final_df[col] = df_encoded[col]

    return final_df

def preprocess_input(data: BaseModel, feature_schema: list, model_type: str):
    df = pd.DataFrame([data.dict()])

    # Categorical columns that were one-hot encoded during training
    cols_to_apply = []
    if model_type == 'model_a':
        cols_to_apply = CATEGORICAL_COLS_MODEL_A
    elif model_type == 'model_b':
        cols_to_apply = CATEGORICAL_COLS_MODEL_B

    # Ensure all listed categorical columns are handled correctly for one-hot encoding
    for col in cols_to_apply:
        if col in df.columns:
            if df[col].dtype == 'bool':
                # Convert boolean to integer (0 or 1) for consistent one-hot encoding behavior
                df[col] = df[col].astype(int)
            else:
                # Convert all other categorical-designated columns to string
                df[col] = df[col].astype(str)

    df_processed = pd.get_dummies(df, columns=cols_to_apply, drop_first=False)

    # Explicitly convert all one-hot encoded columns (uint8 or bool) to int64 for Evidently compatibility
    for col in df_processed.columns:
        if df_processed[col].dtype == 'uint8' or df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype('int64')

    df_final = pd.DataFrame(0, index=[0], columns=feature_schema)
    for col in df_processed.columns:
        if col in df_final.columns:
            df_final[col] = df_processed[col]

    return df_final

def generate_data_drift_report(current_data: pd.DataFrame, reference_data: pd.DataFrame, report_name: str):
    if reference_data is None: # Check if reference_data is None here
        logger.warning(f"Reference data not available for {report_name}. Skipping drift report generation.")
        return None

    try:
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)
        report_path = os.path.join(MODEL_ARTIFACTS_DIR, report_name)
        data_drift_report.save_html(report_path)
        logger.info(f"Data drift report saved to {report_path}")
        return report_path
    except Exception as e:
        logger.exception(f"Error generating data drift report {report_name}: {e}")
        return None

try:
    full_reference_data = pd.read_parquet(DATASET_PATH)

    base_reference_a = full_reference_data.drop(columns=['risk_score', 'claim_status'], errors='ignore')
    reference_data_a = align_dataframe_with_schema(
        base_reference_a,
        CATEGORICAL_COLS_MODEL_A,
        feature_schema_a
    )

    base_reference_b = full_reference_data.drop(columns=['claim_status'], errors='ignore')
    reference_data_b = align_dataframe_with_schema(
        base_reference_b,
        CATEGORICAL_COLS_MODEL_B,
        feature_schema_b
    )

    logger.info("Reference data for drift detection loaded and preprocessed successfully.")
except Exception as e:
    logger.error(f"Error loading and preprocessing reference data for drift detection: {e}")
    full_reference_data = None
    reference_data_a = None
    reference_data_b = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict_a")
async def predict_model_a(data: ModelAInput):

    if model_a is None or not feature_schema_a:
        logger.error("Model A model is missing or Feature Schema is missing.")
        return {"error": "Model A model is missing or Feature Schema is missing."}

    try:
        processed_data = preprocess_input(data, feature_schema_a, 'model_a')

        # Generate data drift report for Model A
        drift_report_path = generate_data_drift_report(processed_data, reference_data_a, 'data_drift_report_a.html')

        processed_data_scaled = scaler_a.transform(processed_data)
        prediction = model_a.predict(processed_data_scaled)
        predicted_class = label_encoder_a.inverse_transform(prediction)[0]

        logger.info(f"Model A Prediction Response: {{'prediction': predicted_class}}")
        return {"prediction": predicted_class, "data_drift_report": drift_report_path}

    except Exception as e:
        logger.exception(f"Error during Model A prediction: {e}")
        return {"error": str(e)}

@app.post("/predict_b")
async def predict_model_b(data: ModelBInput):

    if model_b is None or not feature_schema_b:
        logger.error("Model B is not ready for predictions.")
        return {"error": "Model B is not ready for predictions."}

    try:
        processed_data = preprocess_input(data, feature_schema_b, 'model_b')

        # Generate data drift report for Model B
        drift_report_path = generate_data_drift_report(processed_data, reference_data_b, 'data_drift_report_b.html')

        processed_data_scaled = scaler_b.transform(processed_data)
        prediction = model_b.predict(processed_data_scaled)
        predicted_class = label_encoder_b.inverse_transform(prediction)[0]

        logger.info(f"Model B Prediction Response: {{'prediction': predicted_class}}") # Log prediction result
        return {"prediction": predicted_class, "data_drift_report": drift_report_path}

    except Exception as e:
        logger.exception(f"Error during Model B prediction: {e}")
        return {"error": str(e)}

@app.get("/drift_report/{model_type}")
async def get_drift_report(model_type: Literal['a', 'b']):
    report_filename = f"data_drift_report_{model_type}.html"
    report_path = os.path.join(MODEL_ARTIFACTS_DIR, report_filename)
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            html_content = f.read()
        return {"report_html": html_content}
    else:
        return {"error": f"Drift report for Model {model_type.upper()} not found."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
