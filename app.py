import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from typing import Literal

try:
    import evidently
    logger.info('Evidently module is available at app runtime.')
except ImportError as e:
    logger.error(f'Evidently module is NOT available at app runtime: {e}')
    raise # Re-raise to ensure the error is visible
    
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='predictions.log',
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
FEATURE_SCHEMA_A_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_schema_model_a.joblib') 

# Model B artifacts
MODEL_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'random_forest_model_b.joblib')
FEATURE_SCHEMA_B_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_schema_model_b.joblib') 

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

try:
    full_reference_data = pd.read_parquet(DATASET_PATH)

    reference_data_a = full_reference_data.drop(columns=['risk_score', 'claim_status'], errors='ignore')

    reference_data_b = full_reference_data.drop(columns=['claim_status'], errors='ignore')

    logger.info("Reference data for drift detection loaded successfully.")
except Exception as e:
    logger.error(f"Error loading reference data for drift detection: {e}")
    full_reference_data = None
    reference_data_a = None
    reference_data_b = None


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

def preprocess_input(data: BaseModel, feature_schema: list, model_type: str):
    df = pd.DataFrame([data.model_dump()])

    # Categorical columns that were one-hot encoded during training
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

    for col in cols_to_apply:
        if col in df.columns and df[col].dtype == 'object': # Only convert 'object' types to str
            df[col] = df[col].astype(str)

    df_processed = pd.get_dummies(df, columns=cols_to_apply, drop_first=False)

    df_final = pd.DataFrame(0, index=[0], columns=feature_schema)
    for col in df_processed.columns:
        if col in df_final.columns:
            df_final[col] = df_processed[col]

    return df_final

def generate_data_drift_report(current_data: pd.DataFrame, reference_data: pd.DataFrame, report_name: str):
    if reference_data is None:
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict_a")
async def predict_model_a(data: ModelAInput):
    logger.info(f"Model A Prediction Request: {data.model_dump()}") 

    if model_a is None or not feature_schema_a:
        logger.error("Model A model is missing or Feature Schema is missing.")
        return {"error": "Model A model is missing or Feature Schema is missing."}, 500

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
        return {"error": str(e)}, 500

@app.post("/predict_b")
async def predict_model_b(data: ModelBInput):
    logger.info(f"Model B Prediction Request: {data.model_dump()}") 

    if model_b is None or not feature_schema_b:
        logger.error("Model B is not ready for predictions.")
        return {"error": "Model B is not ready for predictions."}, 500

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
        return {"error": str(e)}, 500

@app.get("/drift_report/{model_type}")
async def get_drift_report(model_type: Literal['a', 'b']):
    report_filename = f"data_drift_report_{model_type}.html"
    report_path = os.path.join(MODEL_ARTIFACTS_DIR, report_filename)
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            html_content = f.read()
        return {"report_html": html_content}
    else:
        return {"error": f"Drift report for Model {model_type.upper()} not found. Make a prediction first to generate it."}, 404

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
