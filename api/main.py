from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "student_success_weighted_xgboost.pkl"
)

LABEL_ENCODER_PATH = (
    BASE_DIR
    / "models"
    / "weighted_xgboost_label_encoder.pkl"
)

FEATURE_NAMES_PATH = (
    BASE_DIR
    / "models"
    / "weighted_xgboost_feature_names.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading student success model...")

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)

feature_names = joblib.load(
    FEATURE_NAMES_PATH
)

print("Model loaded successfully!")
print(
    f"Number of features: {len(feature_names)}"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Student Success Prediction API",
    description=(
        "AI API for predicting student academic outcomes "
        "using a class-weighted XGBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class StudentData(BaseModel):

    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int

    previous_qualification: int
    previous_qualification_grade: float

    nationality: int

    mothers_qualification: int
    fathers_qualification: int

    mothers_occupation: int
    fathers_occupation: int

    admission_grade: float

    displaced: int
    educational_special_needs: int

    debtor: int
    tuition_fees_up_to_date: int

    gender: int
    scholarship_holder: int

    age_at_enrollment: int

    international: int

    curricular_units_1st_sem_credited: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int

    curricular_units_2nd_sem_credited: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_evaluations: int
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_2nd_sem_without_evaluations: int

    unemployment_rate: float
    inflation_rate: float
    gdp: float


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Student Success Prediction API",
        "status": "running",
        "model": "Weighted XGBoost",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features": len(feature_names)
    }


# ============================================================
# PREDICT STUDENT SUCCESS
# ============================================================

@app.post("/api/predict")
def predict_student(student: StudentData):

    try:

        # ----------------------------------------------------
        # Convert request to dictionary
        # ----------------------------------------------------

        student_data = student.model_dump()

        # ----------------------------------------------------
        # Map API names to original dataset names
        # ----------------------------------------------------

        data = {

            "Marital status":
                student_data["marital_status"],

            "Application mode":
                student_data["application_mode"],

            "Application order":
                student_data["application_order"],

            "Course":
                student_data["course"],

            "Daytime/evening attendance":
                student_data[
                    "daytime_evening_attendance"
                ],

            "Previous qualification":
                student_data[
                    "previous_qualification"
                ],

            "Previous qualification (grade)":
                student_data[
                    "previous_qualification_grade"
                ],

            "Nacionality":
                student_data["nationality"],

            "Mother's qualification":
                student_data[
                    "mothers_qualification"
                ],

            "Father's qualification":
                student_data[
                    "fathers_qualification"
                ],

            "Mother's occupation":
                student_data[
                    "mothers_occupation"
                ],

            "Father's occupation":
                student_data[
                    "fathers_occupation"
                ],

            "Admission grade":
                student_data["admission_grade"],

            "Displaced":
                student_data["displaced"],

            "Educational special needs":
                student_data[
                    "educational_special_needs"
                ],

            "Debtor":
                student_data["debtor"],

            "Tuition fees up to date":
                student_data[
                    "tuition_fees_up_to_date"
                ],

            "Gender":
                student_data["gender"],

            "Scholarship holder":
                student_data[
                    "scholarship_holder"
                ],

            "Age at enrollment":
                student_data[
                    "age_at_enrollment"
                ],

            "International":
                student_data["international"],

            "Curricular units 1st sem (credited)":
                student_data[
                    "curricular_units_1st_sem_credited"
                ],

            "Curricular units 1st sem (enrolled)":
                student_data[
                    "curricular_units_1st_sem_enrolled"
                ],

            "Curricular units 1st sem (evaluations)":
                student_data[
                    "curricular_units_1st_sem_evaluations"
                ],

            "Curricular units 1st sem (approved)":
                student_data[
                    "curricular_units_1st_sem_approved"
                ],

            "Curricular units 1st sem (grade)":
                student_data[
                    "curricular_units_1st_sem_grade"
                ],

            "Curricular units 1st sem (without evaluations)":
                student_data[
                    "curricular_units_1st_sem_without_evaluations"
                ],

            "Curricular units 2nd sem (credited)":
                student_data[
                    "curricular_units_2nd_sem_credited"
                ],

            "Curricular units 2nd sem (enrolled)":
                student_data[
                    "curricular_units_2nd_sem_enrolled"
                ],

            "Curricular units 2nd sem (evaluations)":
                student_data[
                    "curricular_units_2nd_sem_evaluations"
                ],

            "Curricular units 2nd sem (approved)":
                student_data[
                    "curricular_units_2nd_sem_approved"
                ],

            "Curricular units 2nd sem (grade)":
                student_data[
                    "curricular_units_2nd_sem_grade"
                ],

            "Curricular units 2nd sem (without evaluations)":
                student_data[
                    "curricular_units_2nd_sem_without_evaluations"
                ],

            "Unemployment rate":
                student_data["unemployment_rate"],

            "Inflation rate":
                student_data["inflation_rate"],

            "GDP":
                student_data["gdp"]
        }

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame([data])

        # ----------------------------------------------------
        # IMPORTANT:
        # Force exactly the same feature order used
        # during model training.
        # ----------------------------------------------------

        input_df = input_df[
            feature_names
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction_encoded = model.predict(
            input_df
        )[0]

        prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_df
        )[0]

        probability_dict = {}

        for index, probability in enumerate(
            probabilities
        ):

            label = label_encoder.inverse_transform(
                [index]
            )[0]

            probability_dict[label] = round(
                float(probability),
                4
            )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            probabilities
        )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "prediction": prediction,

            "confidence": round(
                float(confidence),
                4
            ),

            "probabilities": probability_dict

        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )