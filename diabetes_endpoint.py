
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Deploy Diabetes Model - Logistic Regression",
    version="1.0.0"
)

# ------------------------------------------------------------
# LOAD THE AI MODEL
# ------------------------------------------------------------
model = joblib.load("model/diabetes_logistic_model.pkl")


@app.post("/api/v1/predict-diabetes", tags=["diabetes"])
async def predict(
    Pregnancies: float,
    Glucose: float,
    BloodPressure: float,
    SkinThickness: float,
    Insulin: float,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: float
):
    try:
        input_data = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age
        }

        df = pd.DataFrame(input_data, index=[0])
        prediction = model.predict(df)[0]

        result = "Diabético" if prediction == 1 else "No diabético"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction": result}
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )
