import json
import pickle
import time
from urllib.request import Request
import numpy as np
import pandas as pd
import boto3
import mlflow
import logging
import threading


from typing import Literal, Optional
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

# Configuración básica del logging
logging.basicConfig(level=logging.INFO)

# Función para cargar el modelo
def load_model(model_name: str, alias: str):
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()
        logging.info("MLflow client initialized.")
        
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        logging.info(f"Model data fetched: {model_data_mlflow}")
        
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        logging.info("Model loaded from MLflow.")
        
        version_model_ml = int(model_data_mlflow.version)
    except Exception as e:
        logging.error(f"Error loading model from MLflow: {e}")
        with open('/app/files/model.pkl', 'rb') as file_ml:
            model_ml = pickle.load(file_ml)
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except Exception as e:
        logging.error(f"Error loading data dictionary from S3: {e}")
        with open('/app/files/data.json', 'r') as file_s3:
            data_dictionary = json.load(file_s3)

    return model_ml, version_model_ml, data_dictionary



def transform_single_example(raw_example, wind_dir_mapping):
    """
    Transforma un ejemplo único de datos crudos de clima y devuelve un diccionario con los campos transformados.
    
    Parameters:
    raw_example (dict): Diccionario con los datos crudos de un solo ejemplo.
    wind_dir_mapping (dict): Diccionario que mapea direcciones del viento a ángulos.

    Returns:
    dict: Diccionario con los datos transformados.
    """
    
    # Convertir el ejemplo en un DataFrame
    weather_df = pd.DataFrame([raw_example])

    # Verificar si las tres claves están presentes
    wind_keys = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    if all(key in weather_df.columns for key in wind_keys):
        # Codificación de la dirección del viento en componentes seno y coseno
        def encode_wind_dir(df, col, mapping):
            angles = df[col].map(mapping)
            angles_rad = np.deg2rad(angles)
            df[f'{col}_sin'] = np.sin(angles_rad)
            df[f'{col}_cos'] = np.cos(angles_rad)
            df.drop(columns=col, inplace=True)

        encode_wind_dir(weather_df, 'WindGustDir', wind_dir_mapping)
        encode_wind_dir(weather_df, 'WindDir9am', wind_dir_mapping)
        encode_wind_dir(weather_df, 'WindDir3pm', wind_dir_mapping)

    # Convertir el DataFrame transformado de nuevo a un diccionario
    transformed_data = weather_df.iloc[0].to_dict()
    return transformed_data

# Clase de entrada basada en los 29 features del dataset con validaciones y descripciones
# Los features con alta correlacion se configuraron como opcionales
class ModelInput(BaseModel):
    MinTemp: float = Field(
        description="Minimum temperature of the day", ge=-50, le=60
    )
    MaxTemp: float = Field(
        description="Maximum temperature of the day", ge=-50, le=60
    )
    Rainfall: float = Field(
        description="Amount of rainfall in mm", ge=0, le=500
    )
    Evaporation: float = Field(
        description="Evaporation in mm", ge=0, le=50
    )
    Sunshine: float = Field(
        description="Sunshine duration in hours", ge=0, le=15
    )
    WindGustSpeed: float = Field(
        description="Maximum wind gust speed in km/h", ge=0, le=200
    )
    WindSpeed9am: float = Field(
        description="Wind speed at 9am in km/h", ge=0, le=150
    )
    WindSpeed3pm: float = Field(
        description="Wind speed at 3pm in km/h", ge=0, le=150
    )
    Humidity9am: float = Field(
        description="Humidity at 9am in percentage", ge=0, le=100
    )
    Humidity3pm: float = Field(
        description="Humidity at 3pm in percentage", ge=0, le=100
    )
    Pressure9am: Optional[float] = Field(
        default=None,
        description="Pressure at 9am in hPa", ge=0, le=1100
    )
    Pressure3pm: float = Field(
        description="Pressure at 3pm in hPa", ge=0, le=1100
    )
    Cloud9am: float = Field(
        description="Cloud cover at 9am on a scale of 0-9", ge=0, le=9
    )
    Cloud3pm: float = Field(
        description="Cloud cover at 3pm on a scale of 0-9", ge=0, le=9
    )
    Temp9am: Optional[float] = Field(
        default=None,
        description="Temperature at 9am in Celsius", ge=-50, le=60
    )
    Temp3pm: Optional[float] = Field(
        default=None,
        description="Temperature at 3pm in Celsius", ge=-50, le=60
    )
    RainToday: int = Field(
        description="Whether it rained today: 1 for Yes, 0 for No", ge=0, le=1
    )
    Year: int = Field(
        description="Year of the observation", ge=1900, le=2100
    )
    Month: int = Field(
        description="Month of the observation", ge=1, le=12
    )
    Day: int = Field(
        description="Day of the observation", ge=1, le=31
    )
    Latitude: float = Field(
        description="Latitude of the location", ge=-90, le=90
    )
    Longitude: float = Field(
        description="Longitude of the location", ge=-180, le=180
    )
    WindGustDir_sin: float = Field(
        description="Sine of the wind gust direction", ge=-1, le=1
    )
    WindGustDir_cos: float = Field(
        description="Cosine of the wind gust direction", ge=-1, le=1
    )
    WindDir9am_sin: float = Field(
        description="Sine of the wind direction at 9am", ge=-1, le=1
    )
    WindDir9am_cos: float = Field(
        description="Cosine of the wind direction at 9am", ge=-1, le=1
    )
    WindDir3pm_sin: float = Field(
        description="Sine of the wind direction at 3pm", ge=-1, le=1
    )
    WindDir3pm_cos: float = Field(
        description="Cosine of the wind direction at 3pm", ge=-1, le=1
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MinTemp": 12.3,
                    "MaxTemp": 24.7,
                    "Rainfall": 0.0,
                    "Evaporation": 5.6,
                    "Sunshine": 9.8,
                    "WindGustSpeed": 35.0,
                    "WindSpeed9am": 20.0,
                    "WindSpeed3pm": 24.0,
                    "Humidity9am": 82.0,
                    "Humidity3pm": 55.0,
                    "Pressure9am": 1015.0,
                    "Pressure3pm": 1012.0,
                    "Cloud9am": 4.0,
                    "Cloud3pm": 3.0,
                    "Temp9am": 17.2,
                    "Temp3pm": 22.4,
                    "RainToday": 0,
                    "Year": 2023,
                    "Month": 8,
                    "Day": 24,
                    "Latitude": -33.86,
                    "Longitude": 151.21,
                    "WindGustDir_sin": 0.7071,
                    "WindGustDir_cos": 0.7071,
                    "WindDir9am_sin": 0.5,
                    "WindDir9am_cos": 0.866,
                    "WindDir3pm_sin": -0.5,
                    "WindDir3pm_cos": -0.866
                }
            ]
        }
    }


# Clase de salida para el modelo
class ModelOutput(BaseModel):
    int_output: bool
    str_output: Literal["No rain tomorrow", "Rain expected tomorrow"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Rain expected tomorrow"
                }
            ]
        }
    }


direccion_to_angulo = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
    'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def load_json_from_s3(bucket_name, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    json_data = json.loads(response['Body'].read().decode('utf-8'))
    return json_data

app = FastAPI()

# Cargar el modelo antes de iniciar
model, version_model, data_dict = load_model("rain_australia_model_prod", "champion")


@app.get("/")
async def read_root():
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Weather Prediction API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[ModelInput, Body(embed=True)],
    background_tasks: BackgroundTasks
):
    features_dict = features.dict()

    transformed_data_dict = transform_single_example(features_dict, direccion_to_angulo)

    print(f"transformed_data_dict: {transformed_data_dict}")

    print(f"features: {features_dict}")
    features_list = [features_dict[key] for key in features_dict.keys()]

    # Convertir a DataFrame
    features_df = pd.DataFrame([features_list], columns=features_dict.keys())
    print(f"DataFrame columns: {features_df.columns}")

    # Se escalan los datos utilizando standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    try:
        # Realizar la predicción
        prediction = model.predict(features_df)
        logging.info(f'Prediction result: {prediction}')
    except Exception as e:
        logging.error(f"Error in model prediction: {e}")
        return JSONResponse(content=jsonable_encoder({"error": "Model prediction failed."}), status_code=500)

    # Convertir el resultado a cadena legible
    str_pred = "No rain tomorrow" if prediction[0] == 0 else "Rain expected tomorrow"

    # Verificar cambios de modelo en segundo plano
    background_tasks.add_task(load_model, "rain_australia_model_prod", "champion")

    # Retornar la predicción
    return ModelOutput(int_output=bool(prediction[0]), str_output=str_pred)

class StatusResponse(BaseModel):
    status: str

@app.post("/update-model", response_model=StatusResponse)
async def update_model():
    # Verifica si existe un nuevo modelo para actualizar
    status = check_for_new_model("rain_australia_model_prod", "champion")
    
    if status == 200:
        return StatusResponse(status="Modelo actualizado")
    elif status == 201:
        return StatusResponse(status="Sin cambios")
    else:
        return StatusResponse(status="Se utiliza modelo local")

def check_for_new_model(model_name: str, alias: str):
    global model, version_model, data_dict
    status_code = 400
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)

        new_version = int(model_data_mlflow.version)
        if new_version > version_model:
            logging.info(f"Nueva versión del modelo encontrada: {new_version}")
            model, version_model, data_dict = load_model(model_name, alias)
            logging.info(f"Modelo actualizado a la versión {version_model}")
            status_code = 200
        else:
            logging.info("No se encontró una nueva versión del modelo.")
            status_code = 201

    except Exception as e:
        logging.error(f"No hay modelos disponibles en MLFlow, se utiliza modelo local: {e}")
    
    return status_code

        