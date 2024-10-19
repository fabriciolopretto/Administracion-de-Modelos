from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import pandas as pd

app = FastAPI()
    
# Endpoint para el mensaje de bienvenida
@app.get("/")
async def welcome():
    return {"mensaje": "Bienvenido a la API de predicción de tumores de cáncer de mama."}

# Cargar el modelo
with open('./model/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el scaler
with open('./model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Predecir probabilidades
@app.post("/predict/")
async def predict(data: dict):
    try:
        data_array = np.array([[value for value in data.values()]])
        scaled_inputs = scaler.transform(data_array)
        probabilities = model.predict_proba(scaled_inputs)
                
        return probabilities

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


