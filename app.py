from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import joblib
import pandas as pd
import numpy as np
import os
from model.model_class import LSTMModel
from fastapi.middleware.cors import CORSMiddleware
import psutil

app = FastAPI(
    title="FIAP Tech Challenge - PETR4 Predictor",
    description="API para predição de preços da Petrobras. Nota: A sequência de 20 dias deve ser enviada do mais antigo para o mais recente.",
    version="1.0.0"
)

import time
from fastapi import FastAPI, Request
# ... seus outros imports ...

app = FastAPI()

# MIDDLEWARE DE MONITORAMENTO
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/monitoramento")
async def monitoramento():
    process = psutil.Process(os.getpid())
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_usage_mb": process.memory_info().rss / (1024 * 1024),
        "server_status": "online"
    }

# --- CONFIGURAÇÕES ---
MODEL_PATH = "model/modelo_lstm_final.pth"
SCALER_PATH = "model/scaler_final.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CARREGAMENTO ---
scaler = joblib.load(SCALER_PATH)
model = LSTMModel(input_size=2, hidden_size=119, num_layers=1, dropout=0.0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- MODELO COM INPUT DEFAULT ---
class PredictionInput(BaseModel):
    # Field(..., json_schema_extra={...}) define o exemplo no Swagger
    history: list[dict] = Field(
        ..., 
        description="Lista com 20 dicionários contendo 'close' e 'open'",
        json_schema_extra={
            "example": [
                {"close": 35.10, "open": 34.80}, {"close": 35.30, "open": 35.10},
                {"close": 35.50, "open": 35.30}, {"close": 35.80, "open": 35.50},
                {"close": 36.00, "open": 35.80}, {"close": 36.20, "open": 36.00},
                {"close": 36.40, "open": 36.20}, {"close": 36.30, "open": 36.40},
                {"close": 36.10, "open": 36.30}, {"close": 35.90, "open": 36.10},
                {"close": 35.70, "open": 35.90}, {"close": 35.50, "open": 35.70},
                {"close": 35.30, "open": 35.50}, {"close": 35.10, "open": 35.30},
                {"close": 35.00, "open": 35.10}, {"close": 35.20, "open": 35.00},
                {"close": 35.40, "open": 35.20}, {"close": 35.60, "open": 35.40},
                {"close": 35.80, "open": 35.60}, {"close": 36.10, "open": 35.80}
            ]
        }
    )

@app.get("/")
def home():
    return {"status": "online", "docs": "/docs"}

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        if len(data.history) != 20:
            raise HTTPException(status_code=400, detail="Forneça exatamente 20 dias de histórico.")

        df = pd.DataFrame(data.history)[['close', 'open']]
        scaled_data = scaler.transform(df.values)
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor).cpu().item()

        dummy = np.zeros((1, 2))
        dummy[0, 0] = output
        prediction = scaler.inverse_transform(dummy)[0, 0]

        return {
            "prediction_next_close": round(float(prediction), 2),
            "currency": "BRL"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))