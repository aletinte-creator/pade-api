from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "PADE API funcionando"}

@app.get("/ping")
def ping():
    return {"status": "ok"}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any

app = FastAPI()

# CORS para que tu web (Hostinger) pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo acotamos a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

Signo = Literal["A","B","C","D","E","F"]
Tipo = Literal["S","C","X"]

class ResponseItem(BaseModel):
    id: int
    signo: Signo
    intensidad: int = Field(ge=0, le=5)
    tipo: Tipo

class RunPayload(BaseModel):
    responses: List[ResponseItem]

class RunRequest(BaseModel):
    payload: RunPayload
    alpha: float = Field(ge=0)
    beta: List[float] = Field(min_length=3, max_length=3)
    lambda_c: float = Field(ge=0, le=1)

@app.get("/")
def home():
    return {"message": "PADE API funcionando"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    # STUB: luego conectamos aquí el core MEV + capa semántica cerrada
    return {
        "status": "ProductionReady",
        "system": "PADE 1.1",
        "metrics": {},
        "probabilistic_module": {},
        "report": {
            "summary": "Ejecución correcta (stub)."
        },
        "trace": {}
    }
