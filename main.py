from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
API_SYSTEM = "PADE 1.1"
API_VERSION = "0.1.0"
Signo = Literal["A", "B", "C", "D", "E", "F"]
TipoSituacion = Literal["S", "C", "X"]
class ResponseItem(BaseModel):
    id: int = Field(..., ge=1)
    signo: Signo
    intensidad: int = Field(..., ge=0, le=5)  # G
    tipo: TipoSituacion  # S/C/X
class RunPayload(BaseModel):
    responses: List[ResponseItem]
    @model_validator(mode="after")
    def validate_sequence(self) -> "RunPayload":
        n = len(self.responses)
        if n not in (12, 20):
            raise ValueError("responses debe tener longitud 12 o 20")
        # IDs estrictamente crecientes
        last_id = None
        for r in self.responses:
            if last_id is not None and r.id <= last_id:
                raise ValueError("IDs deben ser estrictamente crecientes")
            last_id = r.id
        return self
class RunRequest(BaseModel):
    payload: RunPayload
    alpha: float = Field(..., ge=0.0)
    beta: List[float] = Field(..., min_length=3, max_length=3)
    lambda_c: float = Field(..., ge=0.0, le=1.0)
    @field_validator("beta")
    @classmethod
    def validate_beta(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError("beta debe tener 3 valores")
        if abs(sum(v) - 1.0) > 1e-9:
            raise ValueError("beta debe sumar 1.0")
        # Regla formal: beta[0] >= beta[2] >= beta[1]
        if not (v[0] >= v[2] >= v[1]):
            raise ValueError("beta debe cumplir beta[0] >= beta[2] >= beta[1]")
        return v
def _protected_report_from_payload(responses: List[ResponseItem]) -> Dict[str, Any]:
    """
    Informe protegido:
    - NO expone A–F
    - NO expone G
    - NO expone S/C/X
    """
    counts = {k: 0 for k in ("A", "B", "C", "D", "E", "F")}
    g_vals: List[int] = []
    t_counts = {"S": 0, "C": 0, "X": 0}
    for r in responses:
        counts[r.signo] += 1
        g_vals.append(r.intensidad)
        t_counts[r.tipo] += 1
    # determinismo en empates
    order = ["A", "B", "C", "D", "E", "F"]
    top = sorted(counts.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    primary = top[0][0]
    secondary = top[1][0]
    # frases oficiales (sin letras)
    phrase = {
        "A": "actuar de forma directa sobre la situación",
        "B": "organizar la situación mediante ajustes",
        "C": "mantener la situación sin intervención activa",
        "D": "retirarse de la situación sin resolución",
        "E": "adaptarse personalmente a las condiciones",
        "F": "responder sin aplicar una estrategia definida",
    }
    mean_g = sum(g_vals) / max(1, len(g_vals))
    if mean_g <= 1.5:
        intensity_txt = "respuestas mayormente inmediatas"
    elif mean_g <= 3.5:
        intensity_txt = "respuestas con un equilibrio entre reflexión y acción"
    else:
        intensity_txt = "respuestas con mayor dificultad o demora en la decisión"
    # bucket de contexto (sin S/C/X)
    max_t = max(t_counts["S"], t_counts["C"], t_counts["X"])
    if t_counts["S"] == max_t and t_counts["S"] >= t_counts["C"] and t_counts["S"] >= t_counts["X"]:
        context_txt = "situaciones de baja exigencia"
    elif t_counts["C"] == max_t and t_counts["C"] >= t_counts["S"] and t_counts["C"] >= t_counts["X"]:
        context_txt = "situaciones con complejidad moderada"
    else:
        context_txt = "situaciones de mayor exigencia o implicación"
    return {
        "level": "medium",
        "summary": f"Se observa un patrón predominante orientado a {phrase[primary]}.",
        "details": [
            f"Este comportamiento se complementa con momentos orientados a {phrase[secondary]}.",
            "El patrón se mantiene estable a lo largo del proceso.",
            f"Las decisiones muestran {intensity_txt}.",
            f"El comportamiento se mantiene consistente ante {context_txt}.",
        ],
        "conclusion": "Se observa una forma de actuación estable, consistente y orientada a la resolución de situaciones.",
    }
app = FastAPI(title="PADE 1.1 API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "PADE API funcionando"}
@app.get("/ping")
def ping() -> Dict[str, str]:
    return {"status": "ok"}
@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    try:
        responses = req.payload.responses
        vector_G = [r.intensidad for r in responses]
        report = _protected_report_from_payload(responses)
        counts = {k: 0 for k in ("A", "B", "C", "D", "E", "F")}
        for r in responses:
            counts[r.signo] += 1
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}
        entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
        max_p = max(probs.values())
        if entropy < 1.0:
            entropy_level = "low"
        elif entropy < 1.5:
            entropy_level = "medium"
        else:
            entropy_level = "high"
        probabilistic_module = {
            "distribution": list(probs.values()),  
            "entropy": entropy,
            "entropy_level": entropy_level,
            "dominance": max_p,
        }
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error al generar el informe protegido"
        )
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ProductionReady",
        "system": API_SYSTEM,
        "timestamp": now,
        "vector_G": vector_G,
        "metrics": {},
        "probabilistic_module": probabilistic_module, 
        "report": report,
        "trace": {
            "api_version": API_VERSION,
            "semantic_layer": "v1.0-closed",
        },
    }
