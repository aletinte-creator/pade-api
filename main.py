from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from mev01_core import MEVResponse, procesar_mev01_v13_rev

API_SYSTEM = "PADE 1.1"
API_VERSION = "0.1.0"

# Descripción institucional por nivel (capa seria)
CONCLUSION_BY_LEVEL = {
    "high": "El patrón presenta una forma de actuación estable y coherente, orientada a la resolución de situaciones.",
    "medium": "Se observa una dinámica de actuación organizada, pero con variaciones a lo largo del proceso.",
    "low": "Esta configuración despliega una distribución variable entre múltiples tendencias, sin una predominancia sostenida.",
}

Signo = Literal["A", "B", "C", "D", "E", "F"]
TipoSituacion = Literal["S", "C", "X"]


class ResponseItem(BaseModel):
    id: int = Field(..., ge=1)
    signo: Signo
    intensidad: int = Field(..., ge=0, le=5)  # G = DEMORA (0 inmediato, 5 mucha demora)
    tipo: TipoSituacion  # S/C/X


class RunPayload(BaseModel):
    responses: List[ResponseItem]

    @model_validator(mode="after")
    def validate_sequence(self) -> "RunPayload":
        n = len(self.responses)
        if n not in (12, 20):
            raise ValueError("responses debe tener longitud 12 o 20")
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
        if not (v[0] >= v[2] >= v[1]):
            raise ValueError("beta debe cumplir beta[0] >= beta[2] >= beta[1]")
        return v


def _protected_report_from_payload(responses: List[ResponseItem]) -> Dict[str, Any]:
    """
    Informe protegido (capa textual):
    - NO expone A–F (en forma de letras)
    - NO expone G explícitamente como valores individuales
    - NO expone S/C/X explícitamente
    (La API puede devolver datos técnicos aparte si decides mantenerlos.)
    """
    counts = {k: 0 for k in ("A", "B", "C", "D", "E", "F")}
    g_vals: List[int] = []
    t_counts = {"S": 0, "C": 0, "X": 0}

    for r in responses:
        counts[r.signo] += 1
        g_vals.append(r.intensidad)  # G = demora
        t_counts[r.tipo] += 1

    # determinismo en empates
    order = ["A", "B", "C", "D", "E", "F"]
    top = sorted(counts.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    primary = top[0][0]
    secondary = top[1][0]

    # frases humanas (sin letras)
    phrase = {
        "A": "actuar de forma directa sobre la situación",
        "B": "organizar la situación mediante ajustes",
        "C": "mantener la situación sin intervención activa",
        "D": "retirarse de la situación sin resolución",
        "E": "adaptarse personalmente a las condiciones",
        "F": "responder a desgano",
    }

    # timing (G = demora)
    mean_g = sum(g_vals) / max(1, len(g_vals))
    if mean_g <= 1.5:
        timing_txt = "respuestas mayormente inmediatas"
    elif mean_g <= 3.5:
        timing_txt = "respuestas con un equilibrio entre reflexión y acción"
    else:
        timing_txt = "respuestas con mayor dificultad o demora en la decisión"

    # bucket de contexto (sin exponer S/C/X)
    max_t = max(t_counts["S"], t_counts["C"], t_counts["X"])
    if t_counts["S"] == max_t and t_counts["S"] >= t_counts["C"] and t_counts["S"] >= t_counts["X"]:
        context_txt = "situaciones de baja exigencia"
    elif t_counts["C"] == max_t and t_counts["C"] >= t_counts["S"] and t_counts["C"] >= t_counts["X"]:
        context_txt = "situaciones con complejidad moderada"
    else:
        context_txt = "situaciones de mayor exigencia o implicación"

    return {
        "level": "medium",  # placeholder (se sobreescribe luego)
        "summary": f"Hay un patrón que se repite: tendés a {phrase[primary]}.",
        "details": [
            f"Aunque no es lo único que hacés: también aparece {phrase[secondary]}.",
            "No es ocasional. Es consistente a lo largo del proceso.",
            f"La clave no es solo la decisión: es el tiempo. Tus respuestas muestran {timing_txt}.",
            f"Y ese patrón se mantiene incluso ante {context_txt}.",
        ],
        "conclusion": "El problema no es lo que elegís. Es cuándo lo hacés.",
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

        # (si quieres “protegido real”, puedes quitar vector_G del output más adelante)
        vector_G = [r.intensidad for r in responses]  # G = demora

        report = _protected_report_from_payload(responses)

        # --- Llamada al CORE MEV01 v1.3 (ya alineado con G=demora) ---
        mev_responses = [
            MEVResponse(id=r.id, signo=r.signo, intensidad=r.intensidad, tipo=r.tipo)
            for r in responses
        ]

        mev_out = procesar_mev01_v13_rev(
            responses=mev_responses,
            alpha=req.alpha,
            beta=req.beta,
            lambda_c=req.lambda_c,
        )

        P_next = mev_out["probabilistic"]["P_opcion_next"]
        H = mev_out["probabilistic"]["H"]
        dominance = max(P_next)

        # --- Entropy level ---
        H_max = math.log(6, 2)
        H_norm = H / H_max

        if H_norm < 0.40:
            entropy_level = "low"
        elif H_norm < 0.65:
            entropy_level = "medium"
        elif H_norm < 0.90:
            entropy_level = "medium_high"
        else:
            entropy_level = "high"

        # --- Level semántico ---
        if H_norm < 0.40:
            level = "high"
        elif H_norm < 0.65:
            level = "medium"
        elif H_norm < 0.90:
            level = "medium"
        else:
            level = "low"

        report["level"] = level
        # ✅ NO pisamos tu conclusión fuerte
        report["level_description"] = CONCLUSION_BY_LEVEL[level]

        # Dominancia textual (y lo metemos como detalle, no con replace frágil)
        if dominance > 0.4:
            dominance_text = "marcadamente dominante"
        elif dominance >= 0.2:
            dominance_text = "predominante"
        else:
            dominance_text = "sin predominancia clara"

        report["details"].insert(0, f"El patrón aparece como {dominance_text}.")

        probabilistic_module = {
            "distribution": P_next,
            "entropy": H,
            "entropy_level": entropy_level,
            "dominance": dominance,
            "lambda_eff": mev_out["probabilistic"]["lambda_eff"],
        }

        metrics = {
            "T": mev_out["core"]["T"],
            "M": mev_out["core"]["M"],
            "M_prime": mev_out["core"]["M_prime"],
            "tau": mev_out["core"]["tau"],
        }

    except Exception:
        raise HTTPException(status_code=400, detail="Error al generar el informe protegido")

    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ProductionReady",
        "system": API_SYSTEM,
        "timestamp": now,
        "g_semantics": {
            "scale": "0-5",
            "meaning": "demora",
            "anchors": {"0": "inmediato", "5": "mucha demora"},
        },
        "vector_G": vector_G,
        "metrics": metrics,
        "probabilistic_module": probabilistic_module,
        "report": report,
        "trace": {
            "api_version": API_VERSION,
            "semantic_layer": "v1.0-closed",
            "core_version": mev_out["trace"]["core_version"],
            "implementation_id": mev_out["trace"]["implementation_id"],
            "hash_sha256": mev_out["trace"]["hash_sha256"],
            "dtype": mev_out["trace"]["dtype"],
            "K_sparse": mev_out["trace"]["K_sparse"],
        },
    }
