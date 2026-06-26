from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from mev01_core import MEVResponse, procesar_mev01_v13_rev
from collections import Counter
import math

def entropy(values):
    total = len(values)
    counts = Counter(values)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log(p, 2) for p in probs if p > 0)
    
def cobertura(responses):
    tipos = [r.tipo for r in responses]
    Hs = [r.H.lower() for r in responses]
    H_tipo = entropy(tipos)
    H_H = entropy(Hs)
    implicantes = sum(
        1 for r in responses if r.H.lower() in ["sabiendo", "te_enteras"]
    ) / len(responses)
    return H_tipo + H_H + implicantes
    
def coverage_text(cov):

    if cov < 1.5:
        return "La variedad de las situaciones es limitada, por lo que las conclusiones deben tomarse con cautela."

    elif cov < 2.5:
        return "Las situaciones presentan cierta diversidad, aunque no cubren completamente todos los aspectos relevantes."

    elif cov < 3.5:
        return "La variedad de las situaciones y su nivel de implicación permiten observar patrones con consistencia."

    elif cov < 4.5:
        return "El análisis se basa en una amplia variedad de situaciones, lo que refuerza la solidez del resultado."

    else:
        return "El análisis cubre una alta diversidad de situaciones y niveles de implicación, fortaleciendo la validez del resultado."
    
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
    H: str

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
    - No expone letras A–F como letras (usa frases humanas).
    - No expone G como lista (usa buckets: inmediato / equilibrio / demora).
    - No expone S/C/X como códigos (usa: simples / complicadas / comprometidas).
    """

    ORDER = ["A", "B", "C", "D", "E", "F"]

    phrase = {
        "A": "actuar de forma directa sobre la situación",
        "B": "organizar la situación mediante ajustes",
        "C": "mantener la situación sin intervención activa",
        "D": "retirarse de la situación sin resolución",
        "E": "adaptarse personalmente a las condiciones",
        "F": "responder a desgano",
    }

    tipo_label = {
        "S": "situaciones simples",
        "C": "situaciones complicadas",
        "X": "situaciones comprometidas",
    }
    payload: RunPayload

    def _timing_bucket(mean_g: float) -> str:
        # G = demora: 0 inmediato .. 5 mucha demora
        if mean_g <= 1.5:
            return "con respuestas mayormente inmediatas"
        elif mean_g <= 3.5:
            return "con un equilibrio entre reflexión y acción"
        else:
            return "con mayor demora para responder"

    def _dominant_signo(subset: List[ResponseItem]) -> str | None:
        if not subset:
            return None
        counts = {k: 0 for k in ORDER}
        for r in subset:
            counts[r.signo] += 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], ORDER.index(kv[0])))
        return top[0][0]

    # --- 1) patrón global (primary/secondary) + contadores ---
    global_counts = {k: 0 for k in ORDER}
    g_vals: List[int] = []
    t_counts = {"S": 0, "C": 0, "X": 0}

    for r in responses:
        global_counts[r.signo] += 1
        g_vals.append(r.intensidad)
        t_counts[r.tipo] += 1

    top_global = sorted(global_counts.items(), key=lambda kv: (-kv[1], ORDER.index(kv[0])))
    primary = top_global[0][0]
    secondary = top_global[1][0]

    # --- 2) patrón por tipo S/C/X ---
    by_tipo = {
        "S": [r for r in responses if r.tipo == "S"],
        "C": [r for r in responses if r.tipo == "C"],
        "X": [r for r in responses if r.tipo == "X"],
    }

    primary_S = _dominant_signo(by_tipo["S"])
    primary_C = _dominant_signo(by_tipo["C"])
    primary_X = _dominant_signo(by_tipo["X"])

    # --- 3) timing global y por tipo ---
    mean_g = sum(g_vals) / max(1, len(g_vals))
    timing_global = _timing_bucket(mean_g)

    def _timing_bucket_for_tipo(tipo: str) -> str | None:
        vals = [r.intensidad for r in by_tipo[tipo]]
        if not vals:
            return None
        return _timing_bucket(sum(vals) / len(vals))

    timing_S = _timing_bucket_for_tipo("S")
    timing_C = _timing_bucket_for_tipo("C")
    timing_X = _timing_bucket_for_tipo("X")

    # --- 4) contexto global (sin exponer S/C/X) ---
    max_t = max(t_counts["S"], t_counts["C"], t_counts["X"])
    if t_counts["S"] == max_t and t_counts["S"] >= t_counts["C"] and t_counts["S"] >= t_counts["X"]:
        context_txt = "situaciones de baja exigencia"
    elif t_counts["C"] == max_t and t_counts["C"] >= t_counts["S"] and t_counts["C"] >= t_counts["X"]:
        context_txt = "situaciones con complejidad moderada"
    else:
        context_txt = "situaciones de mayor exigencia o implicación"

    # --- 5) details base (sin duplicar insights avanzados) ---
    details: List[str] = []

    details.append(
        f"No respondés al azar: tendés a {phrase[primary]}, incluso cuando las situaciones cambian."
    )
    details.append(
        f"Aunque no es lo único que hacés: también aparece {phrase[secondary]}."
    )

    if primary_S:
        extra = f" ({timing_S})" if timing_S else ""
        details.append(f"En {tipo_label['S']}, tendés a {phrase[primary_S]}{extra}.")
    if primary_C:
        extra = f" ({timing_C})" if timing_C else ""
        details.append(f"En {tipo_label['C']}, tendés a {phrase[primary_C]}{extra}.")
    if primary_X:
        extra = f" ({timing_X})" if timing_X else ""
        details.append(f"En {tipo_label['X']}, tendés a {phrase[primary_X]}{extra}.")

    details.append(
        f"La clave no es solo la decisión: es el tiempo. En general respondés {timing_global}."
    )
    details.append("No es ocasional. Es consistente a lo largo del proceso.")
    details.append(f"Y ese patrón se mantiene incluso ante {context_txt}.")

    # --- 6) MOTOR DE INSIGHTS COMBINADOS (patrón + demora + tipo) ---
    def _mean_g(subset: List[ResponseItem]) -> float | None:
        if not subset:
            return None
        return sum(r.intensidad for r in subset) / len(subset)

    mean_S = _mean_g(by_tipo["S"])
    mean_C = _mean_g(by_tipo["C"])
    mean_X = _mean_g(by_tipo["X"])

    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return a - b

    dxs = _delta(mean_X, mean_S)  # >0 => más demora en X que en S
    dcs = _delta(mean_C, mean_S)
    dxc = _delta(mean_X, mean_C)

    insights: List[str] = []

    # Cambio de patrón cuando sube exigencia (S -> X)
    if primary_S and primary_X:
        if primary_S != primary_X:
            insights.append(
                f"Cuando sube la exigencia, cambia tu reacción: en situaciones simples tendés a {phrase[primary_S]}, pero en situaciones comprometidas tendés a {phrase[primary_X]}."
            )
        else:
            insights.append(
                f"Tu reacción se mantiene estable incluso bajo presión: repetís {phrase[primary_S]} tanto en situaciones simples como en comprometidas."
            )

    # Demora aumenta/disminuye con la exigencia
    if dxs is not None and dxs >= 0.9:
        insights.append("Cuando la situación se vuelve comprometida, tu tiempo de respuesta tiende a alargarse.")
    if dxs is not None and dxs <= -0.9:
        insights.append("Cuando la situación se vuelve comprometida, tendés a responder más rápido que en lo simple.")

    # Caso especial: F en X (desgano bajo presión)
    if primary_X == "F":
        if timing_X == "con mayor demora para responder":
            insights.append("En situaciones comprometidas aparece una combinación peligrosa: respondés a desgano y, además, tardás.")
        else:
            insights.append("En situaciones comprometidas tendés a responder a desgano: no es falta de decisión, es falta de sostén.")

    # Diferencia X vs C
    if dxc is not None and dxc >= 0.9:
        insights.append("Entre lo complicado y lo comprometido hay un salto: en lo comprometido se te hace más cuesta arriba responder a tiempo.")

    # Consistencia temporal 
    if dxs is not None and abs(dxs) < 0.5 and dcs is not None and abs(dcs) < 0.5:
        insights.append("Tu tiempo de respuesta es bastante estable: no cambia demasiado según el tipo de situación.")

    if insights:
        details = insights + details

    return {
        "level": "medium",  
        "summary": f"Hay un patrón que se repite: tendés a {phrase[primary]}.",
        "details": details,
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
        coverage = cobertura(responses)
        
        vector_G = [r.intensidad for r in responses]  # G = demora

        report = _protected_report_from_payload(responses)
        cov_text = coverage_text(coverage)
        report["details"].append(cov_text)

        # --- Llamada al CORE MEV01 v1.3 (ya alineado con G=demora) ---
        mev_responses = [
            MEVResponse(
                id=r.id,
                signo=r.signo,
                intensidad=r.intensidad,
                tipo=r.tipo,
                H=r.H   # ← CLAVE
            )
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
        
        report["level_description"] = CONCLUSION_BY_LEVEL[level]


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
        "coverage": coverage,
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
