from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from mev01_core import MEVResponse, procesar_mev01_v13_rev
from collections import Counter

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
    
def inconsistencia_H(responses):

    cambios = 0      # inconsistencia estructural (signo)
    tensiones = 0    # inconsistencia temporal (G)

    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):

            r1 = responses[i]
            r2 = responses[j]

            # mismo tipo, distinto H
            if r1.tipo == r2.tipo and r1.H != r2.H:

                if r1.signo != r2.signo:
                    cambios += 1

                elif abs(r1.intensidad - r2.intensidad) >= 2:
                    tensiones += 1

    total = cambios + 0.5 * tensiones

    return {
        "total": total,
        "cambios": cambios,
        "tensiones": tensiones
    }
    
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

def inconsistencia_text(I):

    cambios = I["cambios"]
    tensiones = I["tensiones"]

    # muy estable
    if cambios == 0 and tensiones == 0:
        return "La manera en que resuelves los dilemas resulta estable ante diferentes situaciones."

    # cambios estructurales leves / moderados / fuertes
    if cambios > 0 and tensiones == 0:
        if cambios < 3:
            return "Se observan ciertos cambios en la forma en que respondes, según la situación."
        elif cambios < 6:
            return "Tu forma de responder varía de manera significativa según cómo se presentan las situaciones."
        else:
            return "Tu manera de resolver cambia de forma marcada siempre ante distintas situaciones."

    # tensión temporal
    if cambios == 0 and tensiones > 0:
        if tensiones < 3:
            return "Tus decisiones se mantienen, aunque el tiempo de respuesta varía en algunas situaciones."
        else:
            return "Tu decisión se mantiene, pero el tiempo de respuesta cambia consistentemente según la situación."

    # mezcla de todo
    if cambios > 0 and tensiones > 0:
        return "Tus decisiones muestran cambios tanto en la forma de actuar como en el tiempo de respuesta según la situación."

    return "Se observan persistentes variaciones en tu forma de responder."
    
API_SYSTEM = "PADE 1.1"
API_VERSION = "0.1.0"

# Descripción institucional por nivel (capa seria)
CONCLUSION_BY_LEVEL = {
    "high": "Este patrón presenta un estilo de actuación estable y coherente, orientada a la resolución de situaciones.",
    "medium": "Se observa una dinámica de actuación organizada, pero con variaciones a lo largo del proceso.",
    "low": "Esta configuración actual, despliega una distribución variable entre múltiples tendencias, sin una predominancia sostenida.",
}

Signo = Literal["A", "B", "C", "D", "E", "F"]
TipoSituacion = Literal["S", "C", "X"]


from typing import Optional
from pydantic import BaseModel, Field, field_validator

class ResponseItem(BaseModel):
    id: int = Field(..., ge=1)
    signo: Signo
    intensidad: int = Field(..., ge=0, le=5)
    tipo: TipoSituacion
    H: Optional[str] = None

    class Config:
        extra = "ignore"

    @field_validator("H")
    @classmethod
    def validate_H(cls, v):
        if v is None:
            return v
        allowed = {"culpa", "presion", "vinculo"}
        if v not in allowed:
            raise ValueError("H inválido")
        return v

class RunPayload(BaseModel):
    responses: List[ResponseItem]

    @model_validator(mode="after")
    def validate_sequence(self) -> "RunPayload":
        n = len(self.responses)
        if n not in (12, 14, 20):
            raise ValueError("responses debe tener longitud 12, 14 o 20")
        last_id = None
        for r in self.responses:
            if last_id is not None and r.id <= last_id:
                raise ValueError("IDs deben ser estrictamente crecientes")
            last_id = r.id
        return self
        
from pydantic import BaseModel, Field, validator
from typing import List

class RunRequest(BaseModel):
    payload: RunPayload
    use_hooks: bool = False
    alpha: float = Field(..., ge=0.0)
    beta: List[float] = Field(..., min_length=3, max_length=3)
    lambda_c: float = Field(..., ge=0.0, le=1.0)

    @validator("beta")
    def validate_beta(cls, v):
        if len(v) != 3:
            raise ValueError("beta debe tener 3 valores")
        if abs(sum(v) - 1.0) > 1e-9:
            raise ValueError("beta debe sumar 1.0")
        if not (v[0] >= v[2] >= v[1]):
            raise ValueError("beta debe cumplir beta[0] >= beta[2] >= beta[1]")
        return v

    @validator("payload")
    def validate_hooks(cls, payload, values):
        use_hooks = values.get("use_hooks", False)

        if use_hooks:
            for r in payload.responses:
                if r.tipo == "S" and not r.H:
                    raise ValueError("Los dilemas tipo S requieren H cuando use_hooks=True")

        return payload
        
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
# --- 7) conclusión dinámica (usa inconsistencia) ---
    I = inconsistencia_H(responses)
    cambios = I["cambios"]
    tensiones = I["tensiones"]

    if cambios == 0 and tensiones == 0:
        conclusion = "Tu forma de actuar se mantiene consistente en distintas situaciones."

    elif cambios > 0 and tensiones == 0:
        conclusion = "La forma en que actuás varía según cómo se presentan las situaciones."

    elif cambios == 0 and tensiones > 0:
        conclusion = "La decisión se mantiene, pero el tiempo de respuesta varía según la situación."

    else:
        conclusion = "Tanto la forma de actuar como el tiempo de respuesta cambian según la situación."

    return {
        "level": "medium",
        "summary": f"Hay un patrón que se repite: tendés a {phrase[primary]}.",
        "details": details,
        "conclusion": conclusion
    }
app = FastAPI(title="PADE 1.1 API", version=API_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # después restringir
    allow_credentials=True, 
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
        I_H = inconsistencia_H(responses)
        I_H_text = inconsistencia_text(I_H)

        vector_G = [r.intensidad for r in responses]

        report = _protected_report_from_payload(responses)

        cov_text = coverage_text(coverage)
        report["details"].append(cov_text)
        report["details"].append(I_H_text)

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
        raise HTTPException(status_code=400, detail=str(e))


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
