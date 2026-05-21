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
CONCLUSION_BY_LEVEL = {
    "high": "El patrón presenta una forma de actuación estable y coherente, orientada a la resolución de situaciones.",
    "medium": "Se configura una dinámica de actuación organizada, pero con variaciones a lo largo del proceso.",
    "low": "Esta configuración despliega una distribución variable entre múltiples tendencias, sin una predominancia sostenida."
}
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
@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    try:
        responses = req.payload.responses
        vector_G = [r.intensidad for r in responses]
        report = _protected_report_from_payload(responses)

        ORDER = ["A", "B", "C", "D", "E", "F"]
        KAPPA = {"S": 0.8, "C": 1.0, "X": 1.2}  
        K_SPARSE = 12  

        def one_hot(signo: str) -> List[float]:
            v = [0.0] * 6
            v[ORDER.index(signo)] = 1.0
            return v

        def vec_add(a: List[float], b: List[float]) -> List[float]:
            return [x + y for x, y in zip(a, b)]

        def vec_sub(a: List[float], b: List[float]) -> List[float]:
            return [x - y for x, y in zip(a, b)]

        def vec_scale(a: List[float], s: float) -> List[float]:
            return [x * s for x in a]

        def vec_div(a: List[float], s: float) -> List[float]:
            return [x / s for x in a]

        def norm2(a: List[float]) -> float:
            return math.sqrt(sum(x * x for x in a))

        n = len(responses)

        # 1) Energía efectiva y pesos:
        g_eff = []
        w = []
        v_list = []

        for r in responses:
            ge = float(r.intensidad) * float(KAPPA[r.tipo])  
            wi = 1.0 + float(req.alpha) * ge                 
            g_eff.append(ge)
            w.append(wi)
            v_list.append(one_hot(r.signo))

        # 2) Trayectoria T_i y estado final T:
        T_series: List[List[float]] = []
        num = [0.0] * 6
        den = 0.0

        for i in range(n):
            num = vec_add(num, vec_scale(v_list[i], w[i]))
            den += w[i]
            T_series.append(vec_div(num, den))

        T_final = T_series[-1]

        # 3) Movimiento M y M':
        if n <= 1:
            M = 0.0
        else:
            acc = 0.0
            for i in range(1, n):
                acc += norm2(vec_sub(T_series[i], T_series[i - 1]))
            M = acc / (n - 1)
        M_prime = M / 2.0 

        # 4) Dispersión D y G_norm 
        D = 1.0 - max(T_final) 
        G_norm = sum(r.intensidad for r in responses) / (5.0 * n)  
       
        # 5) Tensión tau:
        tau = float(req.beta[0]) * D + float(req.beta[1]) * G_norm + float(req.beta[2]) * M_prime  

        # 6) theta (distribución base suavizada):
        sum_w = sum(w)
        theta = []
        for k in ORDER:
            mass = sum(w[i] for i in range(n) if responses[i].signo == k)
            theta.append((mass + 1.0) / (sum_w + 6.0))

        # 7) Matriz de transición Pi con Laplace smoothing:
        counts_jk = {j: {k: 0 for k in ORDER} for j in ORDER}
        for i in range(1, n):
            prev_ = responses[i - 1].signo
            curr_ = responses[i].signo
            counts_jk[prev_][curr_] += 1

        Pi = {j: [] for j in ORDER}
        for j in ORDER:
            row_sum = sum(counts_jk[j].values())
            Pi[j] = [(counts_jk[j][k] + 1.0) / (row_sum + 6.0) for k in ORDER]

        # 8) lambda_eff anti-sparsity:
        m = n - 1
        lambda_eff = float(req.lambda_c) * (m / (m + K_SPARSE)) if m > 0 else 0.0  

        # 9) P_opcion (predicción próxima opción):
        last_signo = responses[-1].signo
        row_last = Pi[last_signo]
        P_opcion_next = [
            lambda_eff * row_last[i] + (1.0 - lambda_eff) * theta[i]
            for i in range(6)
        ]
        # 10) Entropía H en base 2:
        H = -sum(p * math.log(p, 2) for p in P_opcion_next if p > 0.0) 

        dominance = max(P_opcion_next)

        # -------- ENTROPY LEVEL (ahora H es log2) --------
        # (umbrales sugeridos: puedes ajustarlos luego; aquí dejamos 3 bandas simples)
        if H < 1.0:
            entropy_level = "low"
        elif H < 1.8:
            entropy_level = "medium"
        else:
            entropy_level = "high"

        # -------- LEVEL (basado en entropía H) --------
        if H < 1.0:
            level = "high"
        elif H < 1.8:
            level = "medium"
        else:
            level = "low"

        # Aplicar al reporte
        report["level"] = level
        report["conclusion"] = CONCLUSION_BY_LEVEL[level]

        probabilistic_module = {
            "distribution": P_opcion_next,  
            "entropy": H,                   
            "entropy_level": entropy_level,
            "dominance": dominance,
            "lambda_eff": lambda_eff,      
        }

        # -------- DOMINANCE TEXT --------
        if dominance > 0.4:
            dominance_text = "marcadamente dominante"
        elif dominance >= 0.2:
            dominance_text = "predominante"
        else:
            dominance_text = "sin predominancia clara"

        # -------- Summary dinámico --------
        report["summary"] = report["summary"].replace(
            "patrón predominante",
            f"patrón {dominance_text}",
        )

        # (Opcional) Completar metrics con el core MEV
        metrics = {
            "T": T_final,
            "M": M,
            "M_prime": M_prime,
            "tau": tau,
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Error al generar el informe protegido",
        )

    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ProductionReady",
        "system": API_SYSTEM,
        "timestamp": now,
        "vector_G": vector_G,
        "metrics": metrics,
        "probabilistic_module": probabilistic_module,
        "report": report,
        "trace": {
            "api_version": API_VERSION,
            "semantic_layer": "v1.0-closed",
            "core_version": "1.3",
            "K_sparse": 12,
        },
    }
