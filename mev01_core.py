from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict

Signo = Literal["A", "B", "C", "D", "E", "F"]
Tipo = Literal["S", "C", "X"]

ORDER: List[Signo] = ["A", "B", "C", "D", "E", "F"]

# κ(S)=0.8, κ(C)=1.0, κ(X)=1.2
KAPPA: Dict[Tipo, float] = {"S": 0.8, "C": 1.0, "X": 1.2}
KAPPA_H: Dict[str, float] = {
    "escuchaste": 0.8,
    "viste": 0.9,
    "entendiste": 1.0,
    "aprendiste": 1.05,
    "te_enteras": 1.2,
    "sabiendo": 1.3
}

K_SPARSE = 12  # anti-sparsity
G_MAX = 5      # G ∈ [0,5] donde 0=inmediato, 5=mucha demora


from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MEVResponse:
    id: int
    signo: Signo
    intensidad: int
    tipo: Tipo
    H: Optional[str] = None   ✅
    
class CoreOut(TypedDict):
    T: List[float]
    M: float
    M_prime: float
    tau: float


class ProbOut(TypedDict):
    theta: List[float]
    Pi_last_row: List[float]
    P_opcion_next: List[float]
    P_energia_given_O: Dict[str, List[float]]
    H: float
    lambda_eff: float


class TraceOut(TypedDict):
    core_version: str
    implementation_id: str
    hash_sha256: str
    dtype: str
    K_sparse: int


class MEVOut(TypedDict):
    core: CoreOut
    probabilistic: ProbOut
    trace: TraceOut


def _one_hot(signo: Signo) -> List[float]:
    v = [0.0] * 6
    v[ORDER.index(signo)] = 1.0
    return v


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


def _vec_div(a: List[float], s: float) -> List[float]:
    return [x / s for x in a]


def _norm2(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _normalize_for_hash(
    responses: List[MEVResponse],
    alpha: float,
    beta: List[float],
    lambda_c: float,
) -> str:
    # Normalización estable para trazabilidad determinista
    parts = [
        f"alpha={alpha:.12g}",
        f"beta={beta[0]:.12g},{beta[1]:.12g},{beta[2]:.12g}",
        f"lambda_c={lambda_c:.12g}",
        "kappa=S:0.8,C:1.0,X:1.2",
        "G_semantics=demora_0_inmediato_5_tarde",
    ]
    for r in responses:
        parts.append(f"{r.id}:{r.signo}:{r.intensidad}:{r.tipo}:{r.H}")
    return "|".join(parts)


def procesar_mev01_v13_rev(
    *,
    responses: List[MEVResponse],
    alpha: float,
    beta: List[float],
    lambda_c: float,
) -> MEVOut:
    n = len(responses)
    if n == 0:
        raise ValueError("responses no puede estar vacío")

    # 1) Peso por RAPIDEZ (no por demora):
    # G = demora (0 inmediato, 5 tarde)
    # R = rapidez = (5 - G)
    # R_eff = R * κ(tipo) ; w = 1 + α * R_eff
    w: List[float] = []
    v_list: List[List[float]] = []

    for r in responses:
        if r.intensidad < 0 or r.intensidad > G_MAX:
            raise ValueError("intensidad (G) fuera de rango 0..5")

        r_i = float(G_MAX - r.intensidad)       # rapidez

        h_val = (getattr(r, "H", None) or "").lower()
        k_h = KAPPA_H.get(h_val, 1.0)
        r_eff = r_i * KAPPA[r.tipo] * k_h       # ← MODIFICADO

        w_i = 1.0 + float(alpha) * r_eff

        w.append(w_i)
        v_list.append(_one_hot(r.signo))

    # 2) Trayectoria T_i y estado final T:
    # T_i = (∑ w_k v_k)/(∑ w_k)
    T_series: List[List[float]] = []
    num = [0.0] * 6
    den = 0.0
    for i in range(n):
        num = _vec_add(num, _vec_scale(v_list[i], w[i]))
        den += w[i]
        T_series.append(_vec_div(num, den))
    T_final = T_series[-1]

    # 3) Movimiento M y M′:
    # M = (1/(n-1)) ∑ ||T_i - T_{i-1}|| ; M′ = M/2
    if n <= 1:
        M = 0.0
    else:
        acc = 0.0
        for i in range(1, n):
            acc += _norm2(_vec_sub(T_series[i], T_series[i - 1]))
        M = acc / (n - 1)
    M_prime = M / 2.0

    # 4) Dispersión D y demora normalizada:
    # D = 1 - max(T)
    # G_norm = (∑ G_i)/(5n) con G como DEMORA
    D = 1.0 - max(T_final)
    G_norm = sum(r.intensidad for r in responses) / (float(G_MAX) * n)

    # 5) Tensión τ:
    # τ = β1*D + β2*G_norm + β3*M′
    tau = float(beta[0]) * D + float(beta[1]) * G_norm + float(beta[2]) * M_prime

    # 6) Distribución base θ:
    # θ_k = (∑_{i:O_i=k} w_i + 1) / (∑ w_i + 6)
    sum_w = sum(w)
    theta: List[float] = []
    for k in ORDER:
        mass = sum(w[i] for i in range(n) if responses[i].signo == k)
        theta.append((mass + 1.0) / (sum_w + 6.0))

    # 7) Matriz de transición Π:
    # Π(j,k) = (count(j,k)+1)/(∑_k count(j,k)+6)
    counts_jk = {j: {k: 0 for k in ORDER} for j in ORDER}
    for i in range(1, n):
        prev_ = responses[i - 1].signo
        curr_ = responses[i].signo
        counts_jk[prev_][curr_] += 1

    Pi = {j: [] for j in ORDER}
    for j in ORDER:
        row_sum = sum(counts_jk[j].values())
        Pi[j] = [(counts_jk[j][k] + 1.0) / (row_sum + 6.0) for k in ORDER]

    # 8) λ_eff anti-sparsity:
    # λ_eff = λ_c * (m/(m+K)), m=n-1
    m = n - 1
    lambda_eff = float(lambda_c) * (m / (m + K_SPARSE)) if m > 0 else 0.0

    # 9) P_opcion_next:
    # P = λ_eff * Π(last,·) + (1-λ_eff) * θ
    last_signo = responses[-1].signo
    Pi_last_row = Pi[last_signo]
    P_opcion_next = [
        lambda_eff * Pi_last_row[i] + (1.0 - lambda_eff) * theta[i]
        for i in range(6)
    ]
    # --- Bias por gancho (H) ---
   # --- Bias por ganchos (H) ---
HOOK_BIAS = {
    "culpa": {"C": 0.10, "F": 0.10},
    "presion": {"D": 0.12},
    "vinculo": {"E": 0.12}
}
ORDER = ["A", "B", "C", "D", "E", "F"]
# contar ganchos válidos
hook_count = sum(
    1 for r in responses
    if (getattr(r, "H", None) or "").lower() in HOOK_BIAS
)
# aplicar bias
if hook_count > 0:
    for r in responses:
        h_val = (getattr(r, "H", None) or "").lower()

        if h_val in HOOK_BIAS:
            bias = HOOK_BIAS[h_val]

            for signo, val in bias.items():
                idx = ORDER.index(signo)
                P_opcion_next[idx] += val / hook_count
                
# --- Normalizar distribución ---
total = sum(P_opcion_next)
if total > 0:
    P_opcion_next = [p / total for p in P_opcion_next]
    # 10) Entropía H en base 2:
    H = -sum(p * math.log(p, 2) for p in P_opcion_next if p > 0.0)

    # 11) P(G=g | O=k) con smoothing:
    # (count(k,g)+1)/(∑_g count(k,g)+6)  con g∈{0..5}
    counts_kg = {k: {g: 0 for g in range(G_MAX + 1)} for k in ORDER}
    for r in responses:
        counts_kg[r.signo][r.intensidad] += 1

    P_energia_given_O: Dict[str, List[float]] = {}
    for k in ORDER:
        denom = sum(counts_kg[k].values()) + 6.0
        P_energia_given_O[k] = [(counts_kg[k][g] + 1.0) / denom for g in range(G_MAX + 1)]

    # 12) Trace (hash):
    normalized = _normalize_for_hash(responses, alpha, beta, lambda_c)
    hash_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return {
        "core": {
            "T": T_final,
            "M": M,
            "M_prime": M_prime,
            "tau": tau,
        },
        "probabilistic": {
            "theta": theta,
            "Pi_last_row": Pi_last_row,
            "P_opcion_next": P_opcion_next,
            "P_energia_given_O": P_energia_given_O,
            "H": H,
            "lambda_eff": lambda_eff,
        },
        "trace": {
            "core_version": "1.3",
            "implementation_id": "mev01-v1.3-ref",
            "hash_sha256": hash_sha256,
            "dtype": "float64",
            "K_sparse": K_SPARSE,
        },
    }
