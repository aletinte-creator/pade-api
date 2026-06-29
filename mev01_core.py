import math
from dataclasses import dataclass
from typing import List, Optional


# =========================
# CONSTANTES
# =========================

ORDER = ["A", "B", "C", "D", "E", "F"]
G_MAX = 5

KAPPA = {
    "S": 1.0,
    "C": 1.1,
    "X": 1.2
}

KAPPA_H = {
    "culpa": 0.9,
    "presion": 1.1,
    "vinculo": 1.0
}

HOOK_BIAS = {
    "culpa": {"C": 0.10, "F": 0.10},
    "presion": {"D": 0.12},
    "vinculo": {"E": 0.12}
}


# =========================
# MODELO
# =========================

@dataclass(frozen=True)
class MEVResponse:
    id: int
    signo: str
    intensidad: int
    tipo: str
    H: Optional[str] = None


# =========================
# HELPERS
# =========================

def _one_hot(signo: str):
    v = [0.0] * 6
    v[ORDER.index(signo)] = 1.0
    return v


def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]


def _vec_scale(v, s):
    return [x * s for x in v]


def _vec_div(v, d):
    return [x / d for x in v]


def _vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _norm2(v):
    return math.sqrt(sum(x * x for x in v))


# =========================
# CORE MEV
# =========================

def procesar_mev01_v13_rev(
    *,
    responses: List[MEVResponse],
    alpha: float,
    beta: List[float],
    lambda_c: float,
):

    n = len(responses)
    if n == 0:
        raise ValueError("responses no puede estar vacío")

    w = []
    v_list = []

    # -------------------------
    # PESOS (H integrado aquí)
    # -------------------------
    for r in responses:
        if r.intensidad < 0 or r.intensidad > G_MAX:
            raise ValueError("intensidad fuera de rango 0..5")

        r_i = float(G_MAX - r.intensidad)

        h_val = (getattr(r, "H", None) or "").lower()
        k_h = KAPPA_H.get(h_val, 1.0)

        r_eff = r_i * KAPPA[r.tipo] * k_h
        w_i = 1.0 + float(alpha) * r_eff

        w.append(w_i)
        v_list.append(_one_hot(r.signo))

    # -------------------------
    # TRAYECTORIA
    # -------------------------
    T_series = []
    num = [0.0] * 6
    den = 0.0

    for i in range(n):
        num = _vec_add(num, _vec_scale(v_list[i], w[i]))
        den += w[i]
        T_series.append(_vec_div(num, den))

    T_final = T_series[-1]

    # -------------------------
    # MOVIMIENTO
    # -------------------------
    if n <= 1:
        M = 0.0
    else:
        acc = 0.0
        for i in range(1, n):
            acc += _norm2(_vec_sub(T_series[i], T_series[i - 1]))
        M = acc / (n - 1)

    M_prime = M / 2.0

    # -------------------------
    # DISPERSIÓN Y G
    # -------------------------
    D = 1.0 - max(T_final)
    G_norm = sum(r.intensidad for r in responses) / (float(G_MAX) * n)

    # -------------------------
    # TENSIÓN
    # -------------------------
    tau = (
        beta[0] * D +
        beta[1] * G_norm +
        beta[2] * M_prime
    )

    # -------------------------
    # THETA
    # -------------------------
    sum_w = sum(w)
    theta = []

    for k in ORDER:
        mass = sum(w[i] for i in range(n) if responses[i].signo == k)
        theta.append((mass + 1.0) / (sum_w + 6.0))

    # -------------------------
    # TRANSICIÓN (SIMPLIFICADA)
    # -------------------------
    last_signo = responses[-1].signo
    Pi_last_row = theta

    lambda_eff = float(lambda_c)

    # -------------------------
    # DISTRIBUCIÓN BASE
    # -------------------------
    P_opcion_next = [
        lambda_eff * Pi_last_row[i] + (1.0 - lambda_eff) * theta[i]
        for i in range(6)
    ]

    # =========================
    # ✅ BIAS POR H (AQUÍ)
    # =========================
    hook_count = sum(
        1 for r in responses
        if (getattr(r, "H", None) or "").lower() in HOOK_BIAS
    )

    if hook_count > 0:
        for r in responses:
            h_val = (getattr(r, "H", None) or "").lower()

            if h_val in HOOK_BIAS:
                bias = HOOK_BIAS[h_val]

                for signo, val in bias.items():
                    idx = ORDER.index(signo)
                    P_opcion_next[idx] += val / hook_count

    # -------------------------
    # NORMALIZACIÓN
    # -------------------------
    total = sum(P_opcion_next)
    if total > 0:
        P_opcion_next = [p / total for p in P_opcion_next]

    # -------------------------
    # ENTROPÍA
    # -------------------------
    H = -sum(p * math.log(p, 2) for p in P_opcion_next if p > 0.0)

    # -------------------------
    # OUTPUT
    # -------------------------
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
            "H": H,
            "lambda_eff": lambda_eff,
        }
    }
