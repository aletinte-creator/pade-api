from typing import List, Dict
import math

class MEVResponse:
    def __init__(self, id, signo, intensidad, tipo):
        self.id = id
        self.signo = signo
        self.intensidad = intensidad
        self.tipo = tipo

def procesar_mev01_v13_rev(responses, alpha, beta, lambda_c):

    ORDER = ["A", "B", "C", "D", "E", "F"]
    counts = {k: 0 for k in ORDER}

    for r in responses:
        counts[r.signo] += 1

    total = len(responses)

    # Distribución simple (provisoria)
    P = [counts[k] / total for k in ORDER]

    # Entropía base 2
    H = -sum(p * math.log(p, 2) for p in P if p > 0)

    return {
        "core": {
            "T": [p for p in P],
            "M": 0.0,
            "M_prime": 0.0,
            "tau": 0.0,
        },
        "probabilistic": {
            "P_opcion_next": P,
            "H": H,
            "lambda_eff": lambda_c,
        },
        "trace": {
            "core_version": "1.3",
            "implementation_id": "mev01-basic",
            "hash_sha256": "test",
            "dtype": "float64",
            "K_sparse": 12,
        }
    }
