import numpy as np

from erp.qubo_model import QuboModel
from erp.solvers.base import QuboResult


def solve_brute_force(model: QuboModel) -> QuboResult:
    energies = model.energies_all_states()
    k = int(np.argmin(energies))
    bits = [(k >> i) & 1 for i in range(model.num_variables)]
    feasible = model.penalty_energy(bits) == 0
    return QuboResult(
        bits=bits if feasible else None,
        schedule=model.decode(bits) if feasible else None,
        energy=float(energies[k]),
        method=f"Brute force over 2^{model.num_variables} states",
    )
