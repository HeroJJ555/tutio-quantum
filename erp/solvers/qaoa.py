import numpy as np
from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from erp.qubo_model import QuboModel
from erp.solvers import QuboResult


def qubo_to_ising(q: dict[tuple[int, int], float], num_qubits: int) -> SparsePauliOp:
    """Ising operator for x = (1 - Z) / 2; the constant term is dropped."""
    terms = []
    for (i, j), v in q.items():
        if i == j:  # v * x_i = v/2 - v/2 Z_i
            terms.append(("Z", [i], -v / 2))
        else:  # v * x_i x_j = v/4 (1 - Z_i - Z_j + Z_i Z_j)
            terms += [("Z", [i], -v / 4), ("Z", [j], -v / 4), ("ZZ", [i, j], v / 4)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits).simplify()


def solve_qaoa(model: QuboModel, reps: int = 2, restarts: int = 8, shots: int = 4000,
               maxiter: int = 300, seed: int = 0) -> QuboResult:
    n = model.num_variables
    q, _ = model.qubo()
    cost = qubo_to_ising(q, n)
    scale = float(np.max(np.abs(cost.coeffs)))
    cost = cost / scale  # so that useful gammas lie in ~[0, pi]

    ansatz = qaoa_ansatz(cost, reps=reps)
    ansatz = transpile(ansatz, basis_gates=["h", "rx", "rz", "rzz", "cx"], optimization_level=1)
    gammas = [p for p in ansatz.parameters if "γ" in p.name or "gamma" in p.name]
    betas = [p for p in ansatz.parameters if p not in gammas]

    energies = model.energies_all_states()

    def bind(x):
        return ansatz.assign_parameters(dict(zip(gammas + betas, x)))

    def expectation(x):
        probs = Statevector(bind(x)).probabilities()
        return float(probs @ energies)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, reps), rng.uniform(0, np.pi / 2, reps)])
        res = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    circuit = bind(best.x)
    circuit.measure_all()
    counts = StatevectorSampler(seed=seed).run([circuit], shots=shots).result()[0].data.meas.get_counts()

    optimum_energy = float(np.min(energies))
    best_bits, best_energy, feasible_shots, optimal_shots = None, None, 0, 0
    for key, c in counts.items():
        bits = [int(b) for b in reversed(key)]  # qubit 0 is the rightmost character
        if model.penalty_energy(bits) != 0:
            continue
        feasible_shots += c
        e = model.energy(bits)
        if np.isclose(e, optimum_energy):
            optimal_shots += c
        if best_energy is None or e < best_energy:
            best_bits, best_energy = bits, e

    return QuboResult(
        bits=best_bits,
        schedule=model.decode(best_bits) if best_bits else None,
        energy=best_energy,
        method=f"QAOA p={reps} (Qiskit state-vector simulator), {model.label}",
        stats={
            "reps": reps,
            "shots": shots,
            "restarts": restarts,
            "seed": seed,
            "expected_energy": float(best.fun),
            "gammas": [float(v) for v in best.x[:reps]],
            "betas": [float(v) for v in best.x[reps:]],
            "p_feasible": feasible_shots / shots,
            "p_optimal": optimal_shots / shots,
        },
    )
