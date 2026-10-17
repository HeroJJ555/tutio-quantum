from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from erp.qubo_model import Coefficients, QuboModel
from erp.solvers.base import QuboResult


def qubo_to_ising(q: Coefficients, num_qubits: int) -> SparsePauliOp:
    terms = []
    for (i, j), v in q.items():
        if i == j:  # v * x_i = v/2 - v/2 Z_i
            terms.append(("Z", [i], -v / 2))
        else:  # v * x_i x_j = v/4 (1 - Z_i - Z_j + Z_i Z_j)
            terms += [("Z", [i], -v / 4), ("Z", [j], -v / 4), ("ZZ", [i, j], v / 4)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits).simplify()


@dataclass
class Ansatz:
    circuit: QuantumCircuit
    reps: int
    parameters: list[Parameter]

    @classmethod
    def for_model(cls, model: QuboModel, reps: int) -> "Ansatz":
        cost = qubo_to_ising(model.qubo()[0], model.num_variables)
        cost = cost / float(np.max(np.abs(cost.coeffs)))
        circuit = transpile(qaoa_ansatz(cost, reps=reps), basis_gates=["h", "rx", "rz", "rzz", "cx"],
                            optimization_level=1)
        gammas = [p for p in circuit.parameters if "γ" in p.name or "gamma" in p.name]
        betas = [p for p in circuit.parameters if p not in gammas]
        return cls(circuit, reps, gammas + betas)

    def bind(self, gammas_then_betas: np.ndarray) -> QuantumCircuit:
        return self.circuit.assign_parameters(dict(zip(self.parameters, gammas_then_betas, strict=True)))


def optimize_angles(ansatz: Ansatz, energies: np.ndarray, restarts: int, maxiter: int, seed: int):
    def expectation(x):
        return float(Statevector(ansatz.bind(x)).probabilities() @ energies)

    rng = np.random.default_rng(seed)
    starts = [np.concatenate([rng.uniform(0, np.pi, ansatz.reps), rng.uniform(0, np.pi / 2, ansatz.reps)])
              for _ in range(restarts)]
    runs = [minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter}) for x0 in starts]
    return min(runs, key=lambda r: r.fun)


def sample_counts(circuit: QuantumCircuit, shots: int, seed: int) -> dict[tuple[int, ...], int]:
    measured = circuit.copy()
    measured.measure_all()
    counts = StatevectorSampler(seed=seed).run([measured], shots=shots).result()[0].data.meas.get_counts()
    return {tuple(int(b) for b in reversed(key)): c for key, c in counts.items()}  # qubit 0 is the last char


def solve_qaoa(model: QuboModel, reps: int = 2, restarts: int = 8, shots: int = 4000,
               maxiter: int = 300, seed: int = 0) -> QuboResult:
    ansatz = Ansatz.for_model(model, reps)
    energies = model.energies_all_states()
    best = optimize_angles(ansatz, energies, restarts, maxiter, seed)
    counts = sample_counts(ansatz.bind(best.x), shots, seed)

    feasible = {bits: (c, model.energy(bits)) for bits, c in counts.items() if model.penalty_energy(bits) == 0}
    optimum = float(np.min(energies))
    best_bits = min(feasible, key=lambda b: feasible[b][1], default=None)
    return QuboResult(
        bits=list(best_bits) if best_bits else None,
        schedule=model.decode(best_bits) if best_bits else None,
        energy=feasible[best_bits][1] if best_bits else None,
        method=f"QAOA p={reps} (Qiskit state-vector simulator), {model.label}",
        stats={
            "reps": reps,
            "shots": shots,
            "restarts": restarts,
            "seed": seed,
            "expected_energy": float(best.fun),
            "gammas": [float(v) for v in best.x[:reps]],
            "betas": [float(v) for v in best.x[reps:]],
            "p_feasible": sum(c for c, _ in feasible.values()) / shots,
            "p_optimal": sum(c for c, e in feasible.values() if np.isclose(e, optimum)) / shots,
        },
    )
