from dwave.samplers import SimulatedAnnealingSampler

from erp.qubo_model import QuboModel
from erp.solvers.base import QuboResult, best_feasible, to_bqm


def solve_simulated_annealing(model: QuboModel, num_reads: int = 1000, seed: int = 0) -> QuboResult:
    sampleset = SimulatedAnnealingSampler().sample(to_bqm(model), num_reads=num_reads, seed=seed)
    return best_feasible(model, sampleset, f"Simulated annealing (D-Wave Ocean), {model.label}", {"seed": seed})
