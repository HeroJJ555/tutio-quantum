import dimod

try:
    from dwave.samplers import SimulatedAnnealingSampler
except ImportError:
    from neal import SimulatedAnnealingSampler

from erp.qubo_model import QuboModel
from erp.solvers import QuboResult


def solve_simulated_annealing(model: QuboModel, num_reads: int = 1000, seed: int = 0) -> QuboResult:
    q, offset = model.qubo()
    bqm = dimod.BinaryQuadraticModel.from_qubo(q, offset=offset)
    # A variable with no terms would otherwise be missing from the samples.
    for i in range(model.num_variables):
        bqm.add_variable(i, 0.0)
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=seed)

    best_bits, best_energy, feasible_reads = None, None, 0
    for sample, energy, occurrences in sampleset.data(["sample", "energy", "num_occurrences"]):
        bits = [int(sample[i]) for i in range(model.num_variables)]
        if model.penalty_energy(bits) != 0:
            continue
        feasible_reads += occurrences
        if best_energy is None or energy < best_energy:
            best_bits, best_energy = bits, float(energy)

    return QuboResult(
        bits=best_bits,
        schedule=model.decode(best_bits) if best_bits else None,
        energy=best_energy,
        method=f"Simulated annealing (D-Wave Ocean), {model.label}",
        stats={"num_reads": num_reads, "feasible_fraction": float(feasible_reads / num_reads), "seed": seed},
    )
