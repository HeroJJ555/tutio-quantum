from dataclasses import dataclass, field

import dimod

from erp.qubo_model import QuboModel
from erp.schedule import Schedule


@dataclass
class QuboResult:
    bits: list[int] | None
    schedule: Schedule | None
    energy: float | None
    method: str
    stats: dict = field(default_factory=dict)


def to_bqm(model: QuboModel) -> dimod.BinaryQuadraticModel:
    q, offset = model.qubo()
    bqm = dimod.BinaryQuadraticModel.from_qubo(q, offset=offset)
    for i in range(model.num_variables):
        bqm.add_variable(i, 0.0)
    return bqm


def best_feasible(model: QuboModel, sampleset: dimod.SampleSet, method: str, stats: dict) -> QuboResult:
    reads = []
    for sample, occurrences in sampleset.data(["sample", "num_occurrences"]):
        values = {int(k): int(v) for k, v in sample.items()}  # VeloxQ returns labels as strings
        reads.append(([values[i] for i in range(model.num_variables)], int(occurrences)))
    total = sum(c for _, c in reads)
    feasible = [(bits, c, model.energy(bits)) for bits, c in reads if model.penalty_energy(bits) == 0]
    if not feasible:
        return QuboResult(None, None, None, method, {**stats, "reads": total, "feasible_fraction": 0.0})

    bits, _, energy = min(feasible, key=lambda r: r[2])
    return QuboResult(bits, model.decode(bits), energy, method, {
        **stats,
        "reads": total,
        "feasible_fraction": sum(c for _, c, _ in feasible) / total,
        "best_fraction": sum(c for _, c, e in feasible if abs(e - energy) < 1e-9) / total,
    })
