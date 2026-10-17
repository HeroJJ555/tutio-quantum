import numpy as np

from erp.instance import Instance, Operation


def random_instance(n_jobs: int, seed: int, machines: str = "TFS", max_duration: int = 3,
                    max_weight: int = 3, tightness: float = 0.5) -> Instance:
    rng = np.random.default_rng(seed)
    durations = rng.integers(1, max_duration + 1, size=(n_jobs, len(machines)))
    max_slack = max(1, int(tightness * durations.sum() / len(machines)))
    ops, due_dates, weights = [], {}, {}
    for j, job_durations in enumerate(durations):
        job = f"Z{j + 1}"
        ops += [Operation(job, m, int(d), k) for k, (m, d) in enumerate(zip(machines, job_durations, strict=True))]
        due_dates[job] = int(job_durations.sum() + rng.integers(0, max_slack + 1))
        weights[job] = int(rng.integers(1, max_weight + 1))
    return Instance(f"random_n{n_jobs}_s{seed}", tuple(ops), due_dates, weights)
