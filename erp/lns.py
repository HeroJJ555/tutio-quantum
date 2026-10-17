from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from erp.instance import Instance
from erp.objective import Objective
from erp.qubo_time_indexed import TimeIndexedQubo, Windows
from erp.schedule import Schedule, completion_times, left_shift, tardiness, validate, weighted_tardiness
from erp.solvers import solve


@dataclass
class LnsResult:
    schedule: Schedule
    history: list[int] = field(default_factory=list)
    subproblem_sizes: list[int] = field(default_factory=list)
    accepted: int = 0


def free_windows(instance: Instance, schedule: Schedule, free_jobs: list[str], horizon: int,
                 radius: int | None = None) -> Windows:
    busy = defaultdict(list)
    for s in schedule:
        if s.job not in free_jobs:
            busy[s.machine].append((s.start, s.end))
    current = {(s.job, s.machine): s.start for s in schedule}

    def available(op, t):
        near = radius is None or abs(t - current[(op.job, op.machine)]) <= radius
        return near and all(t + op.duration <= start or end <= t for start, end in busy[op.machine])

    return {op: [t for t in range(instance.head(op), horizon - instance.tail(op) + 1) if available(op, t)]
            for job in free_jobs for op in instance.job_operations(job)}


def choose_jobs(instance: Instance, schedule: Schedule, size: int, rng: np.random.Generator) -> list[str]:
    done = completion_times(schedule)
    cost = np.array([instance.weight(j) * t for j, t in tardiness(instance, schedule).items()], dtype=float)
    jobs = instance.jobs
    if cost.sum() > 0 and rng.random() < 0.5:
        pivot = jobs[rng.choice(len(jobs), p=cost / cost.sum())]
    else:
        pivot = jobs[rng.integers(len(jobs))]
    near = sorted((j for j in jobs if j != pivot), key=lambda j: abs(done[j] - done[pivot]))[: 2 * size]
    return [pivot, *rng.choice(near, size=min(size - 1, len(near)), replace=False)]


def large_neighbourhood_search(instance: Instance, start: Schedule, horizon: int, solver: str = "sa",
                               iterations: int = 30, subset_size: int = 3, radius: int | None = None,
                               seed: int = 0) -> LnsResult:
    rng = np.random.default_rng(seed)
    result = LnsResult(left_shift(start))
    tried = set()
    for it in range(iterations):
        free = choose_jobs(instance, result.schedule, subset_size, rng)
        for _ in range(20):
            if frozenset(free) not in tried:
                break
            free = choose_jobs(instance, result.schedule, subset_size, rng)
        tried.add(frozenset(free))
        windows = free_windows(instance, result.schedule, free, horizon, radius)
        probe = TimeIndexedQubo(instance, horizon, objective=Objective.TARDINESS, windows=windows)
        current = probe.encode(result.schedule)
        # infeasible energy >= P > E(current) >= optimum
        model = TimeIndexedQubo(instance, horizon, objective=Objective.TARDINESS, windows=windows,
                                penalty=probe.energy(current) + 1.0)
        found = solve(solver, model, seed=seed * 1000 + it, reads=200)
        result.subproblem_sizes.append(model.num_variables)
        if found.schedule is not None and model.energy(found.bits) < model.energy(current) - 1e-9:
            candidate = left_shift([s for s in result.schedule if s.job not in free] + found.schedule)
            assert validate(instance, candidate) == []
            result.schedule = candidate
            result.accepted += 1
            tried.clear()
        result.history.append(weighted_tardiness(instance, result.schedule))
    return result
