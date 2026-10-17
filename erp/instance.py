import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True)
class Operation:
    job: str
    machine: str
    duration: int
    step: int


@dataclass(frozen=True)
class Instance:
    name: str
    operations: tuple[Operation, ...]
    due_dates: dict[str, int] = field(default_factory=dict, hash=False)
    weights: dict[str, int] = field(default_factory=dict, hash=False)

    @property
    def has_due_dates(self) -> bool:
        return bool(self.due_dates)

    def due_date(self, job: str) -> int | None:
        return self.due_dates.get(job)

    def weight(self, job: str) -> int:
        return self.weights.get(job, 1)

    @cached_property
    def jobs(self) -> list[str]:
        return list(dict.fromkeys(op.job for op in self.operations))

    @cached_property
    def machines(self) -> list[str]:
        return list(dict.fromkeys(op.machine for op in self.operations))

    @cached_property
    def _operations_by_job(self) -> dict[str, list[Operation]]:
        return {job: sorted((op for op in self.operations if op.job == job), key=lambda op: op.step)
                for job in self.jobs}

    def job_operations(self, job: str) -> list[Operation]:
        return self._operations_by_job[job]

    def is_last(self, op: Operation) -> bool:
        return op.step == len(self.job_operations(op.job)) - 1

    def head(self, op: Operation) -> int:
        return sum(o.duration for o in self.job_operations(op.job)[: op.step])

    def tail(self, op: Operation) -> int:
        return sum(o.duration for o in self.job_operations(op.job)[op.step :])

    def makespan_lower_bound(self) -> int:
        bound = max(self.tail(ops[0]) for ops in self._operations_by_job.values())
        for m in self.machines:
            ops = [op for op in self.operations if op.machine == m]
            load = sum(op.duration for op in ops)
            min_head = min(self.head(op) for op in ops)
            min_tail = min(self.tail(op) - op.duration for op in ops)
            bound = max(bound, min_head + load + min_tail)
        return bound


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    ops, due_dates, weights = [], {}, {}
    for job in data["jobs"]:
        ops += [Operation(job["id"], op["machine"], int(op["duration"]), step)
                for step, op in enumerate(job["operations"])]
        if "due_date" in job:
            due_dates[job["id"]] = int(job["due_date"])
        if "weight" in job:
            weights[job["id"]] = int(job["weight"])
    return Instance(path.stem, tuple(ops), due_dates, weights)
