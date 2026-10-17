import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Operation:
    job: str
    machine: str
    duration: int
    step: int  # 0 = first operation of the job


@dataclass(frozen=True)
class Instance:
    name: str
    operations: tuple[Operation, ...]

    @property
    def jobs(self) -> list[str]:
        return list(dict.fromkeys(op.job for op in self.operations))

    @property
    def machines(self) -> list[str]:
        return list(dict.fromkeys(op.machine for op in self.operations))

    def job_operations(self, job: str) -> list[Operation]:
        return sorted((op for op in self.operations if op.job == job), key=lambda op: op.step)

    def head(self, op: Operation) -> int:
        """Work before `op` in its job, i.e. its earliest start."""
        return sum(o.duration for o in self.job_operations(op.job)[: op.step])

    def tail(self, op: Operation) -> int:
        """Work from `op` (inclusive) to the end of its job."""
        return sum(o.duration for o in self.job_operations(op.job)[op.step :])

    def makespan_lower_bound(self) -> int:
        """max(longest job, max over machines of min head + load + min remaining tail)."""
        bound = max(sum(o.duration for o in self.job_operations(j)) for j in self.jobs)
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
    ops = []
    for job in data["jobs"]:
        for step, op in enumerate(job["operations"]):
            ops.append(Operation(job["id"], op["machine"], int(op["duration"]), step))
    return Instance(name=path.stem, operations=tuple(ops))
