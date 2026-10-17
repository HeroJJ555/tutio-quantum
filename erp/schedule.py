import json
from dataclasses import dataclass
from pathlib import Path

from erp.instance import Instance


@dataclass(frozen=True)
class ScheduledOp:
    job: str
    machine: str
    start: int
    end: int


Schedule = list[ScheduledOp]


def makespan(schedule: Schedule) -> int:
    return max(op.end for op in schedule)


def permutation_schedule(instance: Instance, order: list[str]) -> Schedule:
    """Every machine processes the jobs in `order`, each operation as early as possible."""
    machine_free = {m: 0 for m in instance.machines}
    schedule = []
    for job in order:
        ready = 0
        for op in instance.job_operations(job):
            start = max(ready, machine_free[op.machine])
            ready = machine_free[op.machine] = start + op.duration
            schedule.append(ScheduledOp(job, op.machine, start, ready))
    return schedule


def validate(instance: Instance, schedule: Schedule) -> list[str]:
    """Violated constraints; empty means feasible."""
    errors = []
    by_key = {}
    for s in schedule:
        key = (s.job, s.machine)
        if key in by_key:
            errors.append(f"{s.job} on {s.machine} scheduled more than once")
        by_key[key] = s

    for op in instance.operations:
        s = by_key.get((op.job, op.machine))
        if s is None:
            errors.append(f"{op.job} on {op.machine} not scheduled")
        elif s.start < 0:
            errors.append(f"{op.job} on {op.machine} starts before 0")
        elif s.end - s.start != op.duration:
            errors.append(f"{op.job} on {op.machine} has duration {s.end - s.start}, expected {op.duration}")
    if errors:
        return errors

    for job in instance.jobs:
        ops = instance.job_operations(job)
        for prev, nxt in zip(ops, ops[1:]):
            a, b = by_key[(job, prev.machine)], by_key[(job, nxt.machine)]
            if b.start < a.end:
                errors.append(f"{job}: {nxt.machine} starts at {b.start} before {prev.machine} ends at {a.end}")

    for m in instance.machines:
        ops = sorted((s for s in schedule if s.machine == m), key=lambda s: s.start)
        for a, b in zip(ops, ops[1:]):
            if b.start < a.end:
                errors.append(f"machine {m}: {a.job} [{a.start},{a.end}) overlaps {b.job} [{b.start},{b.end})")
    return errors


def to_output_dict(instance: Instance, schedule: Schedule, method: str) -> dict:
    """Format from Section 14 of the challenge."""
    ordered = sorted(schedule, key=lambda s: (s.start, s.machine))
    return {
        "instance": instance.name,
        "method": method,
        "makespan": makespan(schedule),
        "schedule": [
            {"job": s.job, "machine": s.machine, "start": s.start, "end": s.end} for s in ordered
        ],
    }


def write_schedule(path: str | Path, instance: Instance, schedule: Schedule, method: str) -> None:
    errors = validate(instance, schedule)
    if errors:
        raise ValueError("refusing to write an infeasible schedule: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_output_dict(instance, schedule, method), indent=2) + "\n", encoding="utf-8")


def read_schedule(path: str | Path) -> Schedule:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ScheduledOp(s["job"], s["machine"], s["start"], s["end"]) for s in data["schedule"]]
