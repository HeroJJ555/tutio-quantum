import json
from dataclasses import dataclass
from itertools import pairwise
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


def completion_times(schedule: Schedule) -> dict[str, int]:
    done = {}
    for s in schedule:
        done[s.job] = max(done.get(s.job, 0), s.end)
    return done


def tardiness(instance: Instance, schedule: Schedule) -> dict[str, int]:
    done = completion_times(schedule)
    return {
        job: max(0, done[job] - due) if (due := instance.due_date(job)) is not None else 0
        for job in instance.jobs
    }


def weighted_tardiness(instance: Instance, schedule: Schedule) -> int:
    return sum(instance.weight(job) * t for job, t in tardiness(instance, schedule).items())


def left_shift(schedule: Schedule) -> Schedule:
    machine_free, job_ready, shifted = {}, {}, []
    for s in sorted(schedule, key=lambda s: s.start):
        start = max(machine_free.get(s.machine, 0), job_ready.get(s.job, 0))
        end = machine_free[s.machine] = job_ready[s.job] = start + (s.end - s.start)
        shifted.append(ScheduledOp(s.job, s.machine, start, end))
    return shifted


def permutation_schedule(instance: Instance, order: list[str]) -> Schedule:
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
        for prev, nxt in pairwise(ops):
            a, b = by_key[(job, prev.machine)], by_key[(job, nxt.machine)]
            if b.start < a.end:
                errors.append(f"{job}: {nxt.machine} starts at {b.start} before {prev.machine} ends at {a.end}")

    for m in instance.machines:
        ops = sorted((s for s in schedule if s.machine == m), key=lambda s: s.start)
        for a, b in pairwise(ops):
            if b.start < a.end:
                errors.append(f"machine {m}: {a.job} [{a.start},{a.end}) overlaps {b.job} [{b.start},{b.end})")
    return errors


def to_output_dict(instance: Instance, schedule: Schedule, method: str) -> dict:
    ordered = sorted(schedule, key=lambda s: (s.start, s.machine))
    out = {
        "instance": instance.name,
        "method": method,
        "makespan": makespan(schedule),
    }
    if instance.has_due_dates:
        done, late = completion_times(schedule), tardiness(instance, schedule)
        out["weighted_tardiness"] = weighted_tardiness(instance, schedule)
        out["jobs"] = [
            {"job": j, "weight": instance.weight(j), "due_date": instance.due_date(j),
             "completion": done[j], "tardiness": late[j]}
            for j in instance.jobs
        ]
    out["schedule"] = [{"job": s.job, "machine": s.machine, "start": s.start, "end": s.end} for s in ordered]
    return out


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
