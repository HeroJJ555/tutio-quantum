from collections import defaultdict
from collections.abc import Iterator, Sequence
from itertools import combinations, pairwise

from erp.instance import Instance, Operation
from erp.objective import Objective
from erp.qubo_model import QuboModel
from erp.schedule import Schedule, ScheduledOp

Windows = dict[Operation, Sequence[int]]


def start_windows(instance: Instance, horizon: int) -> Windows:
    return {op: range(instance.head(op), horizon - instance.tail(op) + 1) for op in instance.operations}


class TimeIndexedQubo(QuboModel):
    def __init__(self, instance: Instance, horizon: int, penalty: float | None = None,
                 objective: Objective = Objective.MAKESPAN, tie_break: bool = True, windows: Windows | None = None):
        self.instance = instance
        self.horizon = horizon
        self.objective = Objective(objective)
        self.windows = windows if windows is not None else start_windows(instance, horizon)
        self.ops = list(self.windows)
        self.last_ops = [op for op in self.ops if instance.is_last(op)]
        if penalty is None:
            penalty = self.worst_objective() + 1.0
        super().__init__([(op, t) for op in self.ops for t in self.windows[op]], penalty)
        suffix = ", weighted tardiness" if self.objective is Objective.TARDINESS else ""
        self.label = f"time-indexed QUBO{suffix}, {self.num_variables} vars"

        for op in self.ops:
            self._add_exactly_one([self.index[(op, t)] for t in self.windows[op]])
        self._add_precedence()
        self._add_machine_conflicts()
        if self.objective is Objective.TARDINESS:
            self._add_weighted_tardiness()
        if tie_break:
            self._add_completion_tie_break()

    def worst_objective(self) -> float:
        if self.objective is Objective.MAKESPAN:
            return 0.0
        return sum(self.instance.weight(op.job) * max(0, self.horizon - due)
                   for op in self.last_ops if (due := self.instance.due_date(op.job)) is not None)

    def _start_pairs(self, a: Operation, b: Operation) -> Iterator[tuple[int, int, int, int]]:
        for t in self.windows[a]:
            for u in self.windows[b]:
                yield t, u, self.index[(a, t)], self.index[(b, u)]

    def _add_precedence(self) -> None:
        for job in self.instance.jobs:
            for a, b in pairwise(self.instance.job_operations(job)):
                if a in self.windows and b in self.windows:
                    for t, u, i, j in self._start_pairs(a, b):
                        if u < t + a.duration:
                            self._add_penalty(i, j)

    def _add_machine_conflicts(self) -> None:
        for a, b in combinations(self.ops, 2):
            if a.machine == b.machine and a.job != b.job:
                for t, u, i, j in self._start_pairs(a, b):
                    if t < u + b.duration and u < t + a.duration:
                        self._add_penalty(i, j)

    def _add_weighted_tardiness(self) -> None:
        for op in self.last_ops:
            due = self.instance.due_date(op.job)
            if due is None:
                continue
            for t in self.windows[op]:
                if (late := t + op.duration - due) > 0:
                    i = self.index[(op, t)]
                    self._add_objective(i, i, self.instance.weight(op.job) * late)

    def _add_completion_tie_break(self) -> None:
        self.eps = 1.0 / (len(self.last_ops) * self.horizon + 1)  # whole term < 1
        for op in self.last_ops:
            for t in self.windows[op]:
                i = self.index[(op, t)]
                self._add_objective(i, i, self.eps * (t + op.duration))

    def decode(self, bits) -> Schedule | None:
        starts = defaultdict(list)
        for (op, t), b in zip(self.variables, bits, strict=True):
            if b:
                starts[op].append(t)
        if any(len(starts[op]) != 1 for op in self.ops):
            return None
        return [ScheduledOp(op.job, op.machine, starts[op][0], starts[op][0] + op.duration) for op in self.ops]

    def encode(self, schedule: Schedule) -> list[int]:
        start_of = {(s.job, s.machine): s.start for s in schedule}
        bits = [0] * self.num_variables
        for op in self.ops:
            key = (op, start_of[(op.job, op.machine)])
            if key not in self.index:
                raise ValueError(f"{op.job}/{op.machine} starts outside its window for H={self.horizon}")
            bits[self.index[key]] = 1
        return bits
