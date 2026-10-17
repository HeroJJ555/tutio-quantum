"""Time-indexed QUBO: x[o, t] = 1 iff operation o starts at t, t in [head(o), H - tail(o)].

E = P * (one start + precedence + machine conflicts) + eps * sum of job completion times.
Penalty part is 0 iff x is a feasible schedule with makespan <= H; the eps term stays < 1.
"""

from collections import defaultdict
from itertools import combinations

from erp.instance import Instance, Operation
from erp.qubo_model import QuboModel
from erp.schedule import Schedule, ScheduledOp


class TimeIndexedQubo(QuboModel):
    def __init__(self, instance: Instance, horizon: int, penalty: float = 1.0, tie_break: bool = True):
        super().__init__(penalty)
        self.instance = instance
        self.horizon = horizon
        self.ops: list[Operation] = list(instance.operations)

        variables = []
        self.windows: dict[int, range] = {}
        for o, op in enumerate(self.ops):
            window = range(instance.head(op), horizon - instance.tail(op) + 1)
            self.windows[o] = window
            variables.extend((o, t) for t in window)
        self._set_variables(variables)
        self.label = f"time-indexed QUBO, {self.num_variables} vars"

        for o, window in self.windows.items():
            self._add_exactly_one([self.index[(o, t)] for t in window])
        self._add_precedence()
        self._add_machine_conflicts()
        if tie_break:
            self._add_completion_tie_break()

    def _add_precedence(self) -> None:
        for o, op in enumerate(self.ops):
            for o2, op2 in enumerate(self.ops):
                if op2.job != op.job or op2.step != op.step + 1:
                    continue
                for t in self.windows[o]:
                    for t2 in self.windows[o2]:
                        if t2 < t + op.duration:
                            self._add(self.q_penalty, self.index[(o, t)], self.index[(o2, t2)], 1.0)

    def _add_machine_conflicts(self) -> None:
        for (o, op), (o2, op2) in combinations(enumerate(self.ops), 2):
            if op.machine != op2.machine or op.job == op2.job:
                continue
            for t in self.windows[o]:
                for t2 in self.windows[o2]:
                    if t < t2 + op2.duration and t2 < t + op.duration:
                        self._add(self.q_penalty, self.index[(o, t)], self.index[(o2, t2)], 1.0)

    def _add_completion_tie_break(self) -> None:
        # eps < 1 / (jobs * H) keeps the whole term below one unit of penalty.
        last_ops = [o for o, op in enumerate(self.ops) if op.step == len(self.instance.job_operations(op.job)) - 1]
        self.eps = 1.0 / (len(last_ops) * self.horizon + 1)
        for o in last_ops:
            for t in self.windows[o]:
                i = self.index[(o, t)]
                self._add(self.q_objective, i, i, self.eps * (t + self.ops[o].duration))

    def decode(self, bits) -> Schedule | None:
        """None if some operation does not start exactly once."""
        starts = defaultdict(list)
        for i, b in enumerate(bits):
            if b:
                o, t = self.variables[i]
                starts[o].append(t)
        schedule = []
        for o, op in enumerate(self.ops):
            if len(starts[o]) != 1:
                return None
            t = starts[o][0]
            schedule.append(ScheduledOp(op.job, op.machine, t, t + op.duration))
        return schedule

    def encode(self, schedule: Schedule) -> list[int]:
        bits = [0] * self.num_variables
        start_of = {(s.job, s.machine): s.start for s in schedule}
        for o, op in enumerate(self.ops):
            key = (o, start_of[(op.job, op.machine)])
            if key not in self.index:
                raise ValueError(f"{op.job}/{op.machine} starts outside its window for H={self.horizon}")
            bits[self.index[key]] = 1
        return bits
