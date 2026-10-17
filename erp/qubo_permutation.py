from itertools import permutations

from erp.instance import Instance
from erp.objective import Objective
from erp.qubo_model import QuboModel
from erp.schedule import Schedule, makespan, permutation_schedule, weighted_tardiness


class PermutationQubo(QuboModel):
    def __init__(self, instance: Instance, implicit_last: bool = False, penalty: float | None = None,
                 objective: Objective = Objective.MAKESPAN):
        self.instance = instance
        self.jobs = instance.jobs
        self.objective = Objective(objective)
        self.implicit_last = implicit_last
        n = len(self.jobs)
        if n > 3:
            raise ValueError("the exact objective is quadratic only for n <= 3 jobs")
        self.positions = n - 1 if implicit_last else n
        self.order_cost = self._order_costs()
        if penalty is None:
            penalty = max(self.order_cost.values()) + 1.0
        super().__init__([(j, p) for j in self.jobs for p in range(self.positions)], penalty)
        self.label = (f"permutation QUBO{' (implicit last position)' if implicit_last else ''}"
                      f"{', weighted tardiness' if self.objective is Objective.TARDINESS else ''}, "
                      f"{self.num_variables} vars")

        for p in range(self.positions):
            self._add_exactly_one([self.index[(j, p)] for j in self.jobs])
        for j in self.jobs:
            idx = [self.index[(j, p)] for p in range(self.positions)]
            if implicit_last:
                self._add_at_most_one(idx)
            else:
                self._add_exactly_one(idx)
        self._add_order_costs()

    def _order_costs(self) -> dict[tuple[str, ...], float]:
        lower = self.instance.makespan_lower_bound()
        schedules = {order: permutation_schedule(self.instance, list(order)) for order in permutations(self.jobs)}
        extra = {order: makespan(s) - lower for order, s in schedules.items()}
        if self.objective is Objective.MAKESPAN:
            return extra
        scale = max(extra.values()) + 1
        return {order: weighted_tardiness(self.instance, s) + extra[order] / scale for order, s in schedules.items()}

    def _add_order_costs(self) -> None:
        # the first n - 1 positions fix the order
        for order, cost in self.order_cost.items():
            idx = [self.index[(job, p)] for p, job in enumerate(order[:-1])]
            if idx:
                self._add_objective(idx[0], idx[-1], cost)

    def order(self, bits) -> list[str] | None:
        at = {}
        for (job, p), b in zip(self.variables, bits, strict=True):
            if b:
                if p in at:
                    return None
                at[p] = job
        if sorted(at) != list(range(self.positions)) or len(set(at.values())) != len(at):
            return None
        order = [at[p] for p in range(self.positions)]
        return order + [j for j in self.jobs if j not in order] if self.implicit_last else order

    def decode(self, bits) -> Schedule | None:
        order = self.order(bits)
        return permutation_schedule(self.instance, order) if order else None

    def encode(self, order: list[str]) -> list[int]:
        bits = [0] * self.num_variables
        for p, job in enumerate(order[: self.positions]):
            bits[self.index[(job, p)]] = 1
        return bits
