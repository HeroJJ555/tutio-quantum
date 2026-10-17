"""Permutation QUBO: y[j, p] = 1 iff job j is at position p of the common machine order.

For makespan on <= 3 machines some permutation schedule is optimal, so searching over job
orders loses nothing. The first n-1 positions fix the order, hence the exact makespan is a
polynomial of order n-1 in y: quadratic only for n <= 3 jobs.

implicit_last=True drops the variables of the last position (n^2 - n instead of n^2);
the remaining job is placed there by the decoder.

The objective is C(order) - lower bound for every order, i.e. it is built by evaluating all
n! orders classically. That is cheap for n = 3 but is the hidden price of this encoding.
"""

from itertools import combinations, permutations

from erp.instance import Instance
from erp.qubo_model import QuboModel
from erp.schedule import Schedule, makespan, permutation_schedule


class PermutationQubo(QuboModel):
    def __init__(self, instance: Instance, implicit_last: bool = False, penalty: float | None = None):
        jobs = instance.jobs
        n = len(jobs)
        if n > 3:
            raise ValueError("the exact makespan objective is quadratic only for n <= 3 jobs")
        self.instance = instance
        self.jobs = jobs
        self.implicit_last = implicit_last
        self.positions = n - 1 if implicit_last else n

        lower = instance.makespan_lower_bound()
        self.order_cost = {
            order: makespan(permutation_schedule(instance, list(order))) - lower for order in permutations(jobs)
        }
        # Any violation costs >= penalty, more than any objective value a violation could avoid.
        super().__init__(penalty if penalty is not None else 1.0 + max(self.order_cost.values()))

        self._set_variables([(j, p) for j in jobs for p in range(self.positions)])
        self.label = f"permutation QUBO{' (implicit last position)' if implicit_last else ''}, {self.num_variables} vars"

        for p in range(self.positions):
            self._add_exactly_one([self.index[(j, p)] for j in jobs])
        for j in jobs:
            idx = [self.index[(j, p)] for p in range(self.positions)]
            if implicit_last:  # at most one position
                for a, b in combinations(idx, 2):
                    self._add(self.q_penalty, a, b, 1.0)
            else:
                self._add_exactly_one(idx)

        for order, cost in self.order_cost.items():
            idx = [self.index[(order[p], p)] for p in range(n - 1)]
            if len(idx) == 1:
                self._add(self.q_objective, idx[0], idx[0], cost)
            elif len(idx) == 2:
                self._add(self.q_objective, idx[0], idx[1], cost)

    def order(self, bits) -> list[str] | None:
        at = {}
        for i, b in enumerate(bits):
            if b:
                j, p = self.variables[i]
                if p in at:
                    return None
                at[p] = j
        if sorted(at) != list(range(self.positions)) or len(set(at.values())) != len(at):
            return None
        order = [at[p] for p in range(self.positions)]
        if self.implicit_last:
            order += [j for j in self.jobs if j not in order]
        return order

    def decode(self, bits) -> Schedule | None:
        order = self.order(bits)
        return permutation_schedule(self.instance, order) if order else None

    def encode(self, order: list[str]) -> list[int]:
        bits = [0] * self.num_variables
        for p, j in enumerate(order[: self.positions]):
            bits[self.index[(j, p)]] = 1
        return bits
