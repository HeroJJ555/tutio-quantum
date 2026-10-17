from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Hashable, Sequence
from itertools import combinations

import numpy as np

from erp.schedule import Schedule

Coefficients = dict[tuple[int, int], float]


class QuboModel(ABC):
    label: str

    def __init__(self, variables: Sequence[Hashable], penalty: float):
        self.variables = list(variables)
        self.index = {v: i for i, v in enumerate(self.variables)}
        self.penalty = penalty
        self.q_penalty: Coefficients = defaultdict(float)
        self.offset_penalty = 0.0
        self.q_objective: Coefficients = defaultdict(float)

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    @abstractmethod
    def decode(self, bits) -> Schedule | None: ...

    def qubo(self) -> tuple[Coefficients, float]:
        q = defaultdict(float)
        for k, v in self.q_penalty.items():
            q[k] += self.penalty * v
        for k, v in self.q_objective.items():
            q[k] += v
        return dict(q), self.penalty * self.offset_penalty

    def penalty_energy(self, bits) -> float:
        return _evaluate(self.q_penalty, bits) + self.offset_penalty

    def energy(self, bits) -> float:
        q, offset = self.qubo()
        return _evaluate(q, bits) + offset

    def energies_all_states(self) -> np.ndarray:
        # bit i of state k is (k >> i) & 1, as in Qiskit
        n = self.num_variables
        states = np.arange(2**n, dtype=np.int64)
        bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.float64)
        q, offset = self.qubo()
        energies = np.full(2**n, offset)
        for (i, j), v in q.items():
            energies += v * (bits[:, i] if i == j else bits[:, i] * bits[:, j])
        return energies

    def _add_penalty(self, i: int, j: int, value: float = 1.0) -> None:
        _add(self.q_penalty, i, j, value)

    def _add_objective(self, i: int, j: int, value: float) -> None:
        _add(self.q_objective, i, j, value)

    def _add_exactly_one(self, idx: list[int]) -> None:
        # (sum x - 1)^2 = -sum x + 2 * sum_{i<j} x_i x_j + 1   (x^2 = x)
        for i in idx:
            self._add_penalty(i, i, -1.0)
        for i, j in combinations(idx, 2):
            self._add_penalty(i, j, 2.0)
        self.offset_penalty += 1.0

    def _add_at_most_one(self, idx: list[int]) -> None:
        for i, j in combinations(idx, 2):
            self._add_penalty(i, j)


def _add(q: Coefficients, i: int, j: int, value: float) -> None:
    q[(min(i, j), max(i, j))] += value


def _evaluate(q: Coefficients, bits) -> float:
    ones = [i for i, b in enumerate(bits) if b]
    return sum(q.get((i, j), 0.0) for k, i in enumerate(ones) for j in ones[k:])
