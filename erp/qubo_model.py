"""E(x) = penalty * (x^T Q_penalty x + offset_penalty) + x^T Q_objective x"""

from collections import defaultdict
from itertools import combinations

import numpy as np

from erp.schedule import Schedule


class QuboModel:
    label = "QUBO"

    def __init__(self, penalty: float):
        self.penalty = penalty
        self.variables: list = []
        self.index: dict = {}
        self.q_penalty: dict[tuple[int, int], float] = defaultdict(float)
        self.offset_penalty = 0.0
        self.q_objective: dict[tuple[int, int], float] = defaultdict(float)

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    def _set_variables(self, variables: list) -> None:
        self.variables = variables
        self.index = {v: i for i, v in enumerate(variables)}

    def _add(self, q: dict, i: int, j: int, value: float) -> None:
        q[(min(i, j), max(i, j))] += value

    def _add_exactly_one(self, idx: list[int]) -> None:
        # (sum x - 1)^2 = -sum x + 2 * sum_{i<j} x_i x_j + 1   (x^2 = x)
        for i in idx:
            self._add(self.q_penalty, i, i, -1.0)
        for i, j in combinations(idx, 2):
            self._add(self.q_penalty, i, j, 2.0)
        self.offset_penalty += 1.0

    def qubo(self) -> tuple[dict[tuple[int, int], float], float]:
        """(upper-triangular coefficients, constant offset)"""
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
        """Energy of every basis state; bit i of state k is (k >> i) & 1, as in Qiskit."""
        n = self.num_variables
        states = np.arange(2**n, dtype=np.int64)
        bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.float64)
        q, offset = self.qubo()
        energies = np.full(2**n, offset)
        for (i, j), v in q.items():
            energies += v * (bits[:, i] if i == j else bits[:, i] * bits[:, j])
        return energies

    def decode(self, bits) -> Schedule | None:
        raise NotImplementedError


def _evaluate(q: dict[tuple[int, int], float], bits) -> float:
    ones = [i for i, b in enumerate(bits) if b]
    return sum(q.get((i, j), 0.0) for k, i in enumerate(ones) for j in ones[k:])
