from itertools import permutations

import numpy as np
import pytest
from instances import A, B

from erp.instance import Instance, Operation
from erp.qubo_permutation import PermutationQubo
from erp.schedule import makespan, permutation_schedule, validate
from erp.solvers.brute_force import solve_brute_force

CASES = [(A, False, 4), (A, True, 2), (B, False, 9), (B, True, 6)]


@pytest.mark.parametrize("instance", [A, B])
def test_permutation_schedules_are_feasible(instance):
    for order in permutations(instance.jobs):
        assert validate(instance, permutation_schedule(instance, list(order))) == []


def test_best_order_matches_reference():
    assert min(makespan(permutation_schedule(B, list(o))) for o in permutations(B.jobs)) == 8
    assert makespan(permutation_schedule(A, ["Z2", "Z1"])) == 4


@pytest.mark.parametrize("instance,implicit,num_vars", CASES)
def test_variable_count(instance, implicit, num_vars):
    assert PermutationQubo(instance, implicit_last=implicit).num_variables == num_vars


@pytest.mark.parametrize("instance,implicit,num_vars", CASES)
def test_every_order_has_zero_penalty_and_exact_cost(instance, implicit, num_vars):
    model = PermutationQubo(instance, implicit_last=implicit)
    lower = instance.makespan_lower_bound()
    for order in permutations(instance.jobs):
        bits = model.encode(list(order))
        assert model.penalty_energy(bits) == 0
        assert model.order(bits) == list(order)
        assert model.energy(bits) == pytest.approx(makespan(permutation_schedule(instance, list(order))) - lower)


@pytest.mark.parametrize("instance,implicit,num_vars", CASES)
def test_ground_state_is_optimal_and_infeasible_states_cost_more(instance, implicit, num_vars):
    model = PermutationQubo(instance, implicit_last=implicit)
    result = solve_brute_force(model)
    assert result.schedule is not None
    assert makespan(result.schedule) == {A: 4, B: 8}[instance]

    energies = model.energies_all_states()
    worst_feasible = max(model.order_cost.values())
    for k, e in enumerate(energies):
        bits = [(k >> i) & 1 for i in range(model.num_variables)]
        if model.order(bits) is None:
            assert model.penalty_energy(bits) > 0
            assert e > worst_feasible
    assert np.isclose(energies.min(), min(model.order_cost.values()))


def test_more_than_three_jobs_rejected():
    ops = tuple(Operation(f"Z{j}", "T", 1, 0) for j in range(4))
    with pytest.raises(ValueError):
        PermutationQubo(Instance("four_jobs", ops))
