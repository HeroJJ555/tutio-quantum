from itertools import permutations

import pytest
from instances import DUE

from erp.objective import Objective
from erp.qubo_permutation import PermutationQubo
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import completion_times, makespan, permutation_schedule, to_output_dict, validate, weighted_tardiness
from erp.solvers.brute_force import solve_brute_force
from erp.solvers.cpsat import solve_cpsat

TARDINESS = Objective.TARDINESS


def feasible_assignments(model):
    ops = list(model.windows)

    def extend(k, chosen):
        if k == len(ops):
            yield chosen
            return
        for t in model.windows[ops[k]]:
            i = model.index[(ops[k], t)]
            if all(model.q_penalty.get((min(i, j), max(i, j)), 0.0) <= 0 for j in chosen):
                yield from extend(k + 1, chosen + [i])

    for chosen in extend(0, []):
        yield [int(i in chosen) for i in range(model.num_variables)]


def test_due_dates_loaded():
    assert [DUE.due_date(j) for j in DUE.jobs] == [5, 7, 7]
    assert [DUE.weight(j) for j in DUE.jobs] == [1, 1, 3]


def test_cpsat_tardiness_reference():
    schedule, proven = solve_cpsat(DUE, TARDINESS)
    assert proven
    assert weighted_tardiness(DUE, schedule) == 2
    assert makespan(schedule) == 9
    bounded, _ = solve_cpsat(DUE, TARDINESS, horizon=8)
    assert weighted_tardiness(DUE, bounded) == 3


@pytest.mark.parametrize("horizon,expected", [(8, 3), (9, 2)])
def test_time_indexed_qubo_minimum_matches_cpsat(horizon, expected):
    model = TimeIndexedQubo(DUE, horizon, objective=TARDINESS)
    feasible = list(feasible_assignments(model))
    for bits in feasible:
        schedule = model.decode(bits)
        assert validate(DUE, schedule) == []
        assert model.penalty_energy(bits) == 0
        tie_break = model.eps * sum(completion_times(schedule).values())
        assert model.energy(bits) == pytest.approx(weighted_tardiness(DUE, schedule) + tie_break)
    best = min(feasible, key=model.energy)
    assert weighted_tardiness(DUE, model.decode(best)) == expected
    assert model.penalty > max(model.energy(b) for b in feasible)


def test_tardiness_costs_no_extra_variables():
    for horizon in (8, 9, 10):
        assert (TimeIndexedQubo(DUE, horizon, objective=TARDINESS).num_variables
                == TimeIndexedQubo(DUE, horizon).num_variables)


@pytest.mark.parametrize("implicit", [False, True])
def test_permutation_qubo_tardiness(implicit):
    model = PermutationQubo(DUE, implicit_last=implicit, objective=TARDINESS)
    for order in permutations(DUE.jobs):
        wt = weighted_tardiness(DUE, permutation_schedule(DUE, list(order)))
        assert wt <= model.energy(model.encode(list(order))) < wt + 1

    assert weighted_tardiness(DUE, solve_brute_force(model).schedule) == 2

    worst_feasible = max(model.order_cost.values())
    for k, e in enumerate(model.energies_all_states()):
        bits = [(k >> i) & 1 for i in range(model.num_variables)]
        if model.order(bits) is None:
            assert e > worst_feasible


def test_output_has_tardiness_block():
    schedule, _ = solve_cpsat(DUE, TARDINESS)
    out = to_output_dict(DUE, schedule, "CP-SAT")
    assert out["weighted_tardiness"] == 2
    for job in out["jobs"]:
        assert job["tardiness"] == max(0, job["completion"] - job["due_date"])
