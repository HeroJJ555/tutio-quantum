import dimod
import pytest
from instances import B

from erp.generator import random_instance
from erp.objective import Objective
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import makespan, validate
from erp.solvers import solve
from erp.solvers.base import best_feasible
from erp.solvers.cpsat import solve_cpsat


def test_best_feasible_accepts_string_labels():
    model = TimeIndexedQubo(B, 8)
    optimal = model.encode(solve_cpsat(B)[0])
    broken = [0] * model.num_variables
    sampleset = dimod.SampleSet.from_samples(
        [{str(i): b for i, b in enumerate(bits)} for bits in (broken, optimal)],
        vartype="BINARY", energy=[0.0, 0.0], num_occurrences=[3, 1],
    )
    result = best_feasible(model, sampleset, "test", {})
    assert result.bits == optimal
    assert result.stats["feasible_fraction"] == 0.25


def test_random_instances_are_reproducible_and_solvable():
    a = random_instance(5, seed=7)
    assert a == random_instance(5, seed=7)
    assert len(a.jobs) == 5 and a.machines == ["T", "F", "S"]
    assert all(a.due_date(j) >= a.tail(a.job_operations(j)[0]) for j in a.jobs)
    assert validate(a, solve_cpsat(a, Objective.TARDINESS)[0]) == []


def test_simulated_annealing_baseline_b():
    result = solve("sa", TimeIndexedQubo(B, 8), seed=3, reads=200)
    assert makespan(result.schedule) == 8


def test_unknown_solver_rejected():
    with pytest.raises(ValueError):
        solve("nope", TimeIndexedQubo(B, 8))
