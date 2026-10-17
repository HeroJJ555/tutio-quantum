from itertools import product
from pathlib import Path

import pytest

from erp.instance import load_instance
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import ScheduledOp, makespan, read_schedule, validate
from erp.solvers.brute_force import solve_brute_force
from erp.solvers.cpsat import solve_cpsat

ROOT = Path(__file__).resolve().parents[1]
A = load_instance(ROOT / "data" / "instance_2x2.json")
B = load_instance(ROOT / "data" / "instance_3x3.json")


def one_hot_assignments(model):
    for starts in product(*model.windows.values()):
        bits = [0] * model.num_variables
        for o, t in enumerate(starts):
            bits[model.index[(o, t)]] = 1
        yield bits


def test_cpsat_reference_makespans():
    assert makespan(solve_cpsat(A)[0]) == 4
    assert makespan(solve_cpsat(B)[0]) == 8


def test_variable_counts_match_challenge():
    assert TimeIndexedQubo(A, 4).num_variables == 8
    assert TimeIndexedQubo(B, 8).num_variables == 39


def test_lower_bounds():
    assert A.makespan_lower_bound() == 4
    assert B.makespan_lower_bound() == 8


@pytest.mark.parametrize("instance,horizon", [(A, 4), (B, 8)])
def test_optimal_schedule_has_zero_penalty(instance, horizon):
    model = TimeIndexedQubo(instance, horizon)
    schedule, _ = solve_cpsat(instance)
    assert model.penalty_energy(model.encode(schedule)) == 0


def test_each_constraint_type_is_penalised():
    model = TimeIndexedQubo(A, 5)
    good = [ScheduledOp("Z2", "T", 0, 1), ScheduledOp("Z2", "F", 1, 3),
            ScheduledOp("Z1", "T", 1, 3), ScheduledOp("Z1", "F", 3, 4)]
    assert model.penalty_energy(model.encode(good)) == 0

    precedence = [ScheduledOp("Z2", "T", 0, 1), ScheduledOp("Z2", "F", 1, 3),
                  ScheduledOp("Z1", "T", 1, 3), ScheduledOp("Z1", "F", 2, 3)]  # F before T ends
    machine = [ScheduledOp("Z2", "T", 0, 1), ScheduledOp("Z2", "F", 2, 4),
               ScheduledOp("Z1", "T", 0, 2), ScheduledOp("Z1", "F", 3, 4)]  # T used twice at t=0
    for bad in (precedence, machine):
        assert validate(A, bad)
        assert model.penalty_energy(model.encode(bad)) > 0

    no_start = model.encode(good)
    no_start[model.index[(0, 1)]] = 0  # Z1/T never starts
    assert model.penalty_energy(no_start) > 0


@pytest.mark.parametrize("instance,horizon", [(A, 3), (A, 4), (A, 5), (B, 7), (B, 8)])
def test_zero_penalty_iff_validator_accepts(instance, horizon):
    model = TimeIndexedQubo(instance, horizon)
    feasible = 0
    for bits in one_hot_assignments(model):
        schedule = model.decode(bits)
        ok = not validate(instance, schedule) and makespan(schedule) <= horizon
        assert (model.penalty_energy(bits) == 0) == ok
        feasible += ok
    expected = {(A, 3): 0, (A, 4): 1, (A, 5): 7, (B, 7): 0, (B, 8): 13}
    assert feasible == expected[(instance, horizon)]


@pytest.mark.parametrize("horizon", [4, 5])
def test_qubo_ground_state_is_optimal_2x2(horizon):
    result = solve_brute_force(TimeIndexedQubo(A, horizon))
    assert result.schedule is not None
    assert not validate(A, result.schedule)
    assert makespan(result.schedule) == 4


@pytest.mark.parametrize("name,instance,expected", [("schedule_2x2.json", A, 4), ("schedule_3x3.json", B, 8)])
def test_submitted_schedule_files(name, instance, expected):
    path = ROOT / "outputs" / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    schedule = read_schedule(path)
    assert validate(instance, schedule) == []
    assert makespan(schedule) == expected
