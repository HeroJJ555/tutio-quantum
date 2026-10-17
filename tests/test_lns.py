from itertools import pairwise

from instances import DUE

from erp.generator import random_instance
from erp.lns import free_windows, large_neighbourhood_search
from erp.objective import Objective
from erp.schedule import ScheduledOp, left_shift, makespan, permutation_schedule, validate, weighted_tardiness
from erp.solvers.cpsat import solve_cpsat


def edd(instance):
    return permutation_schedule(instance, sorted(instance.jobs, key=instance.due_date))


def test_free_windows_skip_fixed_operations_and_keep_current_starts():
    instance = random_instance(5, seed=1)
    schedule = edd(instance)
    free = instance.jobs[:2]
    windows = free_windows(instance, schedule, free, makespan(schedule) + 1)
    fixed = [s for s in schedule if s.job not in free]
    for s in schedule:
        if s.job in free:
            op = next(op for op in windows if (op.job, op.machine) == (s.job, s.machine))
            assert s.start in windows[op]
            for t in windows[op]:
                assert all(t + op.duration <= f.start or f.end <= t for f in fixed if f.machine == op.machine)


def test_left_shift_keeps_feasibility_and_never_delays():
    instance = random_instance(6, seed=2)
    schedule = solve_cpsat(instance, Objective.TARDINESS)[0]
    delayed = [ScheduledOp(s.job, s.machine, s.start + 2, s.end + 2) for s in schedule]
    shifted = left_shift(delayed)
    assert validate(instance, shifted) == []
    assert weighted_tardiness(instance, shifted) <= weighted_tardiness(instance, schedule)


def test_lns_reaches_optimum_on_due_instance():
    start = edd(DUE)
    result = large_neighbourhood_search(DUE, start, horizon=9, iterations=10, subset_size=2, seed=0)
    assert validate(DUE, result.schedule) == []
    assert result.history[-1] == 2
    assert all(b <= a for a, b in pairwise([weighted_tardiness(DUE, start), *result.history]))


def test_lns_improves_larger_instance_with_small_subproblems():
    instance = random_instance(8, seed=0)
    start = edd(instance)
    result = large_neighbourhood_search(instance, start, makespan(start) + 1, iterations=15, subset_size=3)
    assert validate(instance, result.schedule) == []
    assert result.history[-1] < weighted_tardiness(instance, start)
    assert max(result.subproblem_sizes) < 120
