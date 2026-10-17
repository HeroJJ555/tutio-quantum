"""Exact classical reference, built from the instance rather than from the QUBO."""

from ortools.sat.python import cp_model

from erp.instance import Instance
from erp.schedule import Schedule, ScheduledOp


def solve_cpsat(instance: Instance, time_limit_s: float = 10.0) -> tuple[Schedule, bool]:
    model = cp_model.CpModel()
    horizon = sum(op.duration for op in instance.operations)
    start, end, interval = {}, {}, {}
    for op in instance.operations:
        key = (op.job, op.machine)
        start[key] = model.new_int_var(0, horizon, f"s_{op.job}_{op.machine}")
        end[key] = model.new_int_var(0, horizon, f"e_{op.job}_{op.machine}")
        interval[key] = model.new_interval_var(start[key], op.duration, end[key], f"i_{op.job}_{op.machine}")

    for job in instance.jobs:
        ops = instance.job_operations(job)
        for prev, nxt in zip(ops, ops[1:]):
            model.add(start[(job, nxt.machine)] >= end[(job, prev.machine)])
    for m in instance.machines:
        model.add_no_overlap([interval[(op.job, m)] for op in instance.operations if op.machine == m])

    cmax = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(cmax, list(end.values()))
    model.minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no schedule")
    schedule = [
        ScheduledOp(op.job, op.machine, solver.value(start[(op.job, op.machine)]), solver.value(end[(op.job, op.machine)]))
        for op in instance.operations
    ]
    return schedule, status == cp_model.OPTIMAL
