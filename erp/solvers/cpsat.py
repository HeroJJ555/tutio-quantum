from itertools import pairwise

from ortools.sat.python import cp_model

from erp.instance import Instance
from erp.objective import Objective
from erp.schedule import Schedule, ScheduledOp


def solve_cpsat(instance: Instance, objective: Objective = Objective.MAKESPAN, horizon: int | None = None,
                time_limit_s: float = 10.0) -> tuple[Schedule, bool]:
    model = cp_model.CpModel()
    upper = sum(op.duration for op in instance.operations)
    start, end, interval = {}, {}, {}
    for op in instance.operations:
        name = f"{op.job}_{op.machine}"
        start[op] = model.new_int_var(0, upper, f"s_{name}")
        end[op] = model.new_int_var(0, upper, f"e_{name}")
        interval[op] = model.new_interval_var(start[op], op.duration, end[op], f"i_{name}")

    for job in instance.jobs:
        ops = instance.job_operations(job)
        for a, b in pairwise(ops):
            model.add(start[b] >= end[a])
    for m in instance.machines:
        model.add_no_overlap([interval[op] for op in instance.operations if op.machine == m])

    cmax = model.new_int_var(0, upper, "makespan")
    model.add_max_equality(cmax, list(end.values()))
    if horizon is not None:
        model.add(cmax <= horizon)

    if Objective(objective) is Objective.MAKESPAN:
        model.minimize(cmax)
    else:
        terms = []
        for job, due in instance.due_dates.items():
            late = model.new_int_var(0, upper, f"T_{job}")
            model.add(late >= end[instance.job_operations(job)[-1]] - due)
            terms.append(instance.weight(job) * late)
        model.minimize((upper + 1) * sum(terms) + cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no schedule")
    schedule = [ScheduledOp(op.job, op.machine, solver.value(start[op]), solver.value(end[op]))
                for op in instance.operations]
    return schedule, status == cp_model.OPTIMAL
