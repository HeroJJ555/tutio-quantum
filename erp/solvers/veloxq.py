import time

from erp.env import load_env
from erp.qubo_model import QuboModel
from erp.solvers.base import QuboResult, best_feasible, to_bqm


def solve_veloxq(model: QuboModel, num_rep: int = 4096, num_steps: int = 5000) -> QuboResult:
    load_env()  # the SDK reads VELOX_TOKEN at import time
    from veloxq_sdk import VeloxQParameters, VeloxQSolver

    solver = VeloxQSolver(parameters=VeloxQParameters(num_rep=num_rep, num_steps=num_steps))
    started = time.perf_counter()
    sampleset = solver.sample(to_bqm(model), vartype="BINARY")
    stats = {"num_rep": num_rep, "num_steps": num_steps, "wall_time_s": round(time.perf_counter() - started, 2)}
    return best_feasible(model, sampleset, f"VeloxQ (H100 backend), {model.label}", stats)
