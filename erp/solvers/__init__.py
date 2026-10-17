from erp.qubo_model import QuboModel
from erp.solvers.base import QuboResult

SOLVERS = ("qaoa", "sa", "veloxq", "brute")


def solve(name: str, model: QuboModel, seed: int = 0, reps: int = 2, shots: int = 4000,
          reads: int = 1000) -> QuboResult:
    if name == "qaoa":
        from erp.solvers.qaoa import solve_qaoa
        return solve_qaoa(model, reps=reps, shots=shots, seed=seed)
    if name == "sa":
        from erp.solvers.annealing import solve_simulated_annealing
        return solve_simulated_annealing(model, num_reads=reads, seed=seed)
    if name == "veloxq":
        from erp.solvers.veloxq import solve_veloxq
        return solve_veloxq(model)
    if name == "brute":
        from erp.solvers.brute_force import solve_brute_force
        return solve_brute_force(model)
    raise ValueError(f"unknown solver {name!r}, expected one of {SOLVERS}")
