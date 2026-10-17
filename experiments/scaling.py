"""
    python -m experiments.scaling --sizes 3 4 5 6 7 8 10 --instances 3 --solvers sa veloxq
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

from erp.generator import random_instance
from erp.instance import Instance
from erp.objective import Objective
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import Schedule, makespan, permutation_schedule, validate, weighted_tardiness
from erp.solvers import solve
from erp.solvers.cpsat import solve_cpsat

FIELDS = ["n_jobs", "instance", "horizon", "num_vars", "num_quadratic", "penalty_mode", "penalty",
          "cpsat_wt", "cpsat_wt_h", "solver", "run", "wt", "makespan", "gap_h", "gap",
          "feasible_fraction", "best_fraction", "time_s"]
PENALTY_MODES = ("worst", "edd")


@dataclass
class Case:
    instance: Instance
    edd: Schedule
    horizon: int
    cpsat_wt: int
    cpsat_wt_h: int

    @classmethod
    def generate(cls, n_jobs: int, seed: int, slack: int) -> "Case":
        instance = random_instance(n_jobs, seed)
        edd = permutation_schedule(instance, sorted(instance.jobs, key=instance.due_date))
        horizon = makespan(edd) + slack

        def optimum(**limit):
            return weighted_tardiness(instance, solve_cpsat(instance, Objective.TARDINESS, **limit)[0])

        return cls(instance, edd, horizon, optimum(), optimum(horizon=horizon))

    def model(self, penalty_mode: str) -> TimeIndexedQubo:
        model = TimeIndexedQubo(self.instance, self.horizon, objective=Objective.TARDINESS)
        if penalty_mode == "edd":  # infeasible energy >= P > E(EDD) >= optimum
            penalty = model.energy(model.encode(self.edd)) + 1.0
            model = TimeIndexedQubo(self.instance, self.horizon, objective=Objective.TARDINESS, penalty=penalty)
        return model


def measure(case: Case, model: TimeIndexedQubo, penalty_mode: str, solver: str, run: int) -> dict:
    started = time.perf_counter()
    result = solve(solver, model, seed=run)
    row = {
        "n_jobs": len(case.instance.jobs), "instance": case.instance.name, "horizon": case.horizon,
        "num_vars": model.num_variables, "num_quadratic": sum(1 for i, j in model.qubo()[0] if i != j),
        "penalty_mode": penalty_mode, "penalty": round(model.penalty, 3),
        "cpsat_wt": case.cpsat_wt, "cpsat_wt_h": case.cpsat_wt_h, "solver": solver, "run": run,
        "feasible_fraction": round(result.stats.get("feasible_fraction", 0.0), 4),
        "best_fraction": round(result.stats.get("best_fraction", 0.0), 4),
        "time_s": round(time.perf_counter() - started, 3),
    }
    if result.schedule is not None:
        assert validate(case.instance, result.schedule) == []
        wt = weighted_tardiness(case.instance, result.schedule)
        row |= {"wt": wt, "makespan": makespan(result.schedule),
                "gap_h": wt - case.cpsat_wt_h, "gap": wt - case.cpsat_wt}
    return row


def finished_runs(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {(r["instance"], r["penalty_mode"], r["solver"], r["run"]) for r in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    parser.add_argument("--instances", type=int, default=3, help="random instances per size")
    parser.add_argument("--solvers", nargs="+", default=["sa", "veloxq"])
    parser.add_argument("--runs", type=int, default=3, help="independent runs per solver and instance")
    parser.add_argument("--slack", type=int, default=1, help="horizon = EDD makespan + slack")
    parser.add_argument("--penalties", nargs="+", choices=PENALTY_MODES, default=list(PENALTY_MODES))
    parser.add_argument("--out", type=Path, default=Path("results/scaling.csv"))
    args = parser.parse_args()

    done = finished_runs(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for n in args.sizes:
            for seed in range(args.instances):
                case = Case.generate(n, seed, args.slack)
                for mode in args.penalties:
                    model = case.model(mode)
                    print(f"{case.instance.name}: H={case.horizon}, {model.num_variables} vars, "
                          f"penalty {mode}={model.penalty:.1f}, CP-SAT {case.cpsat_wt_h}")
                    for solver in args.solvers:
                        for run in range(args.runs):
                            if (case.instance.name, mode, solver, str(run)) in done:
                                continue
                            row = measure(case, model, mode, solver, run)
                            writer.writerow(row)
                            f.flush()
                            print(f"  {solver} run {run}: wt={row.get('wt', '-')} gap_h={row.get('gap_h', '-')} "
                                  f"t={row['time_s']}s")


if __name__ == "__main__":
    main()
