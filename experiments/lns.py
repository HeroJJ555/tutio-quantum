"""
    python -m experiments.lns --sizes 5 6 7 8 10 12 --solvers sa --runs 3 --out results/lns_sa.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

from erp.lns import large_neighbourhood_search
from erp.schedule import weighted_tardiness
from experiments.scaling import Case

FIELDS = ["n_jobs", "instance", "horizon", "cpsat_wt_h", "edd_wt", "solver", "run", "subset_size", "iterations",
          "wt", "gap_h", "accepted", "max_vars", "mean_vars", "time_s", "history"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 6, 7, 8, 10, 12])
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--solvers", nargs="+", default=["sa"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--slack", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/lns.csv"))
    args = parser.parse_args()

    new_file, done = not args.out.exists(), set()
    if not new_file:
        with args.out.open(newline="") as f:
            done = {(r["instance"], r["solver"], r["run"]) for r in csv.DictReader(f)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for n in args.sizes:
            for seed in range(args.instances):
                case = Case.generate(n, seed, args.slack)
                edd_wt = weighted_tardiness(case.instance, case.edd)
                for solver in args.solvers:
                    for run in range(args.runs):
                        if (case.instance.name, solver, str(run)) in done:
                            continue
                        started = time.perf_counter()
                        result = large_neighbourhood_search(case.instance, case.edd, case.horizon, solver,
                                                            args.iterations, args.subset_size, seed=run)
                        wt = result.history[-1]
                        writer.writerow({
                            "n_jobs": n, "instance": case.instance.name, "horizon": case.horizon,
                            "cpsat_wt_h": case.cpsat_wt_h, "edd_wt": edd_wt, "solver": solver, "run": run,
                            "subset_size": args.subset_size, "iterations": args.iterations,
                            "wt": wt, "gap_h": wt - case.cpsat_wt_h, "accepted": result.accepted,
                            "max_vars": max(result.subproblem_sizes),
                            "mean_vars": round(sum(result.subproblem_sizes) / len(result.subproblem_sizes)),
                            "time_s": round(time.perf_counter() - started, 1),
                            "history": json.dumps(result.history),
                        })
                        f.flush()
                        print(f"{case.instance.name} {solver} run {run}: EDD {edd_wt} -> {wt} "
                              f"(CP-SAT {case.cpsat_wt_h}), max {max(result.subproblem_sizes)} vars")


if __name__ == "__main__":
    main()
