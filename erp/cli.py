"""
    python -m erp.cli --instance data/instance_2x2.json --solver qaoa
    python -m erp.cli --instance data/instance_3x3.json --solver sa
    python -m erp.cli --instance data/instance_3x3.json --solver qaoa --encoding perm-implicit
    python -m erp.cli --instance data/instance_3x3_due.json --solver veloxq --objective tardiness --horizon 9
"""

import argparse
from pathlib import Path

from erp.gantt import plot_gantt
from erp.instance import Instance, load_instance
from erp.objective import Objective, score
from erp.qubo_model import QuboModel
from erp.qubo_permutation import PermutationQubo
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import makespan, validate, write_schedule
from erp.solvers import SOLVERS, solve
from erp.solvers.base import QuboResult
from erp.solvers.cpsat import solve_cpsat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--solver", choices=SOLVERS, required=True)
    parser.add_argument("--encoding", choices=["time", "perm", "perm-implicit"], default="time")
    parser.add_argument("--objective", type=Objective, choices=list(Objective), default=Objective.MAKESPAN)
    parser.add_argument("--horizon", type=int, help="horizon of the time-indexed tardiness QUBO")
    parser.add_argument("--out", type=Path, help="schedule JSON (default: under outputs/)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2, help="QAOA layers p")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--reads", type=int, default=1000, help="simulated-annealing reads")
    return parser.parse_args()


def is_baseline(args) -> bool:
    return args.encoding == "time" and args.objective is Objective.MAKESPAN


def default_output(args, instance: Instance) -> Path:
    size = instance.name.removeprefix("instance_")
    if is_baseline(args):
        return Path(f"outputs/schedule_{size}.json")
    return Path(f"outputs/extension/schedule_{size}_{args.objective}_{args.encoding}.json")


def run(args, model: QuboModel) -> QuboResult:
    return solve(args.solver, model, seed=args.seed, reps=args.reps, shots=args.shots, reads=args.reads)


def solve_by_horizon_scan(args, instance: Instance) -> QuboResult:
    lower, upper = instance.makespan_lower_bound(), sum(op.duration for op in instance.operations)
    print(f"horizon scan from lower bound {lower} (trivial upper bound {upper})")
    for horizon in range(lower, upper + 1):
        model = TimeIndexedQubo(instance, horizon)
        result = run(args, model)
        print(f"  H={horizon}: {model.num_variables} variables -> "
              f"{'feasible' if result.schedule else 'no feasible sample'} {result.stats}")
        if result.schedule:
            return result
    raise SystemExit("no feasible schedule found up to the trivial upper bound")


def build_single_model(args, instance: Instance) -> QuboModel:
    if args.encoding == "time":
        horizon = args.horizon or instance.makespan_lower_bound() + 1
        bounded, _ = solve_cpsat(instance, args.objective, horizon=horizon)
        print(f"CP-SAT {args.objective} with makespan <= {horizon}: {score(args.objective, instance, bounded)}")
        return TimeIndexedQubo(instance, horizon, objective=args.objective)
    return PermutationQubo(instance, implicit_last=args.encoding == "perm-implicit", objective=args.objective)


def main() -> None:
    args = parse_args()
    instance = load_instance(args.instance)
    if args.objective is Objective.TARDINESS and not instance.has_due_dates:
        raise SystemExit("tardiness objective needs due dates in the instance")

    reference, proven = solve_cpsat(instance, args.objective)
    best = score(args.objective, instance, reference)
    print(f"instance {instance.name}: {len(instance.jobs)} jobs, machines {instance.machines}")
    print(f"CP-SAT {args.objective}: {best} with makespan {makespan(reference)} "
          f"({'proven optimal' if proven else 'not proven'})")

    if is_baseline(args):
        result = solve_by_horizon_scan(args, instance)
    else:
        model = build_single_model(args, instance)
        result = run(args, model)
        print(f"  {model.num_variables} variables, penalty {model.penalty} -> {result.stats}")
        if result.schedule is None:
            raise SystemExit("no feasible sample")

    if errors := validate(instance, result.schedule):
        raise SystemExit("decoded schedule is infeasible: " + "; ".join(errors))
    value, cmax = score(args.objective, instance, result.schedule), makespan(result.schedule)
    summary = f"{result.method}: {args.objective} {value} (CP-SAT {best}, gap {value - best})"
    if args.objective is Objective.TARDINESS:
        summary += f", makespan {cmax}"
    print(summary)
    for s in sorted(result.schedule, key=lambda s: (s.start, s.machine)):
        print(f"  {s.job} {s.machine} [{s.start}, {s.end})")

    out = args.out or default_output(args, instance)
    write_schedule(out, instance, result.schedule, result.method)
    gantt = out.with_name(out.stem.replace("schedule", "gantt") + ".png")
    title = f"{instance.name} - makespan {cmax}"
    if args.objective is Objective.TARDINESS:
        title += f", weighted tardiness {value}"
    plot_gantt(instance, result.schedule, gantt, title=f"{title}\n{result.method}")
    print(f"wrote {out} and {gantt}")


if __name__ == "__main__":
    main()
