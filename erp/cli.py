"""Solve a scheduling instance through a QUBO and compare with CP-SAT.

time: feasibility QUBOs for H = lower bound, +1, ...; the first feasible H is the makespan.
perm, perm-implicit: one QUBO over job orders whose minimum is the optimal makespan.

    python -m erp.cli --instance data/instance_2x2.json --solver qaoa
    python -m erp.cli --instance data/instance_3x3.json --solver sa
    python -m erp.cli --instance data/instance_3x3.json --solver qaoa --encoding perm-implicit
"""

import argparse
from pathlib import Path

from erp.gantt import plot_gantt
from erp.instance import load_instance
from erp.qubo_permutation import PermutationQubo
from erp.qubo_time_indexed import TimeIndexedQubo
from erp.schedule import makespan, validate, write_schedule
from erp.solvers.cpsat import solve_cpsat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--solver", choices=["qaoa", "sa", "brute"], required=True)
    parser.add_argument("--encoding", choices=["time", "perm", "perm-implicit"], default="time")
    parser.add_argument("--out", help="schedule JSON (default: outputs/schedule_<size>.json for time, "
                                      "outputs/extension/schedule_<size>_<encoding>.json otherwise)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2, help="QAOA layers p")
    parser.add_argument("--shots", type=int, default=4000)
    parser.add_argument("--reads", type=int, default=1000, help="simulated-annealing reads")
    args = parser.parse_args()

    instance = load_instance(args.instance)
    size = instance.name.removeprefix("instance_")
    if args.encoding == "time":
        default_out = f"outputs/schedule_{size}.json"
    else:
        default_out = f"outputs/extension/schedule_{size}_{args.encoding}.json"
    out = Path(args.out or default_out)

    reference, proven = solve_cpsat(instance)
    print(f"instance {instance.name}: {len(instance.jobs)} jobs, machines {instance.machines}")
    print(f"CP-SAT reference makespan: {makespan(reference)} ({'proven optimal' if proven else 'not proven'})")

    if args.encoding == "time":
        lower, upper = instance.makespan_lower_bound(), sum(op.duration for op in instance.operations)
        print(f"horizon scan from lower bound {lower} (trivial upper bound {upper})")
        for horizon in range(lower, upper + 1):
            model = TimeIndexedQubo(instance, horizon)
            result = solve(model, args)
            found = result.schedule is not None
            print(f"  H={horizon}: {model.num_variables} variables -> {'feasible' if found else 'no feasible sample'} {result.stats}")
            if found:
                break
        else:
            raise SystemExit("no feasible schedule found up to the trivial upper bound")
    else:
        model = PermutationQubo(instance, implicit_last=args.encoding == "perm-implicit")
        result = solve(model, args)
        print(f"  {model.num_variables} variables, penalty {model.penalty} -> {result.stats}")
        if result.schedule is None:
            raise SystemExit("no feasible sample")

    errors = validate(instance, result.schedule)
    if errors:
        raise SystemExit("decoded schedule is infeasible: " + "; ".join(errors))
    cmax = makespan(result.schedule)
    print(f"{result.method}: makespan {cmax} (CP-SAT {makespan(reference)}, gap {cmax - makespan(reference)})")
    for s in sorted(result.schedule, key=lambda s: (s.start, s.machine)):
        print(f"  {s.job} {s.machine} [{s.start}, {s.end})")

    write_schedule(out, instance, result.schedule, result.method)
    gantt = out.with_name(out.stem.replace("schedule", "gantt") + ".png")
    plot_gantt(instance, result.schedule, gantt, title=f"{instance.name} - makespan {cmax}\n{result.method}")
    print(f"wrote {out} and {gantt}")


def solve(model, args):
    if args.solver == "qaoa":
        from erp.solvers.qaoa import solve_qaoa
        return solve_qaoa(model, reps=args.reps, shots=args.shots, seed=args.seed)
    if args.solver == "sa":
        from erp.solvers.annealing import solve_simulated_annealing
        return solve_simulated_annealing(model, num_reads=args.reads, seed=args.seed)
    from erp.solvers.brute_force import solve_brute_force
    return solve_brute_force(model)


if __name__ == "__main__":
    main()
