import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rel_gap(value, optimum) -> float:
    return 100 * (float(value) - float(optimum)) / max(float(optimum), 1.0)


def series(rows, key, value) -> dict[int, list[float]]:
    out = defaultdict(list)
    for r in rows:
        if key(r):
            out[int(r["n_jobs"])].append(value(r))
    return out


def plot(ax, per_n, label, color, style="-"):
    ns = sorted(per_n)
    if ns:
        ax.plot(ns, [np.mean(per_n[n]) for n in ns], label=label, color=color, linestyle=style, marker="o")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lns", type=Path, nargs="+",
                        default=[Path("results/lns_sa.csv"), Path("results/lns_veloxq.csv")])
    parser.add_argument("--direct", type=Path, default=Path("results/scaling.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/lns_vs_direct.png"))
    args = parser.parse_args()

    lns = [r for path in args.lns for r in read(path)]
    direct = [r for r in read(args.direct) if r["penalty_mode"] == "edd" and r["wt"] != ""]
    edd = {r["instance"]: r for r in lns}.values()

    fig, (gap_ax, size_ax, conv_ax) = plt.subplots(1, 3, figsize=(16, 4.5))
    plot(gap_ax, series(edd, bool, lambda r: rel_gap(r["edd_wt"], r["cpsat_wt_h"])), "EDD heuristic", "gray", ":")
    for solver, color in (("sa", "tab:blue"), ("veloxq", "tab:red")):
        name = "SA" if solver == "sa" else "VeloxQ"
        plot(gap_ax, series(direct, lambda r, s=solver: r["solver"] == s, lambda r: rel_gap(r["wt"], r["cpsat_wt_h"])),
             f"direct QUBO, {name}", color, "--")
        plot(gap_ax, series(lns, lambda r, s=solver: r["solver"] == s, lambda r: rel_gap(r["wt"], r["cpsat_wt_h"])),
             f"QUBO-LNS, {name}", color)
    gap_ax.set(title="Gap to CP-SAT optimum (mean)", xlabel="number of jobs", ylabel="gap [%]")

    plot(size_ax, series(direct, lambda r: r["solver"] == "sa" and r["run"] == "0", lambda r: int(r["num_vars"])),
         "direct QUBO", "black", "--")
    plot(size_ax, series(lns, bool, lambda r: int(r["max_vars"])), "QUBO-LNS, largest subproblem", "black")
    size_ax.set(title="Variables per QUBO", xlabel="number of jobs", ylabel="variables", yscale="log")

    largest = max((int(r["n_jobs"]) for r in lns), default=0)
    for r in lns:
        if int(r["n_jobs"]) == largest and r["instance"].endswith("_s0"):
            history = [int(r["edd_wt"]), *json.loads(r["history"])]
            color = "tab:blue" if r["solver"] == "sa" else "tab:red"
            conv_ax.plot(history, color=color, alpha=0.7, label=f"{r['solver']} run {r['run']}")
            conv_ax.axhline(int(r["cpsat_wt_h"]), color="gray", linestyle=":")
    conv_ax.set(title=f"Convergence, {largest} jobs (dotted: CP-SAT)", xlabel="iteration",
                ylabel="weighted tardiness")

    for ax in (gap_ax, size_ax, conv_ax):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
