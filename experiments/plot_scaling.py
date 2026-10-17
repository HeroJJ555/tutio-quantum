import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES_STYLE = {
    ("sa", "worst"): ("SA, penalty worst-case", "tab:blue", "--"),
    ("sa", "edd"): ("SA, penalty from EDD", "tab:blue", "-"),
    ("veloxq", "worst"): ("VeloxQ, penalty worst-case", "tab:red", "--"),
    ("veloxq", "edd"): ("VeloxQ, penalty from EDD", "tab:red", "-"),
}


def load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n_jobs"] = int(r["n_jobs"])
        r["solved"] = r["wt"] != ""
        r["optimal"] = r["solved"] and float(r["gap_h"]) == 0
        if r["solved"]:
            r["rel_gap"] = 100 * float(r["gap_h"]) / max(float(r["cpsat_wt_h"]), 1.0)
    return rows


def by_series(rows, value) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r["solver"], r["penalty_mode"])][r["n_jobs"]].append(value(r))
    return out


def plot_series(ax, data, title, ylabel, error_bars=True):
    for key, per_n in sorted(data.items()):
        label, color, style = SERIES_STYLE.get(key, (str(key), None, "-"))
        ns = sorted(per_n)
        means = [np.mean(per_n[n]) for n in ns]
        stds = [np.std(per_n[n]) for n in ns] if error_bars else None
        ax.errorbar(ns, means, yerr=stds, label=label, color=color, linestyle=style, marker="o", capsize=3)
    ax.set(title=title, xlabel="number of jobs", ylabel=ylabel)
    ax.grid(alpha=0.3)


def plot_quality(rows, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    solved = [r for r in rows if r["solved"]]
    plot_series(axes[0, 0], by_series(solved, lambda r: r["rel_gap"]),
                "Solution quality (mean +- std)", "gap to CP-SAT optimum [%]")
    plot_series(axes[0, 1], by_series(rows, lambda r: 100 * r["optimal"]),
                "Optimum found", "runs reaching the optimum [%]", error_bars=False)
    plot_series(axes[1, 0], by_series(rows, lambda r: 100 * float(r["feasible_fraction"])),
                "Share of feasible samples", "feasible reads [%]")
    axes[1, 0].set_yscale("symlog", linthresh=1)
    plot_series(axes[1, 1], by_series(rows, lambda r: float(r["time_s"])),
                "Time per run (VeloxQ includes network and queue)", "wall time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Weighted-tardiness time-indexed QUBO: scaling with the number of jobs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_size(rows, path: Path) -> None:
    sizes = defaultdict(dict)
    for r in rows:
        sizes[r["n_jobs"]][r["instance"]] = (int(r["num_vars"]), int(r["num_quadratic"]))
    ns = sorted(sizes)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (label, marker) in enumerate([("variables", "o"), ("quadratic terms", "s")]):
        ax.plot(ns, [np.mean([v[k] for v in sizes[n].values()]) for n in ns], marker=marker, label=label)
    ax.set(yscale="log", xlabel="number of jobs", title="QUBO size (time-indexed, weighted tardiness)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_table(rows) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["n_jobs"], r["solver"], r["penalty_mode"])].append(r)
    lines = ["| jobs | vars | solver | penalty | mean gap [%] | optimum found | feasible reads | time [s] |",
             "|---|---|---|---|---|---|---|---|"]
    for (n, solver, mode), rs in sorted(groups.items()):
        solved = [r for r in rs if r["solved"]]
        gap = f"{np.mean([r['rel_gap'] for r in solved]):.1f}" if solved else "-"
        lines.append(
            f"| {n} | {np.mean([int(r['num_vars']) for r in rs]):.0f} | {solver} | {mode} | {gap} | "
            f"{sum(r['optimal'] for r in rs)}/{len(rs)} | "
            f"{100 * np.mean([float(r['feasible_fraction']) for r in rs]):.1f}% | "
            f"{np.mean([float(r['time_s']) for r in rs]):.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/scaling.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = load(args.csv)
    plot_quality(rows, args.out_dir / "scaling_quality.png")
    plot_size(rows, args.out_dir / "scaling_size.png")
    table = summary_table(rows)
    (args.out_dir / "scaling_summary.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
