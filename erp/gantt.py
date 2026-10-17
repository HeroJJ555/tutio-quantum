from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erp.instance import Instance
from erp.schedule import Schedule, makespan


def plot_gantt(instance: Instance, schedule: Schedule, path: str | Path, title: str = "") -> None:
    machines = instance.machines
    colors = dict(zip(instance.jobs, plt.get_cmap("tab10").colors, strict=False))
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(machines) + 1.2))
    for s in schedule:
        row = len(machines) - 1 - machines.index(s.machine)
        ax.barh(row, s.end - s.start, left=s.start, color=colors[s.job], edgecolor="black")
        ax.text((s.start + s.end) / 2, row, s.job, ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks(range(len(machines)), list(reversed(machines)))
    ax.set_xticks(range(makespan(schedule) + 1))
    ax.set_xlabel("time [h]")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title or f"{instance.name} - makespan {makespan(schedule)}", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
