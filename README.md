# tutio-quantum

QPoland Quantum Hackathon 2026 - ERP Challenge: production scheduling at MetalTech Ltd.
(flow shop: lathe **T** -> milling **F** -> welding **S**) formulated as QUBO and solved with QAOA
on a quantum simulator, with simulated annealing and with VeloxQ.

**Team:** Jan Jakubowski, Wiktor Alisch

## Stage 1 - baseline

| Instance | Method | QUBO size | Makespan | CP-SAT reference |
|---|---|---|---|---|
| A - 2 jobs x 2 machines | QAOA p=2, Qiskit state-vector simulator | 8 qubits (H=4) | **4** | 4 (proven optimal) |
| B - 3 jobs x 3 machines | Simulated annealing on the QUBO | 39 variables (H=8) | **8** | 8 (proven optimal) |

Schedules in the Section 14 format: [`outputs/schedule_2x2.json`](outputs/schedule_2x2.json),
[`outputs/schedule_3x3.json`](outputs/schedule_3x3.json); Gantt charts next to them.

## Stage 2 - extension

**Due dates and weighted tardiness.** The urgent order from the key customer gets a due date and a
weight. In the time-indexed QUBO tardiness is a *linear* term - it costs no extra variables, only a
longer horizon. On [`data/instance_3x3_due.json`](data/instance_3x3_due.json) (Z3 = key customer,
weight 3) the tardiness optimum is 2 and needs makespan 9; the makespan-optimal horizon H=8 can
only reach 3.

**Qubit reduction for 3x3** (every row verified against CP-SAT):

| Encoding | Variables | Makespan objective | Weighted tardiness |
|---|---|---|---|
| time-indexed, naive | 72 | exact | exact |
| time-indexed, start windows | 39 (H=8) / 48 (H=9) | exact | exact at H=9 |
| permutation (job x position) | 9 | exact, QAOA p(opt) 19.7 % | exact, QAOA p(opt) 3.5 % |
| permutation, implicit last position | **6** | exact, QAOA p(opt) 37.6 % | exact, QAOA p(opt) 3.6 % |

**Scaling wall and QUBO-LNS.** On random instances with 3-12 jobs the direct time-indexed QUBO
(66 -> 699 variables) stops finding the optimum at 5-6 jobs for both simulated annealing and VeloxQ
([`results/scaling_quality.png`](results/scaling_quality.png)). Our answer is a large-neighbourhood
search in which every step re-optimises 4 jobs with a small QUBO (at most ~160 variables) solved by
VeloxQ or simulated annealing, while the remaining jobs block their machine time:

| Jobs | Direct QUBO gap (SA / VeloxQ) | QUBO-LNS gap (SA / VeloxQ) | Largest LNS subproblem |
|---|---|---|---|
| 5 | 8.8 % / 12.8 % | 0 % / 0 % | 139 vars |
| 6 | 17.3 % / 21.4 % | 3.9 % / 0 % | 144 vars |
| 8 | 71.4 % / 82.1 % | 11.9 % / 0 % | 157 vars |
| 10 | 84.2 % / 112.8 % | 10.4 % / pending | 150 vars |
| 12 | - | 6.3 % / pending | 161 vars |

Gap = mean distance to the CP-SAT optimum of the same model. Plots: [`results/lns_vs_direct.png`](results/lns_vs_direct.png).

## How to run

Requires Python 3.12+ (tested on 3.14).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt        # Linux / macOS
```

VeloxQ: copy `.env.example` to `.env` and set `VELOX_TOKEN` (`.env` is gitignored).

Baseline:

```bash
.venv/Scripts/python -m erp.cli --instance data/instance_2x2.json --solver qaoa
.venv/Scripts/python -m erp.cli --instance data/instance_3x3.json --solver sa
```

Extension:

```bash
.venv/Scripts/python -m erp.cli --instance data/instance_3x3.json --solver qaoa --encoding perm-implicit
.venv/Scripts/python -m erp.cli --instance data/instance_3x3_due.json --solver veloxq --objective tardiness --horizon 9
.venv/Scripts/python -m experiments.scaling --sizes 3 4 5 6 7 8 10 --solvers sa veloxq
.venv/Scripts/python -m experiments.lns --solvers sa --runs 3 --out results/lns_sa.csv
.venv/Scripts/python -m experiments.plot_scaling
.venv/Scripts/python -m experiments.plot_lns
```

CLI options: `--solver {qaoa,sa,veloxq,brute}`, `--encoding {time,perm,perm-implicit}`,
`--objective {makespan,tardiness}`, `--horizon`, `--seed`, `--reps`, `--shots`, `--reads`.
Baseline runs write to `outputs/`, extension runs to `outputs/extension/`.

Tests (QUBO correctness against an independent validator and CP-SAT, ~1 min) and lint:

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check erp experiments tests
```

## Repository layout

```
data/          instances (challenge JSON format, optional due_date / weight per job)
erp/           instance.py, schedule.py      - data model, validator, JSON export, left shift
               objective.py                  - makespan / weighted tardiness
               qubo_model.py                 - QUBO base class (penalty and objective parts)
               qubo_time_indexed.py          - time-indexed QUBO over explicit start windows
               qubo_permutation.py           - permutation QUBO (9 and 6 variables for 3x3)
               lns.py                        - QUBO large-neighbourhood search
               generator.py, gantt.py, cli.py, env.py
               solvers/                      - qaoa, annealing, veloxq, brute_force, cpsat (reference)
experiments/   scaling.py, lns.py, plot_*.py - experiments behind results/
outputs/       baseline schedules (+ extension/)
results/       experiment CSVs, logs, plots, summary tables
tests/         correctness tests
docs/          DEVLOG.md, DESIGN.md
```

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) - data model, QUBO formulations, solvers, LNS, verification, assumptions
- [docs/DEVLOG.md](docs/DEVLOG.md) - dated development log

## How AI was used

We used an AI coding assistant to speed up writing tedious parts of code and documentation.
Formulations and design decisions are ours and are described in `docs/DESIGN.md`.

## License

MIT - see [LICENSE](LICENSE).
