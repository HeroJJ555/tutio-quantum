# tutio-quantum

QPoland Quantum Hackathon 2026 - ERP Challenge: production scheduling at MetalTech Ltd.
(flow shop: lathe **T** -> milling **F** -> welding **S**, minimise makespan) formulated as a
QUBO and solved with QAOA on a quantum simulator and with a QUBO solver.

**Team:** Jan Jakubowski, Wiktor Alisch

## Results (Stage 1 baseline)

| Instance | Method | QUBO size | Makespan | Classical reference (CP-SAT) |
|---|---|---|---|---|
| A - 2 jobs x 2 machines | QAOA p=2, Qiskit state-vector simulator | 8 qubits (H=4) | **4** | 4 (proven optimal) |
| B - 3 jobs x 3 machines | Simulated annealing on the QUBO | 39 variables (H=8) | **8** | 8 (proven optimal) |

Schedules: [`outputs/schedule_2x2.json`](outputs/schedule_2x2.json), [`outputs/schedule_3x3.json`](outputs/schedule_3x3.json)
(format from Section 14 of the challenge). Gantt charts: `outputs/gantt_2x2.png`, `outputs/gantt_3x3.png`.

Stage 2 extension: work in progress.

## How to run

Requires Python 3.12+ (tested on 3.14).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt        # Linux / macOS
```

VeloxQ: copy `.env.example` to `.env` and set `VELOXQ_TOKEN` (`.env` is gitignored).

Instance A (QAOA, ~15 s):

```bash
.venv/Scripts/python -m erp.cli --instance data/instance_2x2.json --solver qaoa
```

Instance B (simulated annealing, a few seconds):

```bash
.venv/Scripts/python -m erp.cli --instance data/instance_3x3.json --solver sa
```

Each run prints the CP-SAT reference, the horizon scan, the decoded schedule and writes the JSON + Gantt chart to `outputs/`.
Options: `--seed`, `--reps` (QAOA layers), `--shots`, `--reads` (annealing reads), `--solver brute` (exhaustive, small QUBOs only).

Tests (QUBO correctness against an independent validator and CP-SAT, ~30 s):

```bash
.venv/Scripts/python -m pytest
```

## Repository layout

```
data/        input instances (JSON, challenge format)
erp/         instance.py           - loading instances, lower bound
             schedule.py           - independent feasibility validator, JSON export
             qubo_time_indexed.py  - time-indexed QUBO, encoder/decoder
             solvers/              - qaoa.py, annealing.py, brute_force.py, cpsat.py (reference)
             gantt.py, cli.py
outputs/     schedule_2x2.json, schedule_3x3.json, Gantt charts
tests/       correctness tests
docs/        DEVLOG.md (development log), DESIGN.md (project design)
```

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) - data model, QUBO formulation, solvers, assumptions
- [docs/DEVLOG.md](docs/DEVLOG.md) - dated development log

## How AI was used

We used an AI coding assistant to speed up writing tedious parts of code and documentation.
Formulations and design decisions are ours and are described in `docs/DESIGN.md`.

## License

MIT - see [LICENSE](LICENSE).
