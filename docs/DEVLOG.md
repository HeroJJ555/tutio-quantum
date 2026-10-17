# Development Log

## 2026-10-16 (Fri) - kickoff, plan, Stage 1 baseline

**Plan.** After the 18:00 kickoff we read the challenge and decided the order of work: first a
complete, verified baseline (Instances A and B), only then the Stage 2 extension.

**Numbers checked before coding** (exhaustive enumeration of permutations / start-time windows):
- 2x2: optimum makespan 4 (Z2 before Z1); 16 naive variables, 8 after start-window pruning at H=4,
  and at H=4 exactly one feasible schedule exists.
- 3x3: optimum 8 (reached by 4 of 6 job orders); 72 naive / 39 pruned variables at H=8;
  no feasible schedule at H=7.

**Design decisions.**
- *Encoding:* time-indexed `x[o,t]` with start-window pruning - the standard, exact formulation;
  compact encodings are planned as the Stage 2 extension.
- *Makespan:* instead of a max() objective (not quadratic) we solve a sequence of feasibility QUBOs
  over H, starting from a lower bound. Rejected for now: auxiliary one-hot "makespan = h" variables
  (exact, but more qubits for the baseline).
- *Tie-break:* small linear term `eps * sum of completion times`, `eps = 1/(#jobs*H+1)` so the whole
  term stays < 1 = cost of one violation.
- *Penalty weights:* P = 1 for all constraints (every violation >= 1, objective < 1). Per-constraint
  tuning for QAOA postponed.
- *Solvers:* QAOA (Qiskit state-vector) for A, simulated annealing for B (39 qubits are out of reach
  for a state-vector simulator); CP-SAT as an independent classical reference; VeloxQ token requested.

**Problems and fixes.**
- The lower bound we first estimated for B (7, from the milling machine) was too weak; the lathe
  gives 5 h of work + at least 3 h of F + S after the last turning = **8**, so the horizon scan
  starts directly at the optimum. Bound implemented in `Instance.makespan_lower_bound`.
- First test run took 6 min: the energy was evaluated over all QUBO terms for ~3.4 M assignments.
  Fixed by summing only over pairs of set bits; dropped the H=9 enumeration case -> tests run in ~30 s.
- Python 3.14: all dependencies (qiskit 2.6, ortools, dwave-samplers) installed fine, no need for 3.12.

**Results.**
- A: QAOA finds the optimal schedule (makespan 4). Probability of sampling the optimum:
  p=1: 2.4 %, p=2: 3.6 %, p=3: 3.6 % (seed 1, 4000 shots) vs 1/256 = 0.4 % for uniform sampling,
  i.e. ~9x amplification. p=2 chosen (p=3 gives no gain and takes longer).
- B: simulated annealing, 1000 reads -> makespan 8, 94 % of reads feasible.
- Both match CP-SAT (proven optimal). 15 tests pass.

**Next:** VeloxQ integration for B; choose the Stage 2 extension (qubit-reduction ladder +
one realistic constraint, e.g. due dates / weighted tardiness); improve QAOA success probability
(penalty tuning, XY mixer, parameter initialisation).

## 2026-10-17 (Sat) - extension: compact encodings, due dates, VeloxQ, scaling, QUBO-LNS

**Extension chosen:** one realistic constraint done fully (due dates + weighted tardiness), the
qubit-reduction ladder on top of it, and a scaling study with VeloxQ. Rejected: setup times, a second
milling machine and outages as separate features - depth over count.

**Permutation encoding (morning).** 3x3 as a job order: 9 variables, 6 with an implicit last
position. For n = 3 the first two positions fix the order, so the exact makespan is quadratic.
QAOA p=2 now solves instance B too: optimum in 19.7 % (9 qubits) and 37.6 % (6 qubits) of shots.
Decision: present it as the smallest correct QUBO, but state its price (all n! orders evaluated
classically, higher-order terms for n > 3).

**Due dates and weighted tardiness.** Optional `due_date` / `weight` per job in the JSON. In the
time-indexed QUBO tardiness is linear in the last-operation variables, so no new variables. We
searched due dates for which the tardiness optimum and the makespan optimum differ and chose
`instance_3x3_due.json` (Z3 = key customer, weight 3): optimum 2 at makespan 9, while H=8 can only
give 3. The realism costs horizon (39 -> 48 variables), not variables per constraint. Problem: the
penalty must now exceed the objective; with the worst-case P QAOA on the permutation QUBOs drops
to ~3.5 % optimal shots.

**VeloxQ.** Integrated through `veloxq-sdk`: the QUBO is sent as a BINARY `dimod` model with its
offset. Issue: VeloxQ returns variable labels as strings ("0", "1", ...), first run crashed with
KeyError - fixed in the sample decoding. Energies match ours exactly. On the baseline 3x3 QUBO:
makespan 8 in 4.1 s, 98 % feasible reads; on the tardiness QUBO (H=9): optimum 2. Decision: keep
baseline B on simulated annealing, use VeloxQ in the extension.

**Scaling experiment (afternoon).** Random instances with 3-10 jobs, 3 instances x 3 runs, SA and
VeloxQ, two penalty strategies. Findings:
- `P = E(EDD schedule) + 1` is still provably sufficient and several times smaller than the
  worst-case P; it lifts 4-job optimum hits from 1/9 to 9/9 (SA) and 6/9 to 9/9 (VeloxQ).
- The direct time-indexed QUBO hits a wall: no optimum from 6 jobs on (250+ variables), 84-113 %
  gap at 10 jobs (699 variables), about as good as the EDD heuristic.

**Refactoring.** Objective as an enum, QUBO base class with penalty helpers, explicit start
windows in the time-indexed QUBO, a single solver dispatch for CLI and experiments, ruff lint.
All outputs were re-checked to be identical before and after.

**QUBO-LNS (evening).** Idea: keep every QUBO small instead of looking for a stronger solver. Free
4 jobs, keep the rest, remove their machine time from the freed jobs' windows, solve the small QUBO,
accept only improvements. Iterations:
1. First version improved EDD but stalled at 10 jobs (109 -> 94, optimum 68). Diagnosis: the
   subproblems were solved well (200 and 2000 SA reads gave the same energy), but fixed jobs left
   gaps and the same subsets were chosen again and again.
2. Added a left shift after every accepted move and random neighbours among the 2k jobs finishing
   closest to the pivot, plus a list of subsets that did not help since the last improvement.
   10 jobs: 109 -> 68 = optimum, largest subproblem 125 variables instead of 699.
Results (40 iterations): mean gap 0-12 % up to 12 jobs with SA; with VeloxQ the optimum in 9 of the
first 10 instances (5-8 jobs). Subproblems stay at ~110-160 variables for every n.

**Problems.** The VeloxQ API timed out twice in the evening (connection time ~19 s, other sites
fine) - most likely load from other teams. Added retries with back-off and made the LNS experiment
resumable; VeloxQ runs for 10 and 12 jobs are being repeated.

**Next:** finish VeloxQ LNS runs, final plots, video and presentation.
