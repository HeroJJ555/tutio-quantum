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
