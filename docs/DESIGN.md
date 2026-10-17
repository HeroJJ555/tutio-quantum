# Project Design

## 1. Optimisation problem

MetalTech has three machines - lathe T, milling F, welding S. Each job Z1..Zn consists of one
operation per machine in the fixed order T -> F -> S, with known integer durations (hours).
A machine processes one operation at a time; operations are not interrupted.
**Goal:** choose start times of all operations so that the completion time of the last operation
(the makespan, C_max) is minimal. This is the flow-shop special case of job-shop scheduling.

## 2. Data model

| Concept | Representation |
|---|---|
| Operation `o` | (job, machine, duration `d_o`, step = position in the job) |
| Horizon `H` | maximal allowed makespan; time is discrete, start times `t` are integers |
| head(o) | total duration of the operations before `o` in its job -> earliest start |
| tail(o) | duration of `o` + all later operations of its job -> latest start = H - tail(o) |
| **Variable** `x[o,t]` | 1 if operation `o` starts at time `t`, for `t` in the window [head(o), H - tail(o)] |

A *state* is one bitstring over all `x[o,t]`; it encodes a schedule iff every operation has exactly
one bit set. Start-window pruning keeps only start times that can still finish by `H`:

| Instance | Naive `#ops * H` | With windows |
|---|---|---|
| A (2x2, H=4) | 16 | **8** |
| B (3x3, H=8) | 72 | **39** |

**Why H is a parameter:** the time-indexed encoding needs a finite time grid. H fixes the number of
variables and turns the makespan question into "is there a schedule finishing by H?".

## 3. Constraints

All constraints become quadratic penalties with weight P (we use P = 1, see 5).

1. **Each operation starts exactly once** - `(sum_t x[o,t] - 1)^2`.
   Using `x^2 = x`: `-sum_t x[o,t] + 2 * sum_{t<t'} x[o,t] x[o,t'] + 1`.
2. **Technological order** (next operation `o'` of the same job) -
   `x[o,t] * x[o',t']` for every pair with `t' < t + d_o`.
3. **No machine overlap** (operations `o`, `o'` of different jobs on the same machine) -
   `x[o,t] * x[o',t']` for every pair whose intervals `[t, t+d_o)` and `[t', t'+d_o')` intersect.
4. **Makespan <= H** - enforced for free by the windows (no variable exists for a start that would end after H).

Dependencies: constraint 1 couples variables of one operation; 2 couples consecutive operations
of one job; 3 couples operations on one machine. Variables of unrelated operations interact only
through these chains.

## 4. Objective and the QUBO

```
E(x) = P * [ C1 + C2 + C3 ](x)  +  eps * sum_{j} sum_t x[last_j, t] * (t + d_last_j)
```

- The penalty part (linear terms from C1, quadratic terms from C1-C3, constant `P * #ops`) is 0
  exactly for feasible schedules with makespan <= H and >= 1 otherwise.
- The second part is a small **linear tie-break** (sum of job completion times). With
  `eps = 1 / (#jobs * H + 1)` it is always < 1, so it can never compensate a violated constraint;
  when several feasible schedules exist it prefers earlier completion.

**Makespan minimisation = horizon scan.** Starting from a lower bound, we build the QUBO for H,
solve it, and stop at the first H where the solver returns a feasible state. Lower bound used:
`max(longest job, max over machines of min head + machine load + min remaining tail)`.
For both baseline instances the bound already equals the optimum (A: 4, B: 8 - on the lathe the
total work is 5 h and after the last turning at least F + S = 3 h remain).

## 5. Feasibility and weights

- **Feasibility condition:** a zero-penalty state exists iff H >= optimal makespan. Exhaustive
  enumeration (`tests/`) gives: A - 0 feasible states at H=3, exactly 1 at H=4, 7 at H=5;
  B - 0 at H=7, 13 at H=8 (all with makespan 8).
- **Penalty weight P.** The ground state is feasible whenever every infeasible state costs more than
  some feasible one. An infeasible state costs at least P (one violated unit), so any
  `P > E(known feasible schedule)` is enough. For the makespan baseline the objective is only the
  tie-break (< 1), so P = 1. For other objectives see section 10.

## 6. Solvers

| Solver | Used for | Details |
|---|---|---|
| **QAOA** (Qiskit 2.6, state-vector simulator) | instance A, permutation QUBOs | Ising mapping `x = (1 - Z)/2`; cost operator normalised by its largest coefficient; `qaoa_ansatz` with X mixer, p = 2; COBYLA on the exact expectation value, 8 random restarts; 4000 shots; lowest-energy feasible sample returned |
| **Simulated annealing** (`dwave-samplers`) | instance B, scaling, LNS | 1000 reads (200 inside LNS) |
| **VeloxQ** (`veloxq-sdk`, H100 backend) | instance B, scaling, LNS | the QUBO is sent as a BINARY `dimod` model with its offset, so the returned energies equal ours; 4096 repetitions x 5000 steps; retried on network errors |
| **CP-SAT** (OR-Tools) | every result | exact, built directly from the instance - independent reference; for tardiness ties are broken by makespan; an optional horizon gives the optimum of the same model as a given QUBO |
| **Brute force** | QUBOs up to ~22 variables | all 2^n states - certifies the QUBO ground state |

QAOA on the 39-variable QUBO of instance B is out of reach for a state-vector simulator (2^39
amplitudes); compact encodings (section 9) bring 3x3 down to 6-9 qubits.

## 7. Verification

- `schedule.validate` checks every output independently of the QUBO (one start per operation,
  durations, T -> F -> S order, no machine overlap); the CLI refuses to write an infeasible schedule.
- Tests: the CP-SAT optimum encodes to a zero-penalty state; schedules violating each constraint
  type get positive penalty; on every one-hot assignment "penalty = 0" agrees with the validator;
  exhaustive search returns the optimum for every small QUBO; for the tardiness QUBO all feasible
  assignments are enumerated and the minimum matches CP-SAT at H=8 and H=9; LNS never worsens the
  schedule.

## 8. Assumptions and simplifications

- Integer durations, unit time grid of 1 hour; no setup times or release dates.
- Fixed technological order T -> F -> S for every job.
- All jobs available at t = 0; due dates and weights are integers.

## 9. Permutation encoding (qubit reduction)

For makespan on two or three machines some optimal schedule processes the jobs in the same order on
every machine, so it is enough to choose a job order.

- Variables `y[j,p] = 1` iff job j is at position p; constraints: each position one job, each job one
  position (one-hot penalties). 3x3: **9 variables**.
- The decoder builds the schedule from the order (every operation as early as possible).
- The first n-1 positions determine the order, so the cost of an order is a product of n-1 variables:
  `sum over orders C(order) * y[order_1, 1] * ... * y[order_{n-1}, n-1]`. For n = 3 this is
  **quadratic**. The cost is `C(order) - lower bound`, and P = 1 + max cost.
- **Implicit last position:** drop the variables of position n (the remaining job goes there) and
  require "at most one position" per job: 3x3 needs **6 variables**.
- **Price:** building the objective evaluates all n! orders classically, and for n > 3 the exact
  objective needs higher-order terms. We report it as the smallest *correct* QUBO for the 3x3
  baseline, not as a scalable method.

| 3x3 encoding | Variables | QAOA p=2, probability of the optimum |
|---|---|---|
| time-indexed, start windows (H=8) | 39 | not simulable |
| permutation | 9 | 19.7 % (random guess 0.8 %) |
| permutation, implicit last | 6 | 37.6 % (random guess 6.3 %) |

## 10. Due dates and weighted tardiness

Each job may have a due date `d_j` and a weight `w_j`; tardiness `T_j = max(0, C_j - d_j)`;
objective `sum_j w_j T_j`, makespan only breaks ties.

- **Time-indexed:** the completion time of job j is `t + d` of its last operation, so
  `sum_j sum_t w_j * max(0, t + d - d_j) * x[last_j, t]` is a **linear** term. No extra variables.
- **The price is the horizon.** On `instance_3x3_due.json` (due 5 / 7 / 7, weights 1 / 1 / 3) the
  tardiness optimum (2) finishes at 9; the makespan-optimal H=8 (39 variables) can only reach 3,
  H=9 (48 variables) reaches the optimum.
- **Permutation:** the same order-cost construction with `cost = WT(order) + (C - LB)/(spread + 1)`.
  Permutation schedules are not guaranteed optimal for tardiness on three machines, so we compare
  with CP-SAT; on our instance both reach 2.
- **Penalty:** a default `P = 1 + sum_j w_j * max(0, H - d_j)` (worst possible objective) is always
  valid but large. A much smaller valid choice is `P = E(EDD schedule) + 1` (earliest due date order,
  section 11). With large P QAOA on the 6/9-qubit tardiness QUBOs finds the optimum in ~3.5 % of shots
  instead of 20-38 % for makespan - the landscape is dominated by the penalty.

| Constraint / objective | Variables, naive time-indexed | Variables, our encoding |
|---|---|---|
| baseline makespan 3x3 | 72 | 39 (windows), 9 / 6 (permutation) |
| + weighted tardiness | +0 (linear term), needs H=9: 48 | +0 (permutation 9 / 6) |
| + fixed operations / blocked machine time (LNS) | +0 | removes variables |

## 11. Scaling experiment

Random instances (`erp/generator.py`): n jobs on T -> F -> S, durations 1-3, weights 1-3, due date =
own work + random slack. Horizon H = makespan of the EDD order + 1. For n = 3, 4, 5, 6, 7, 8, 10, three
instances each, three runs per solver and penalty strategy; every result is compared with CP-SAT on
the same model (makespan <= H). Data: `results/scaling.csv`, plots `results/scaling_*.png`.

- QUBO size grows from 66 to 699 variables (and thousands of quadratic terms).
- **Penalty strategy matters:** with `P = E(EDD) + 1` both solvers find the optimum in 9/9 runs at
  4 jobs, against 1/9 (SA) and 6/9 (VeloxQ) with the worst-case P; the gap is lower at every size.
- **Wall:** from 5-6 jobs no run reaches the optimum; at 10 jobs the mean gap is 84 % (SA) and 113 %
  (VeloxQ), around the EDD heuristic itself. VeloxQ is faster (3-7 s vs 2-34 s per run, network
  included) but the share of feasible samples drops below 1 % at 700 variables.

## 12. QUBO large-neighbourhood search

The wall comes from the size of a single QUBO, so we keep every QUBO small (`erp/lns.py`):

1. Start from the EDD schedule, shifted left (every operation as early as possible in the same
   machine and job order - this never increases tardiness).
2. Pick 4 jobs: a pivot (biased towards jobs with high `w_j T_j`) and random jobs among those
   finishing closest to it. Subsets that did not help are not repeated until the next improvement.
3. All other operations stay fixed. The freed operations get start windows from which the time
   occupied by fixed operations on the same machine is removed - fixed work acts like blocked machine
   time, so the subproblem QUBO is the same time-indexed model over smaller windows.
4. P = E(current sub-schedule) + 1, which keeps the ground state feasible (section 5).
5. Solve with VeloxQ or simulated annealing; accept the result only if its energy is lower, then
   shift left again.

Every subproblem has at most ~160 variables regardless of n. Results (`results/lns_*.csv`,
`results/lns_vs_direct.png`, 40 iterations):

| Jobs | EDD gap | Direct QUBO gap (SA / VeloxQ) | LNS gap (SA / VeloxQ) | LNS optimum hits (SA / VeloxQ) |
|---|---|---|---|---|
| 5 | 37.5 % | 8.8 % / 12.8 % | 0 % / 0 % | 9/9 / 3/3 |
| 6 | 42.6 % | 17.3 % / 21.4 % | 3.9 % / 0 % | 6/9 / 3/3 |
| 7 | 78.1 % | 44.1 % / 39.2 % | 1.1 % / 2.0 % | 7/9 / 2/3 |
| 8 | 75.0 % | 71.4 % / 82.1 % | 11.9 % / 0 % | 4/9 / 1/1 |
| 10 | 77.1 % | 84.2 % / 112.8 % | 10.4 % / pending | 3/9 / pending |
| 12 | 52.1 % | - | 6.3 % / pending | 0/9 / pending |

The same mechanism handles machine outages: an outage is one more blocked interval in step 3.

## References

- D. Venturelli, D. Marchand, G. Rojo - *Quantum Annealing Implementation of Job-Shop Scheduling* (time-indexed formulation, variable pruning).
- A. Lucas - *Ising formulations of many NP problems*.
- P. Shaw - *Using constraint programming and local search methods to solve vehicle routing problems* (large neighbourhood search).
- R. W. Conway, W. L. Maxwell, L. W. Miller - *Theory of Scheduling* (permutation schedules for flow shops with up to three machines).
