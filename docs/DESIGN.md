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
- **Penalty weight P = 1** for all constraint types: every violation costs at least 1 and the
  objective is < 1, so the ground state is always feasible if H is feasible. Tuning P separately per
  constraint (to shape the landscape for QAOA) is left for Stage 2.

## 6. Solvers

| Instance | Solver | Details |
|---|---|---|
| A | **QAOA** (Qiskit 2.6, state-vector simulator) | Ising mapping `x = (1 - Z)/2`; cost operator normalised by its largest coefficient; `qaoa_ansatz` with X mixer, p = 2; COBYLA on the exact expectation value, 8 random restarts; final sampling 4000 shots; lowest-energy feasible sample returned |
| B | **Simulated annealing** (D-Wave Ocean `dwave-samplers`) | 1000 reads on the 39-variable QUBO |
| both | **CP-SAT** (OR-Tools) | exact, built directly from the instance - independent reference |
| small | **Brute force** | all 2^n states - certifies the QUBO ground state |

QAOA on the 39-variable QUBO of instance B is not feasible on a state-vector simulator (2^39 amplitudes);
this is why B uses a QUBO solver (VeloxQ to be integrated once the access token arrives).

## 7. Verification

- `schedule.validate` checks every output independently of the QUBO (one start per operation,
  durations, T -> F -> S order, no machine overlap).
- Tests: the CP-SAT optimum encodes to a zero-penalty state; schedules violating each constraint
  type get positive penalty; on every one-hot assignment "penalty = 0" agrees with the validator;
  exhaustive search of the 2x2 QUBO returns a makespan-4 schedule.

## 8. Assumptions and simplifications

- Integer durations, unit time grid of 1 hour; no setup times, release dates or breakdowns.
- Fixed technological order T -> F -> S for every job (configurable order is a Stage 2 topic).
- All jobs available at t = 0.

## References

- D. Venturelli, D. Marchand, G. Rojo - *Quantum Annealing Implementation of Job-Shop Scheduling* (time-indexed formulation, variable pruning).
- A. Lucas - *Ising formulations of many NP problems*.
