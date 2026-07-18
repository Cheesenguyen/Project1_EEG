# Standard Convolution Architecture & Dataflow Simulator

## 1. Core Objective

This project is a C-based hardware simulator designed to evaluate the
performance of a Standard Convolution computing architecture. It models
how tiling, spatial unrolling, and stationary (dataflow) strategies affect
memory traffic and compute cycles, and cross-checks the numerical
correctness of the simulated convolution output against a Python
(NumPy/SciPy) reference implementation.

## 2. Hardware Architecture Model

The simulated system models a 3-tier memory and compute hierarchy:

- **DRAM (Off-chip Memory)**: Stores the complete original data (Input
  Feature Maps, Weights) and the final Output. It has high latency and
  high energy consumption.
- **SRAM (On-chip Buffer)**: Fast access but limited capacity. It is
  logically partitioned into three independent regions:
  - **Input Buffer**: Stores the current Input Tile.
  - **Weight Buffer**: Stores the current Weight Tile.
  - **Psum/Output Buffer (PS Buffer)**: Temporarily holds intermediate
    Partial Sums (Psums) and completed Output values before flushing
    them to DRAM.
- **PE (Processing Elements) Array**: Modeled as an ideal parallel
  computation "black box."
  - **Fixed Quantity**: The array has a **fixed physical size of
    `PE_MAX = 16` PEs**. This is a hardware constraint, not a
    dynamically allocated resource. For a given cycle, the number of
    active PEs is `pe_used = min(N_candidates, PE_MAX)`, where
    `N_candidates` is however many data-independent MACs the current
    unrolling scheme wants to issue in parallel. If `N_candidates`
    exceeds 16, the workload is split into multiple sequential
    **batches** (`n_pe_batches = ceil(N_candidates / PE_MAX)`), and each
    batch consumes 1 additional cycle.
  - **Structure**: Each PE contains only a tiny Accumulator register to
    hold the immediate Psum for the current cycle; it has no long-term
    storage capability.
  - **Clock Cycle Rule**: Whenever a set of up to `PE_MAX` (16)
    data-independent Multiply-Accumulate (MAC) operations are mapped to
    the PE array, they are executed SIMULTANEOUSLY. This single parallel
    execution event (one batch) is counted as **1 Cycle**.

## 3. Simulation Workflow & Tasks

The C program utilizes nested loops (`th → tw → tk → tc`) to simulate
three primary phases of data movement and processing for every tile:

### Phase 1: Tiling (DRAM <-> SRAM)

- **Task**: Partition the original data into tile dimensions
  `TH, TW, TC, TK`. These tile sizes are **supplied by the user** through
  the config file — the simulator does not search for or auto-tune them.
- **Objective**: Let the user manually explore how different tile sizes
  trade off DRAM access frequency against the SRAM capacity required.
  The simulator reports the resulting metrics for whatever tile sizes
  are given, so the user can compare configurations and pick a suitable
  one — it does not run an internal optimizer.

### Phase 2: Spatial Unrolling & Data Fetching (SRAM -> PE Array)

- **Task**: Determine the number of active PEs (capped at
  `PE_MAX = 16`) and the spatial "shape" of the data fetched for each
  compute cycle. Two unrolling schemes are implemented, and **both are
  run for every test case** (results reported side by side):
  - **Unroll TK** (`pe_compute_tk`): parallelizes MACs across output
    channels K — up to 16 output channels computed at once for a fixed
    spatial position.
  - **Unroll TH×TW** (`pe_compute_thw`): parallelizes MACs across
    spatial output positions (OH × OW) — up to 16 output pixels
    computed at once for a fixed output channel.
- **Objective**: Maximize Spatial Reuse. Optimize the fetching shape so
  that multiple PEs can share or broadcast the same Input/Weight value
  read from SRAM within a single cycle, reducing SRAM read accesses.

### Phase 3: Stationary Strategies (PE Computation over Cycles)

- **Task**: Implement and simulate three dataflow strategies: Weight
  Stationary (WS), Input Stationary (IS), and Output Stationary (OS).
- **Objective**: Maximize Temporal Reuse. Evaluate how each strategy
  affects the system's reliance on the SRAM PS Buffer — specifically,
  how many Psum/Output slots must be held concurrently rather than
  round-tripped through the PS Buffer mid-computation (see "Num PS
  Buffers" in Table B).

Every case defined in the config file is simulated **twice** — once per
unrolling mode (TK and TH×TW) — and results for both are printed to the
terminal and written to two separate CSV files (`results_tk.csv` and
`results_thw.csv`).

## 4. Output Profiling Metrics

### Table A: Memory Hierarchy Profiling (Varying Tile Sizes)

| Metric | Description |
|---|---|
| DRAM Loads | Total number of **elements** read from DRAM for Inputs and Weights combined (`dram_input_loads + dram_weight_loads`). |
| DRAM Stores | Total number of **tile-store events** writing completed Output tiles back to DRAM — counted once per `(th, tw, tk)` tile, **not** per output element. |
| SRAM Input Traffic | Total bytes of Input tile data moved from DRAM into SRAM, accumulated over the entire run (`dram_input_loads × 4`). This is cumulative traffic, useful for comparing dataflow strategies (OS/WS/IS) by how much redundant re-fetching each one causes. |
| SRAM Weight Traffic | Total bytes of Weight tile data moved from DRAM into SRAM, accumulated over the entire run (`dram_weight_loads × 4`). Same interpretation as above. |
| SRAM Total | Sum of SRAM Input Traffic and SRAM Weight Traffic. |

> **Unit note**: `DRAM Loads` (input/weight) counts individual data
> **elements**, while `DRAM Stores` counts **tile-store events**. The two
> are not directly comparable without also knowing the tile size.

> **Note on "SRAM Input/Weight Traffic"**: these report *cumulative
> bytes moved into SRAM over the entire run* (`dram_*_loads × 4`), not a
> static per-tile buffer capacity. They are best used to compare how
> much redundant DRAM traffic each dataflow strategy (OS/WS/IS) causes,
> rather than as a fixed hardware sizing number.

### Table B: Compute Architecture Profiling (Varying Unrolling & Stationary Strategies)

| Metric | Description |
|---|---|
| Max PE Utilization (N) | The peak number of PEs actually mobilized in a single cycle across the run, `pe_used = min(N_candidates, PE_MAX)`. Since the PE array is physically fixed at 16, this metric reflects how efficiently a given config **utilizes** the fixed array (values close to 16 are efficient), not the hardware area/cost — area is constant across all configs. |
| Num PS Buffers | The number of Psum/Output **slots that must be held concurrently** for the current tile, which differs by strategy: for **OS**, only `pe_used` slots are needed, because each active PE keeps its own Psum resident in its accumulator across the full R×S×C reduction — nothing needs to be evicted to the PS Buffer mid-reduction; for **WS** and **IS**, the full tile of `TH_eff × TW_eff × TK_eff` slots must be resident in the SRAM PS Buffer simultaneously, since in these strategies the reduction is not completed within a single PE pass and partial sums must persist across passes. This is a structural "concurrent slots required" figure, not a running counter of individual evict/read-back events. |
| SRAM Input Reads | Total Input fetches from SRAM by the PE array. Used to evaluate spatial/temporal reuse of inputs — lower is better reuse. |
| SRAM Weight Reads | Total Weight fetches from SRAM by the PE array. Used to evaluate weight broadcasting efficiency — lower is better reuse. |
| Total Compute Cycles | Total number of parallel execution cycles triggered, accumulated as `n_pe_batches` for every PE-array invocation across the whole run. Represents the actual execution latency of the system. |

## 5. Correctness Verification (`verify.py`)

Alongside the profiling metrics, the project includes a Python
cross-check script, `verify.py`, that validates the **actual numerical
output values** of the C convolution (not the performance metrics):

- It reads the case definitions and the per-channel output statistics
  exported by the C program (`conv_stats.csv`, written by
  `conv_write_stats()`).
- It regenerates the **exact same pseudo-random Input/Weight tensors**
  used by the C simulator, using an identical Linear Congruential
  Generator (LCG) with the same seed (`42`) and update rule
  (`seed = seed * 1664525 + 1013904223`), so the Python and C runs
  operate on bit-for-bit comparable test data.
- It computes a reference convolution using
  `scipy.signal.correlate2d` (`mode='full'`, cropped/strided afterward to
  match the configured padding and stride).
- For each `(label, output channel k)`, it compares `min`, `max`, `mean`,
  and `sum` between the Python reference and the C output, using
  absolute tolerances of `5e-2` for min/max/mean and `5.0` for sum (the
  sum tolerance is looser since it accumulates over tens of thousands of
  float32 values, where rounding error compounds).
- It prints a PASS/FAIL verdict per case, plus an overall summary.

## 6. Config File Format (`configs.txt`)

Each simulation case is defined as a block of `key=value` lines,
separated by a blank line. Supported keys:

```
P, Q            # Input spatial dimensions (height, width)
H, W            # Output spatial dimensions (optional — derived from
                #   P, Q, R, S, stride, padding if omitted)
C               # Input channels
K               # Output channels
R, S            # Kernel height, width
TH, TW, TC, TK  # Tile sizes
stride          # default: 1
padding         # default: 0
strategy        # OS | WS | IS   (default: OS)
label           # case name      (default: "Case N")
```

Example:

```
label=case1_small
P=8
Q=8
C=3
K=4
R=3
S=3
stride=1
padding=1
TH=4
TW=4
TC=3
TK=2
strategy=OS
```

Unknown keys are ignored with a warning; cases missing required fields
(any of H, W, C, K, R, S, TH, TW, TC, TK, or a non-positive stride) are
skipped with a warning and excluded from the run.

## 7. Program Outputs

Running the simulator (`./sim configs.txt results_tk.csv results_thw.csv
conv_stats.csv`) produces:

- **Terminal report**: Table A and Table B, printed once per unroll mode
  (TK and TH×TW), for every valid case.
- **`results_tk.csv`** / **`results_thw.csv`**: the same metrics in CSV
  form, one row per case, tagged with `unroll = TK` or `THW`.
- **`conv_stats.csv`**: per-output-channel statistics (`min`, `max`,
  `mean`, `sum`) of the actual convolution result, used by `verify.py`
  for correctness checking.

## 8. Known Implementation Notes

- `ps_buf_load()` and `ps_buf_save()` (in `stationary.c`) are defined but
  are **not currently invoked** anywhere in the main simulation loop
  (`conv_forward()` in `conv.c`). They appear to be scaffolding for a
  not-yet-wired feature (e.g. future support for spilling/reloading
  Psums across tile boundaries) and do not currently affect any reported
  metric.
- "SRAM Input/Weight Traffic" in Table A reports **cumulative bytes
  moved over the whole run**, not the static per-tile buffer capacity.
  See the note under Table A for details.
