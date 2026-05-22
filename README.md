# Standard Convolution Architecture & Dataflow Simulator

## 1. Core Objective
This project is a C-based hardware simulator designed to evaluate the performance of a Standard Convolution computing architecture. 

## 2. Hardware Architecture Model
The simulated system models a 3-tier memory and compute hierarchy:

*   **DRAM (Off-chip Memory):** Stores the complete original data (Input Feature Maps, Weights) and the final Output. It has high latency and high energy consumption.
*   **SRAM (On-chip Buffer):** Fast access but limited capacity. It is logically partitioned into three independent regions:
    *   **Input Buffer:** Stores the current Input Tile.
    *   **Weight Buffer:** Stores the current Weight Tile.
    *   **Psum/Output Buffer (PS Buffer):** Temporarily holds intermediate Partial Sums (Psums) and complete Output values before flushing them to DRAM.
*   **PE (Processing Elements) Array:** Modeled as an ideal parallel computation "black box."
    *   **Unbounded Quantity:** The number of active PEs ($N$) is dynamically allocated based on the unrolling configuration.
    *   **Structure:** Each PE contains only a tiny Accumulator register to hold the immediate Psum for the current cycle; it has no long-term storage capability.
    *   **Clock Cycle Rule:** Whenever a set of $N$ data-independent Multiply-Accumulate (MAC) operations are mapped to $N$ PEs, they are executed SIMULTANEOUSLY. This single parallel execution event is counted as **1 Cycle**.

## 3. Simulation Workflow & Tasks
The C program utilizes nested loops to simulate three primary phases of data movement and processing:

### Phase 1: Tiling (DRAM <-> SRAM)
*   **Task:** Partition the original data into specific Tile dimensions ($T_C, T_K, T_H, T_W$).
*   **Objective:** Identify the optimal tile sizes to minimize DRAM access frequency without exceeding SRAM capacity.

### Phase 2: Spatial Unrolling & Data Fetching (SRAM -> PE Array)
*   **Task:** Determine the number of active PEs ($N$) and the spatial "shape" of the fetched data for each compute cycle (e.g., fetching a planar region vs. fetching deeply across channels).
*   **Objective:** Maximize **Spatial Reuse**. Optimize the fetching shape to allow multiple PEs to share or broadcast Inputs/Weights in a single read cycle, thereby reducing SRAM read accesses.

### Phase 3: Stationary Strategies (PE Computation over Cycles)
*   **Task:** Implement and simulate three dataflow strategies: Weight Stationary (WS), Input Stationary (IS), and Output Stationary (OS).
*   **Objective:** Maximize **Temporal Reuse**. Evaluate how each strategy affects the system's reliance on the SRAM PS Buffer (i.e., how often intermediate Psums must be evicted from the PE and read back later).

## 4. Output Profiling Metrics
Upon completion, the simulator outputs the following metrics to evaluate trade-offs:

### Table A: Memory Hierarchy Profiling (Varying Tile Sizes)
| Metric | Description |
| :--- | :--- |
| **DRAM Loads** | Total number of times original Inputs and Weights are read from DRAM. |
| **DRAM Stores** | Total number of times completed Outputs are written back to DRAM. |
| **SRAM Input Size** | Memory capacity required to store the current Input Tile. |
| **SRAM Weight Size** | Memory capacity required to store the current Weight Tile. |
| **SRAM Psum/Output Size** | Maximum capacity needed to maintain active Psums and completed Outputs before flushing to DRAM. |

### Table B: Compute Architecture Profiling (Varying Unrolling & Stationary Strategies)
| Metric | Description |
| :--- | :--- |
| **SRAM Input Reads** | Total Input fetches from SRAM by the PE array. Used to evaluate spatial reuse of inputs. |
| **SRAM Weight Reads** | Total Weight fetches from SRAM by the PE array. Used to evaluate weight broadcasting efficiency. |
| **PS Buffer Reads/Writes** | The number of times intermediate Psums are evicted from PEs to the SRAM PS Buffer and read back for accumulation. |
| **Max PE Utilization ($N$)** | The peak number of hardware PEs mobilized in a single cycle (Represents the area/hardware cost of the system). |
| **Total Compute Cycles** | Total number of parallel execution cycles triggered. (Represents the actual execution time / latency of the system).