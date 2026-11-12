# Project Progress Presentation — Race Conditions

**Prepared November 11, 2026.** These notes cover the eight requested presentation topics. Use the “Slide content” bullets on slides and the accompanying detail as speaking notes.

**Project objective:** Build a CPU/GPU lattice Boltzmann simulator for three-dimensional channel flow around a solid obstacle. The proposal targets stable flow, mass conservation within 0.5%, and CPU/GPU agreement within 1%.

**Current checkpoint:** A working CUDA D2Q9 solver with periodic boundaries, independent update checks, a timestep loop, CSV output, and local visualization. The 2 × 2 case has documented successful runs on Vista. Current source defaults are 16 × 16 with density 1.10 at `(8,8)`; those newer settings must not be confused with the earlier 2 × 2 validation results.

**Planning basis:** The schedule and assignments below are proposed next steps, not completed work or confirmed team commitments. They use the original proposal's ownership and December deadlines. Confirm teammate work outside this directory before presenting the overall team status.

## 1. Our plan from now until the end of the project

### Slide content

- Finish stronger 2D validation and establish the CPU reference.
- Extend the same collision/streaming framework to a small 3D grid.
- Add channel boundaries and one simple obstacle before increasing problem size.
- Measure correctness and performance, then freeze the demonstration and report.

### Proposed schedule and completion gates

| Dates | Work | Evidence needed before moving on | Proposed owners from the proposal |
| --- | --- | --- | --- |
| Nov 11–14 | Validate 3 × 3 directional streaming and the 16 × 16 disturbance; add a serial CPU reference and a uniform-flow test | Tests that distinguish positive/negative directions; matching CPU/GPU fields on small cases; saved logs | Harish: reference/correctness; Esteban: GPU |
| Nov 15–19 | Implement a 3D lattice, tentatively D3Q19, including 3D indexing and periodic streaming; start at 8³–32³ | Rest and uniform-flow preservation, correct streaming, conservation, CPU/GPU comparison | Harish and Esteban; Ibraheem: integration |
| Nov 20 | Integration and scope decision | Decide whether the 3D foundation is reliable enough to finish boundaries and validation on schedule | All |
| Nov 20–24 | Add a simple channel, wall treatment, one obstacle, and a known reference flow; add binary field output and slices | Stable channel/obstacle case, defined boundary mass accounting, understandable visualization | Ibraheem: geometry/configuration; Harish: correctness; Esteban: GPU; Akshat: output |
| Nov 25–28 | Run 64³ and 128³; attempt 256³ if correctness, memory, and time permit; profile one bottleneck and evaluate one optimization | CPU/GPU error metrics; kernel/transfer/output timing; repeatable performance data | Esteban and Akshat, with reference checks from Harish |
| Nov 29–30 | Freeze the demo, reproduce runs, prepare graphs and presentation | Saved configuration, executable/build instructions, logs, plots, and recorded demo | All |
| Dec 1 or 3 | Final presentation, on the assigned course date | Demonstration and results with clear limitations | All |
| Dec 4–6 | Complete final analysis and report | Reproducible results, attribution, limitations, and final report by Dec 6 | All |

### Speaker detail

The first 3D deliverable should be a small verified grid, not an immediate 256³ run. The CPU reference and GPU implementation need the same direction convention, initial conditions, boundary rules, and output definitions. Define the CPU/GPU error metric before claiming agreement, including how near-zero velocities are handled.

The original Nov 20 target was validated 128³/256³ runs plus profiling and optimization. This revised plan is later than that target. If the 3D integration gate fails, propose activating the original reduced scope: **2D channel flow around an obstacle, CPU/GPU comparison, stability checks, and visualization**. The current periodic density-bump demonstration alone does not satisfy that fallback.

## 2. Risks: what happened, what did not, and how mitigation worked

### Slide content

- Small-grid symmetry made the first visualization less informative than expected.
- Numerical and memory checks passed for the tested checkpoint cases.
- Remote visualization was handled by saving data and rendering locally.
- Large-grid performance and full fluid correctness remain untested risks.

### Risk assessment

| Risk | Observed status | Mitigation and outcome |
| --- | --- | --- |
| Plausible-looking but incorrect fluid motion | Still an open risk for general flow. No instability or mass failure was observed in the documented 2 × 2 tests | Separate equilibrium, collision, and streaming checks passed; 1,000 rest timesteps passed. This verifies basic updates, not channel-flow accuracy |
| GPU memory-access errors | No errors reported in the four tested sanitizer modes | Explicit bounds checks, separate streaming buffer, launch checks, and synchronization helped make failures observable; Compute Sanitizer reported zero errors |
| No visible velocity in the 2 × 2 demonstration | Actually encountered | Explained by periodic directional aliasing: east/west reach the same cell, so opposing populations cancel. Larger-grid configuration was added; a 3 × 3 directional test is still needed |
| No X-forwarding for remote graphics | Actually encountered | CSV snapshots were copied locally and rendered into GIF/PNG without a remote GUI; the 2 × 2 animation was reviewed successfully |
| GPU environment/architecture mismatch | Initial Vista build used the default `sm_80`; corrected before the documented tests | Rebuilt with `sm_90`, confirmed compute capability 9.0, and ran successfully on GH200. No failed GPU run from the initial architecture selection was reported |
| Memory bandwidth, transfers, or launch overhead limit speed | Anticipated; no performance bottleneck has yet been measured | Persistent device buffers and direction-major storage are in place. Profiling, reduced output frequency, and selective synchronization are future work—not demonstrated optimizations |
| Teammate unavailable | No occurrence documented in the available evidence | Keep the proposal's regular check-ins and backup ownership; do not claim reassignment was needed unless the team confirms it |
| Schedule slips relative to proposal | Evident in the available implementation | Focus on a verified end-to-end case, set a Nov 20 decision gate, and preserve the substantive 2D fallback |

### Speaker detail

Separate “not observed” from “eliminated.” Passing a tiny periodic test does not remove instability, incorrect boundaries, or poor performance as risks at larger sizes. The mitigation that demonstrably worked was incremental validation: hand-checkable cases exposed what the simulation should do before relying on an animation.

## 3. Where we stand on finishing on time

### Slide content

- The GPU checkpoint and local visualization are working.
- We are behind the original 3D/CPU-comparison milestones.
- Full completion is at risk; it depends on a working small 3D solver and reference by the integration gate.
- A verified 2D channel/obstacle fallback remains possible, but still requires additional work.

### Schedule assessment

The original Oct 30 milestone called for multithreaded CPU and GPU versions agreeing on a 64³ case. This directory contains a 2D CUDA solver; it does not contain a CPU reference, OpenMP implementation, 3D lattice, channel boundaries, or obstacle treatment. There is no evidence here that the Oct 30 milestone has been achieved.

Reusable work reduces the remaining effort: device-buffer ownership, error handling, launch interfaces, the timestep loop, diagnostics, configuration structure, and output separation already exist. The main uncertainty is the remaining numerical work and integration, not whether CUDA can run on the available hardware.

**Suggested statement:** “We have completed and demonstrated the GPU foundation, but we are behind our original schedule. We are prioritizing a verified channel/obstacle case and CPU comparison, with a scope decision on November 20. We cannot yet promise validated 256³ results.”

## 4. Hurdles we are stuck on or worried about

### Slide content

- No current access or build blocker: Vista compilation and GPU execution worked.
- Boundary conditions and physical validation are the largest unresolved technical tasks.
- CPU/GPU agreement and 3D directional correctness still need implementation.
- Large 3D snapshots require a more efficient output workflow than CSV.

### Specific concerns and next actions

1. **Boundary correctness:** Implement walls and inlet/outlet behavior without artificial mass drift. Validate a simple channel first, then introduce one obstacle. For an open channel, evaluate mass balance with inflow/outflow; constant total domain mass is not the only relevant criterion.
2. **Directional correctness:** The 2 × 2 periodic test cannot distinguish sign mistakes in streaming. Add a 3 × 3 case before extending the routing to 3D.
3. **Physical validation:** A rest state and a spreading density bump are not a benchmark for realistic flow. Select an analytical or published reference case and specify comparison metrics.
4. **Performance at scale:** The checkpoint synchronizes after each operation and downloads every disturbance frame. Keep this debug behavior separate from timed runs so output and synchronization do not dominate the measurement.
5. **3D output and visualization:** Plan binary structured-grid snapshots with density and velocity, initially viewed as slices. Streamlines are a later visualization improvement; a water–air surface is outside the current single-phase model.

For planning, two D3Q19 population buffers in double precision require approximately **608 MiB at 128³** and **4.75 GiB at 256³**, before masks, output arrays, and runtime overhead. These are calculated estimates, not measured allocations or achieved runs. They support using TACC for the larger case rather than assuming a laptop GPU will fit it.

## 5. What we need to finish successfully

### Slide content

- Continued TACC GPU access and enough scheduled time for validation and profiling.
- Agreement on the final acceptance metrics and fallback scope.
- Team ownership of CPU reference, GPU solver, boundaries/integration, and visualization/benchmarking.
- Feedback on the revised schedule before investing in the largest runs.

### Requests to instructors or staff

- Confirm that Vista GH200 results satisfy the hardware requirement; the proposal originally requested Lonestar6/A100 access, but Vista access has now been demonstrated.
- Confirm how the proposed 1% CPU/GPU agreement and 0.5% mass criterion should be measured, especially for inlet/outlet cases.
- Clarify whether 256³ is required or whether a validated smaller 3D domain with a clear scaling study is sufficient.
- Confirm the assigned final presentation date, Dec 1 or Dec 3, and whether the Nov 20 scope decision is acceptable.
- If queue access or allocation hours become limiting, help arrange continued GPU availability. No current allocation failure has been reported.

### Requests within the team

Confirm the proposed assignments in section 1, identify any existing work outside this repository, and agree on common field layouts and boundary definitions. Preserve run configurations and logs with every performance graph. Confirm remaining GPU allocation and storage capacity rather than treating the proposal's requested resources as already available.

## 6. Data and evidence of work completed

### Slide content

- CUDA compiled and executed on **NVIDIA GH200 120GB**, compute capability **9.0**.
- Initialization, collision, streaming, and 1,000-step rest-state checks passed.
- Four Compute Sanitizer runs reported **zero errors**.
- The disturbance run saved **101 snapshots**, conserved mass, and produced a reviewed animation.

### Numerical evidence from the documented 2 × 2 run

| Test | Setup | Observed result |
| --- | --- | --- |
| Initialization | Four cells, density 1, zero velocity | PASS; mass 4; relative mass error about `2.22e-16` |
| Collision | Cell 0: east/west each `+0.01`, rest `−0.02`; `tau=1` | PASS; all nine populations returned to equilibrium |
| Streaming | Unique labels in all 36 populations | PASS against the one-step table; two steps restored the original array |
| Repeated rest state | Steps 0, 1, 2, 10, and 1,000 checked | All passed; maximum reported population error about `1.11e-16`, density error about `2.22e-16`, velocity zero |
| Density disturbance | Initial density 1.01 at one cell, 1 elsewhere; 100 steps | 101 snapshots saved; mass 4.01; printed relative mass errors at most about `2.21e-16` |
| GPU memory checking | Initialization, collision, streaming, and rest simulation | Each reported `ERROR SUMMARY: 0 errors` |

The numerical acceptance tolerance was `1e-12`. The sanitizer results do not include a separate disturbance-mode run, and memcheck is not proof of correct fluid physics.

### Artifacts to show

- [Density redistribution animation](results/disturbance.gif).
- [Initial state](results/disturbance_initial.png) and [final state](results/disturbance_final.png).
- [CSV snapshots](results/disturbance.csv): the original 2 × 2 run contains 404 data rows; final densities are approximately 1.0025.
- [16 × 16 animation artifact](results/16x16/disturbance.gif): available in the directory. Confirm the generating configuration and retain its run log before attaching quantitative claims to it.
- [Implementation summary](work_done.md), [GPU kernels](src/kernels.cu), [simulation controller](src/simulation.cu), and [visualization script](scripts/animate.py).

**Suggested demonstration:** Show initial and final density values, play the GIF, and explain that total mass is unchanged. Then show a short extract of the PASS/error summaries. State that the 2 × 2 zero-velocity behavior is a symmetry limitation, not evidence of general fluid-flow accuracy.

No performance measurements, speedup claims, CPU/GPU agreement results, or validated 128³/256³ runs are available in the evidence reviewed for this outline. The Vista terminal results were supplied by Harish in the project discussion; save their raw logs in the repository before the final presentation rather than treating this summary as a replacement for logs.

## 7. Prior art and sources of ideas or implementation support

### Slide content

- The numerical method is established lattice Boltzmann BGK collision plus streaming.
- CUDA provides the execution and memory-management infrastructure.
- Matplotlib and Pillow provide the visualization backend.
- The checkpoint code was developed with Codex assistance and validated through user-run GPU tests.

### Attribution details

| Source | What it informed or provided | Relationship to our code |
| --- | --- | --- |
| [M-Star theory and implementation documentation](https://docs.mstarcfd.com/17_Theory/theory-and-implementation.html) | Equilibrium distributions, relaxation, and collision/streaming concepts | Referenced in the checkpoint plan as theory support. Its displayed lattice examples include D3Q19; our current code uses D2Q9 weights and directions |
| [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | CUDA kernel/thread execution and runtime concepts | Platform documentation; CUDA kernels, allocation, copies, and synchronization form the execution framework |
| [NVIDIA Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | GPU memory checking and interpretation of memcheck | Tool used for the reported zero-error runs; it complements numerical tests |
| [TACC Vista guide](https://docs.tacc.utexas.edu/hpc/vista/) and [Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/) | GPU environment and execution guidance | Supported the remote CUDA workflow; the demonstrated runs used Vista |
| [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html) and [quiver documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html) | GIF encoding and velocity-arrow rendering | Used by the local animation script together with NumPy and Pillow |
| Codex assistance in this project | Skeleton, kernels, checks, output, plotting script, and documentation | AI-assisted implementation, followed by Harish's Vista execution and visual review; disclose according to course policy |

The available project record does not identify an imported third-party LBM solver or copied solver repository. Do not claim that BGK, D2Q9, pull streaming, or double buffering are novel inventions. Add any other references or borrowed code used independently by teammates before presenting this section.

**Planned, not implemented:** [VTK ImageData output](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html) and [ParaView filters](https://docs.paraview.org/en/latest/UsersGuide/filteringData.html) were discussed as a future 3D visualization route. They are not current runtime dependencies or completed integration work.

## 8. Questions for instructors

### Slide content / discussion prompts

1. Should we prioritize a thoroughly validated 3D channel/obstacle case at a smaller grid, or is completing the 256³ benchmark essential?
2. Which reference case and error metric would you accept for demonstrating physical correctness and the proposed 1% CPU/GPU agreement?
3. For an open channel, should the 0.5% criterion measure mass-balance residual after accounting for inflow/outflow, or another quantity?
4. Does a CPU reference plus GPU solver satisfy the hybrid requirement, or must CPU and GPU perform simulation work concurrently? How much OpenMP work is required?
5. If our Nov 20 gate fails, does the proposal's 2D channel/obstacle fallback remain acceptable, and what evidence would make it a complete submission?
6. Are Vista GH200 performance results acceptable in place of A100 results, and what AI-assistance disclosure should accompany the final code/report?

### Before presenting

Confirm the final presentation date, current teammate progress, proposed task ownership, and remaining compute allocation. Preserve the distinction between the verified 2 × 2 checkpoint, current 16 × 16 settings, and planned 3D work. This allows the audience to assess both the demonstrated progress and the remaining schedule risk accurately.
