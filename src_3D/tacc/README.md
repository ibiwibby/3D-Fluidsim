# Run Harish's D3Q19 solver at 64 cubed on Lonestar6

On Vista, request `idev -p gh-dev -N 1 -n 1 -t 01:00:00`, load
`module load gcc cuda`, and run `CUDA_ARCH=sm_90 bash tacc/run64.sh both`
from src_3D. The source can also be obtained from GitHub after this change
is pushed. Build on the allocated Vista node.

To upload directly from Windows to Vista using the MAC workaround,
run from the local repository root.

```powershell
powershell -ExecutionPolicy Bypass -File .\src_3D\tacc\package.ps1 -Send -Remote ibraheem@vista.tacc.utexas.edu -SSHMac hmac-sha2-512
```

The original 32-cubed modes remain available. Runtime options configure the
same CUDA executable for a 64-cubed grid with a radius-8 sphere.

The preset uses tau=1, force_x=1.25e-6, **600 steps with a snapshot every
5 steps**, giving 121 VTI files plus the PVD timeline. This captures the early
closed-box response; it is not a promise that either case reaches equilibrium.
The previous 12000-step / every-200 recording skipped the strongest early
motion (the uploaded closed run peaked at step 54).

The force remains one eighth of the original 32-cubed force. Only recording
length and sampling cadence changed; solver physics and boundaries are unchanged.
Override either setting when needed, for example on an allocated Vista GPU:

```bash
CHANNEL64_STEPS=600 CHANNEL64_OUTPUT_EVERY=5 CUDA_ARCH=sm_90 bash tacc/run64.sh closed
```

Two boundary choices are available.

- **periodic** connects the x ends and maintains the original circulating
  channel, with bounce-back y/z walls and the sphere.
- **closed** adds stationary bounce-back x walls. Momentum hitting a wall
  reflects into the opposite population. Continuous forcing eventually builds
  a pressure gradient that can balance it, with the fluid approaching rest.

Each run uses about 76 MiB for its two population buffers, plus 8 MiB for
GPU field extraction. A snapshot has about 10 MiB of raw Float64 field
data including the solid mask. Allow roughly 1.2 GiB per recording (2.4 GiB for both).

## Send from Windows

Run this in PowerShell from the repository root. It creates a source archive
under src_3D/build/transfer and sends it to the configured lonestar6 account.
TACC authentication happens in scp.

```powershell
powershell -ExecutionPolicy Bypass -File .\src_3D\tacc\package.ps1 -Send
ssh lonestar6
```

Omit `-Send` to package locally. Use `-Remote YOUR_SSH_ALIAS` to choose
another destination. The archive contains the CUDA source and run scripts.

For Windows OpenSSH reporting `Corrupted MAC on input`, the package script
accepts `-SSHMac hmac-sha2-512`. Use the matching login command
`ssh -m hmac-sha2-512 ibraheem@vista.tacc.utexas.edu`.
A noninteractive connection with this MAC reached Vista authentication on
the development laptop. Interactive TACC authentication remains necessary.
This workaround is documented in the
[Windows OpenSSH issue tracker](https://github.com/PowerShell/Win32-OpenSSH/issues/2195).

## Build and run on TACC

In the TACC shell, unpack under scratch and request a GPU allocation.
Add `-A YOUR_ALLOCATION` to idev if you need a particular project.
Lonestar6 documents the `gpu-a100-dev` interactive GPU queue.
[Lonestar6 guide](https://docs.tacc.utexas.edu/hpc/lonestar6/)

```bash
export task_dir="$SCRATCH/fluid3d64_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$task_dir"
cd "$task_dir"
tar -xzf "$HOME/fluid3d64.tar.gz"
idev -p gpu-a100-dev -N 1 -n 1 -t 01:00:00
module load cuda
cd "$task_dir/src_3D"
bash tacc/run64.sh both
```

If idev changes directory or drops the shell variable, use the full scratch
path printed by `pwd` before requesting the allocation. If the default
CUDA module is unavailable, select an installed version with
`module spider cuda`. The run script compiles for A100 (`sm_80`),
runs the GPU regression suite, then runs both 64-cubed cases in sequence.
It exits on failed tests or failed simulations. The printed output paths
and sibling logs (including stderr) identify each run by job ID, boundary mode and timestamp.
Use `periodic` or `closed` instead of `both` to run one case.

The new closed-end GPU tests require execution on the target GPU.
Local checks cover wall geometry, dimensions and binary output. This
machine has no NVIDIA CUDA compiler, so no completed 64-cubed CUDA run
is claimed here.

## Direct commands

After building inside the allocation, these are equivalent presets. Output
paths are relative to src_3D when using make -C.

```bash
make -C src_3D channel64 CUDA_ARCH=sm_80
make -C src_3D channel64 CUDA_ARCH=sm_80 CHANNEL64_BOUNDARIES=--closed-ends CHANNEL64_OUTPUT=../results/3D_closed64
```

Or pass the options directly.

```bash
./src_3D/build/sm_80/debug-0/fluid3d --channel --size 64 --radius 8 --force 1.25e-6 --closed-ends --steps 600 --output-every 5 --output results/closed64
```

## Bring the results back

In local PowerShell, replace REMOTE_OUTPUT_DIRECTORY with a complete
directory printed by the run script.

```powershell
scp -r lonestar6:REMOTE_OUTPUT_DIRECTORY results/
python -m pip install -r renderer/requirements.txt
python renderer/src/import_vtk.py results/LOCAL_RUN_FOLDER/flow.pvd renderer/output/vtk_64.html
Start-Process renderer/output/vtk_64.html
```

Check diagnostics.csv for fluid mass error, density, maximum Mach number and
mean x velocity. The browser consumes density, velocity and the solid mask
for both boundary choices. Density changes and transient flow show the
closed-box response. The continuing periodic flow shows recirculation.
