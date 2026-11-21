# Handoff notes

- Extracted the working VTI/PVD viewer from the local free_surface_tank experiments. The renderer folder has no dependency on that folder; Three.js 0.160.1 and its MIT license are included.
- Changed the 64-cubed launcher to 600 steps, saving every 5. This targets the early transient; it does not guarantee equilibrium. Old recordings remain unchanged and must not be mistaken for the new sampling cadence.
- Verified standalone conversion of the existing closed 64-cubed series (26 frames, steps 0?5000), then checked Chrome playback, scrubbing, volume/slice controls, solid visibility, arrows, and tracer movement/pause. The generated HTML is about 76 MB and is intentionally not committed.
- The new 600-step CUDA run still needs execution on TACC. Browser rendering and interpolated display tracers are separate from the GPU solver and are not solver benchmarks.
