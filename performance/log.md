# Performance log (branch `performance`)

## Baseline (commit 0661d5a, CPU 14 cores, 8 threads, Julia 1.11.5, 64³, 5 trials, min of runs 1 / 2)
- time_step! 115 / 121 ms (1.1 MiB); reference_height 14 / 30 ms; reference_tendency_work 163 / 155 ms (264 MiB);
  θ_production 12.2 / 12.1 ms; spurious_ape_production 32 / 33 ms. Noise: ~5 % on all rows, 2× on reference_height.
- Budget output (67 exported diagnostics, one shared z★, `compute_at!` at a new time): 8.6 / 9.0 s, 21.4 GB allocated.
- Profile, budget: 90 % in the Ḃ kernel of the reference tendencies (`_compute_buoyancy_tendency!`), dynamic
  dispatch per face (`Ax_q*`, `δ*` frames non-inlined), ~1 KB allocated per cell per compute.
- Profile, time_step!: 36 % Oceananigans `compute_Gc!`/`tracer_tendency`, of which σ_θ (θ_productionᶜᶜᶜ) 12 % of
  the step; the rest pressure solve, halos, other Oceananigans kernels.
- First call (fresh process, 16³): see table at the top once measured.
- The laptop was suspended for 5-10 s during the first budget profile; sample counts only, no timings taken from it.

## Changes
- Ḃ of reference tendencies through Oceananigans' KernelFunctionOperation + `_compute!` instead of a kernel taking the function (reference_state.jl): reference_tendency_work 163 → 49 ms, 264 → 0.14 MiB allocated per compute (CPU-only: GPU kernels have no dynamic dispatch); 512³: no change in resident memory. Kept.
- Ḃ★ gathered straight into Ċ, scanned in place, and Ċ held once in z★'s SortingWorkspace, shared by its reference tendencies (sorting.jl): reference_tendency_work 49 → 48 ms (noise); 512³: −2.1 GB per reference tendency, +1.1 GB once (full budget, 9 reference tendencies: 19.3 → 1.1 GB). Kept.
