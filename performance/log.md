# Performance log (branch `performance`)

## Summary
- Ḃ of the reference tendencies through Oceananigans' `KernelFunctionOperation` + `_compute!` (a CPU-only dynamic
  dispatch removed): `reference_tendency_work` 175 → 38 ms at 64³ (4.6×), 1317 → 276 ms at 128³ (4.8×), 264 → 0.14 MiB per compute.
- One Ḃ and one Ċ per z★ shared by its reference tendencies (∂ₜ𝒜 keeps its own Ḃ), the b★-budget integrals as level
  Fields without Ṗ★(z★), an `Int32` permutation: z★ and the nine reference-tendency diagnostics 39 → 6.6 GB at 512³.
  Budget output of 67 diagnostics at 64³: 8.6 s, 21.4 GB allocated → 0.33 s, 2.7 MiB. Values bit-identical to `main`.
- Tests: the diagnostics of one z★ agree bit for bit in two orders, with `compute!` and through `fetch_output`;
  `compute!` of every diagnostic allocates the same at 4³ and 8³; GPU tests at 32³ and in two orders (not yet run).
- Decisions (below): R★ keeps its own Ḃ, independent of Gⁿ; ⟨ADV_b⟩ and K_num keep separate passes; closure fields
  deferred. CPU `time_step!` consistent / Fickian = 3.4× at 64³, 2.3× at 128³: above 1.5×, so the CPU criterion is met.
- Cluster: `sbatch benchmark/gpu.slurm` on `performance` (benchmark, `gpu_registers.jl`, `TEST_ARCHITECTURE=GPU
  TEST_GROUP=all`), then on `main`: `git checkout main && julia --project=benchmark benchmark/benchmark.jl GPU`.
  - Benchmark: `reference_height` / `reference_tendency_work` near 1 means the sort dominates z★ and R★ on the GPU.
  - `gpu_registers.jl`: more registers and a lower occupancy for the consistent θ tendency kernel than the Fickian
    one means it is register-bound; that or the CPU ratio above decides for prototyping closure fields.

## Result: benchmark/benchmark.jl, CPU (14 cores, 8 threads), Julia 1.11.5, 5 trials, minimum (median)

| grid | operation | start (0661d5a) | branch | |
|---|---|---|---|---|
| 64³ | time_step! | 118 (122) ms | 119 (124) ms | noise |
| 64³ | reference_height | 16 (30) ms | 14 (14) ms | noise (2× spread on this row) |
| 64³ | reference_tendency_work | 175 (177) ms | 38 (51) ms | **4.6× faster**, 264 MiB → 0.14 MiB allocated |
| 64³ | θ_production | 12.0 (12.3) ms | 12.1 (14.1) ms | noise |
| 64³ | spurious_ape_production | 18 (19) ms | 17 (18) ms | noise |
| 128³ | time_step! | 811 (817) ms | 829 (861) ms | noise |
| 128³ | reference_height | 220 (267) ms | 180 (276) ms | noise |
| 128³ | reference_tendency_work | 1317 (1421) ms | 276 (379) ms | **4.8× faster** |
| 128³ | θ_production | 68 (76) ms | 71 (76) ms | noise |
| 128³ | spurious_ape_production | 153 (291) ms | 165 (272) ms | noise |
| 64³ | budget output, 67 diagnostics, one z★ (scratch driver) | 8.6 s, 21.4 GB allocated | 0.33 s, 2.7 MiB | **26× faster** |
| 16³ | fresh process: first time_step! | 1.6 s | 1.5 s | noise |
| 16³ | fresh process: build the 67 budget Fields (compiles the kernels) | 80.9 s | 81.4 s | noise |
| 16³ | fresh process: first compute_at! of the budget at a Float64 time | 11.4 s | 12.8 s | noise (11-12 s over runs) |

Resident memory at 512³ (Float64, N = 1.34 × 10⁸, 1.07 GB per N-vector, 1.1 GB per 3D field with halos), for z★
and the nine diagnostics with a reference tendency (R★, ∂ₜ𝒜, ∂ₜℬ and the six of the b★ budget): before, each
reference tendency held a 3D Ḃ, two N-vectors and a 3D Ṗ★(z★), 4.4 GB, 39 GB for nine, and z★ two `Int` N-vectors;
after, one shared Ḃ and Ċ, a private Ḃ for ∂ₜ𝒜, three Ṗ★(z★) (R★, ∂ₜ𝒜, ∂ₜℬ) and `Int32` indices: 6.6 GB.
**About −34 GB at 512³**; z★ alone costs 1.1 GB more (the shared Ḃ and Ċ, less the `Int32` indices). The b★ budget alone: 26 → 2.2 GB.

The speedups are CPU-side: the Ḃ kernel took the kernel function as an argument, which KernelAbstractions does not
specialise on for the CPU, so every face flux was a dynamic dispatch. GPU kernels are compiled fully specialised, so
the GPU should see the memory savings and the passes removed (one 3D kernel and halo fill per b★ diagnostic), not the
4.6×. Values are bit-identical to the start at 16³ (67 diagnostics, fresh processes, FFTW.ESTIMATE plans).

## Decisions
1. R★ keeps its independent Ḃ = ADV_b + σ_b − ∇·j_b; it is not computed from the model's tendencies Gⁿ. R★ is a term
   of a budget that is checked for closure, so it must not depend on the Gⁿ that ∂ₜ𝒜 (`ape_tendency`) reads, or the
   closure check partly checks Gⁿ against itself; `ape_tendency` already gives the Gⁿ-based tendency. The docstring
   of `reference_tendency_work` states this. (The Ḃ kernel of R★ is ~13 ms of the 18 ms tendency step at 64³ and its
   heaviest compile, 3.5 s.)
2. ⟨ADV_b⟩ and K_num keep separate passes. Sharing their integral (𝓘[-ADV_b] = -𝓘[ADV_b] exactly) saves one 3D kernel
   pass, a gather, a scan and a sum per output, but needs a public API change or hidden caching, either of which costs
   more readability than the pass costs time.
3. Closure fields for the cell-centre thermodynamic coefficients (cₚ/π, T_S/T_θ, Γ_S) are deferred, not rejected. They
   are worth prototyping only if the consistent model's `time_step!` is more than 1.5× the Fickian model's on the CPU,
   or the GPU register counts show the θ tendency kernel is register-bound.
   CPU evidence (`benchmark/benchmark.jl CPU 5`, 8 threads, first call / minimum / median of 5):

   | grid | consistent | Fickian | ratio (min / median) |
   |---|---|---|---|
   | 64³ | 833 ms / 63.6 ms / 68.5 ms | 266 ms / 18.4 ms / 22.5 ms | 3.45 / 3.05 |
   | 128³ | 1.62 s / 513 ms / 534 ms | 479 ms / 225 ms / 231 ms | 2.28 / 2.31 |

   **The CPU criterion is met** (> 1.5× at both sizes). The ratio is the whole cost of consistency (b̃ in the w
   equation, the `ThermodynamicDiffusivity` fluxes, σ_θ), of which re-evaluating the coefficients is a part: σ_θ alone
   was 12 % of the step in the baseline profile. The GPU evidence comes from `benchmark/gpu_registers.jl`. These
   absolute times are lower than the table above (64³ `time_step!` 64 vs 119 ms), from a different machine state;
   the ratio is taken within one run.

## Baseline (commit 0661d5a, CPU 14 cores, 8 threads, Julia 1.11.5, 64³, 5 trials, min of runs 1 / 2)
- time_step! 115 / 121 ms (1.1 MiB); reference_height 14 / 30 ms; reference_tendency_work 163 / 155 ms (264 MiB);
  θ_production 12.2 / 12.1 ms; spurious_ape_production 32 / 33 ms. Noise: ~5 % on all rows, 2× on reference_height.
- Budget output (67 exported diagnostics, one shared z★, `compute_at!` at a new time): 8.6 / 9.0 s, 21.4 GB allocated.
- Profile, budget: 90 % in the Ḃ kernel of the reference tendencies (`_compute_buoyancy_tendency!`), dynamic
  dispatch per face (`Ax_q*`, `δ*` frames non-inlined), ~1 KB allocated per cell per compute.
- Profile, time_step!: 36 % Oceananigans `compute_Gc!`/`tracer_tendency`, of which σ_θ (θ_productionᶜᶜᶜ) 12 % of
  the step; the rest pressure solve, halos, other Oceananigans kernels.
- First call (fresh process, 16³): first time_step! 1.6 s; budget construction 81 s, first compute_at! 11 s.
- The laptop was suspended for 5-10 s during the first budget profile; sample counts only, no timings taken from it.

## Changes
- Ḃ of reference tendencies through Oceananigans' KernelFunctionOperation + `_compute!` instead of a kernel taking the function (reference_state.jl): reference_tendency_work 163 → 49 ms, 264 → 0.14 MiB allocated per compute (CPU-only: GPU kernels have no dynamic dispatch); 512³: no change in resident memory. Kept.
- Ḃ★ gathered straight into Ċ, scanned in place, and Ċ held once in z★'s SortingWorkspace, shared by its reference tendencies (sorting.jl): reference_tendency_work 49 → 48 ms (noise); 512³: −2.1 GB per reference tendency, +1.1 GB once (full budget, 9 reference tendencies: 19.3 → 1.1 GB). Kept.
- Upper bound for the duplicate j_S in j_θ: j_θ without its j_S term (thermodynamic_diffusivity.jl), θ_production 12.2 → 11.5-12.4 ms (noise): the compiler already shares the duplicate evaluations; this idea and the duplicate T in cₚ dropped. Reverted.
- One Ḃ scratch per z★ (ReferenceState.buoyancy_tendency), shared by its reference tendencies; ape_tendency, which reads its Ḃ afterwards, keeps its own (reference_state.jl, ape_budget.jl): speed unchanged (no pass removed); 512³: −1.07 GB per reference tendency but one, +1.07 GB for z★ alone (full budget: 9 → 2 Ḃ arrays, −7.5 GB). Kept.
- The six b★ diagnostics read 𝓘[f] through `isopycnal_integral`, a Field at (Nothing, Nothing, Center) sharing the data of Ṗ★ᶜᶜᶜ, so Ṗ★(z★) is neither stored nor computed (reference_state.jl, sorting.jl, reference_buoyancy_budget.jl): six b★ diagnostics 103 → 93-104 ms (noise on the CPU; one 3D kernel pass and halo fill fewer each); 512³: −1.09 GB per b★ diagnostic, −6.5 GB for the six. Kept.
- Int32 permutation and slot indices, as the bands already are (sorting.jl): reference_height 30 → 30 ms, reference_tendency_work 52 → 53 ms (noise); 512³: −1.07 GB. Kept.
- Compile: time-independent bodies for compute!(z★) and compute!(reference tendency), as Oceananigans' compute_computed_field! (reference_state.jl): first budget output at 16³, construction 57 → 55 s, first compute_at!(f, ::Float64) 11.1 → 11.8 s (noise; z★ alone 52 → 8 ms of compile). Reverted.
- Not tried, stop condition: z★ is 77 % `sortperm!` (Base; CUDA on the GPU) and 9 % `compute_bands!` (2.6 of 30 ms,
  inside the 2× noise of that row); the rest is single-pass physics kernels and Oceananigans' time step. The 12 s
  first compute_at! at a Float64 time is Oceananigans' `compute_at!(::KernelFunctionOperation, time)`, whose
  `Tuple(generator)` compiles in proportion to the argument types (0.32 s for a small KFO, 0.69 s for a b★ one).

## Jean Zay checklist (benchmark/gpu.slurm)
- [ ] Run gpu.slurm on `performance` (benchmark, `gpu_registers.jl` and `TEST_ARCHITECTURE=GPU TEST_GROUP=all`), then the benchmark alone
      on `main` (`git checkout main && julia --project=benchmark benchmark/benchmark.jl GPU`) for the comparison.
- [ ] Report the test summary, in particular: `reference_height`, `reference_tendency_work`, `ape_tendency`,
      `bpe_tendency` and the six isopycnal testsets of `GPU diagnostics match the CPU`,
      `GPU diagnostics of one z★ computed in different orders [Float64]` and
      `GPU reference state of a 32³ grid matches the CPU [Float64]`.
- [ ] Report the register table of `gpu_registers.jl` (θ tendency, consistent and Fickian; the two R★ kernels).
- [ ] Memory: at 256³ and 512³, `CUDA.memory_status()` after building z★ and the nine reference-tendency diagnostics
      (`exported_diagnostics` of test/utils_for_runtests.jl), on `main` and on `performance`.
- Kept changes the GPU could overturn:
  - Int32 permutation (`sortperm!` into a `CuVector{Int32}`): CUDA.jl may refuse it, or sort more slowly than into
    `Int`; then revert ebfd395 (−1.07 GB at 512³ only).
  - `cumsum!(Ċ, Ċ)` in place on the device: if CUDA.jl's scan mishandles aliasing, R★ and the b★ budget are wrong,
    which the GPU tests catch; then gather into a second N-vector again.
  - The isopycnal integral Field at (Nothing, Nothing, Center) with a `ReferenceTendency` operand (halo fill of a
    reduced Field on the device).
  - The KernelFunctionOperation for Ḃ: no GPU speedup expected (a CPU-only dispatch problem), only a check that
    Oceananigans' `_compute!` gives the same values.
