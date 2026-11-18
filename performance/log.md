# Performance log (branch `performance`)

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

## For Ben
- R★ from Gⁿ: the Ḃ kernel of R★ is now its largest run-time part (~13 ms of the 18 ms tendency step at 64³) and its
  largest compile (3.5 s, the heaviest kernel of the budget); reading Gⁿ would remove both.
- Closure fields for cₚ/π, T_S/T_θ, Γ_S: σ_θ is 12 % of time_step!, and ~10 budget kernels at 11-14 ms each
  re-evaluate these at both cells of every face. Within a kernel the compiler already shares repeated evaluations
  (j_S upper-bound test below); storage would only save work across kernels, at 1.1 GB per coefficient at 512³.
- Public API: ⟨ADV_b⟩ and K_num each integrate ±ADV_b in their own reference tendency; 𝓘[-ADV_b] = -𝓘[ADV_b] exactly,
  so sharing one would save a 3D kernel pass, a gather, a scan and a sum per output, but needs a shared argument.

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
- [ ] Run gpu.slurm on `performance` (benchmark + `TEST_ARCHITECTURE=GPU TEST_GROUP=all`), then the benchmark alone
      on `main` (`git checkout main && julia --project=benchmark benchmark/benchmark.jl GPU`) for the comparison.
- [ ] Report the test summary, in particular: `reference_height`, `reference_tendency_work`, `ape_tendency`,
      `bpe_tendency` and the six isopycnal testsets of `GPU diagnostics match the CPU`, and the new
      `GPU reference tendencies of one z★ computed together [Float64]`.
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
