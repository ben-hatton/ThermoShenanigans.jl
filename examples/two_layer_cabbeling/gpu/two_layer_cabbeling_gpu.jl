# # Two-layer cabbeling instability in 3D on a GPU
#
# The three-dimensional version of `../two_layer_cabbeling.jl`, after experiment IV of
# Bisits, Zika & Sohail (2025, JFM 1011 A17). It runs headless, for example on a cluster with
# `run_gpu.slurm`, and its output is plotted offline with `plot_gpu_output.jl`.
#
# This example demonstrates:
#
#   * How to run a thermodynamically consistent model on a GPU.
#   * How to adapt the time step to both the advective and the diffusive CFL with `conjure_time_step_wizard!`.
#   * How to save slices, horizontal averages with `Average`, and volume integrals with `Integral`.
#
# Run with `julia --project=examples/two_layer_cabbeling/gpu examples/two_layer_cabbeling/gpu/two_layer_cabbeling_gpu.jl`.
# Output files are written to the current directory.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using CUDA
using Printf

@info "Running on " * CUDA.name(CUDA.device())

# ## Thermodynamics
#
# As in the 2D example, b = g [α θ′ + ½ γ θ′² - β S′] is linearised about the deep layer,
# with α chosen so that b(shallow) - b(deep) = Δb.

θ_deep, S_deep       = 273.15 + 0.5, 0.0347  # Θ = 0.5 °C,  S = 34.7 g/kg
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458 # Θ = -1.5 °C, S = 34.58 g/kg

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027 # b(shallow) - b(deep), from Δρ = -0.0046 kg m⁻³
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ

thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                           reference_temperature = θ_deep, reference_salinity = S_deep,
                                           reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

# ## Model

Nx, Nz = 64, 512

grid = RectilinearGrid(GPU(); size = (Nx, Nx, Nz), x = (-0.035, 0.035), y = (-0.035, 0.035), z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

components = thermodynamic_model_components(grid, thermo; viscosity = 1e-6, thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7)
model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

θᵢ(x, y, z) = z > -0.5 ? θ_shallow : θ_deep
Sᵢ(x, y, z) = (z > -0.5 ? S_shallow : S_deep) + 2e-7 * randn() * exp(-((z + 0.5) / 0.02)^2)
set!(model, θ = θᵢ, S = Sᵢ)

# ## Simulation
#
# At this resolution the explicit viscous limit, rather than the advective CFL, sets the time step.

simulation = Simulation(model; Δt = 0.1, stop_time = 10minutes)
conjure_time_step_wizard!(simulation, IterationInterval(10); cfl = 0.5, diffusive_cfl = 0.5)

u, v, w = model.velocities
θ, S = model.tracers

progress(sim) = @printf("iteration: %d, time: %s, Δt: %s, wall time: %s, max|w|: %.2e m s⁻¹\n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(sim.run_wall_time),
                        maximum(abs, w))

add_callback!(simulation, progress, IterationInterval(100))

# ## Output
#
# We save a vertical slice, horizontally averaged profiles, and volume integrals.

diagnostics = thermodynamic_diagnostics(model)
prefix = "two_layer_cabbeling"

simulation.output_writers[:slices] = JLD2Writer(model, (; S, w, T = diagnostics.T, σ_η = diagnostics.σ_η);
                                                filename = prefix * "_slices", indices = (:, Nx ÷ 2, :),
                                                schedule = TimeInterval(30), overwrite_files = true)

simulation.output_writers[:profiles] = JLD2Writer(model, (; θ = Average(θ, dims = (1, 2)), S = Average(S, dims = (1, 2)));
                                                  filename = prefix * "_profiles",
                                                  schedule = TimeInterval(30), overwrite_files = true)

simulation.output_writers[:integrals] = JLD2Writer(model, (; ∫θ = Integral(θ), ∫S = Integral(S),
                                                             ∫KE = Integral((u^2 + v^2 + w^2) / 2),
                                                             ∫ε = Integral(diagnostics.ε), ∫σ_η = Integral(diagnostics.σ_η));
                                                   filename = prefix * "_integrals",
                                                   schedule = TimeInterval(10), array_type = Array{Float64},
                                                   overwrite_files = true)

run!(simulation)
