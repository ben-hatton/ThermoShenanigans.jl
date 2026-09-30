# Minimal headless GPU run: 3D low-resolution two-layer cabbeling instability,
# after experiment IV of Bisits, Zika & Sohail (2025, JFM 1011 A17).
# No plotting; output is visualised offline with `examples/plot_gpu_output.jl`.
#
# Run with `julia --project=examples/gpu examples/gpu/two_layer_cabbeling_gpu.jl`.
# Output files are written to the current directory.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using CUDA

arch = GPU()
Nx, Nz = 64, 512 # Ny = Nx
stop_time = 10minutes
prefix = "two_layer_cabbeling"

@info "CUDA functional: $(CUDA.functional())" # false means no GPU was found
if CUDA.functional()
    @info "Device: $(CUDA.name(CUDA.device()))"
    CUDA.versioninfo()
end

#####
##### Thermodynamics, linearised about the deep layer (θ absolute, S in kg/kg)
#####

θ_deep, S_deep       = 273.15 + 0.5, 0.0347     # Θ = 0.5 °C,  S = 34.7 g/kg
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458    # Θ = -1.5 °C, S = 34.58 g/kg

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027 # b(shallow) - b(deep), from Δρ = -0.0046 kg m⁻³
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ # so that b = g [α θ′ + ½ γ θ′² - β S′] gives Δb

thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                           reference_temperature = θ_deep, reference_salinity = S_deep,
                                           reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

#####
##### Model
#####

grid = RectilinearGrid(arch; size = (Nx, Nx, Nz), x = (-0.035, 0.035), y = (-0.035, 0.035), z = (-1, 0),
                       topology = (Periodic, Periodic, Bounded))

closure = (ScalarDiffusivity(ν = 1e-6), ThermodynamicDiffusivity(thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7))
model = NonhydrostaticModel(grid; tracers = (:θ, :S), buoyancy = ThermodynamicBuoyancy(grid, thermo), closure)

θᵢ(x, y, z) = z > -0.5 ? θ_shallow : θ_deep
Sᵢ(x, y, z) = (z > -0.5 ? S_shallow : S_deep) + 2e-7 * randn() * exp(-((z + 0.5) / 0.02)^2)
set!(model, θ = θᵢ, S = Sᵢ)

u, v, w = model.velocities
θ, S = model.tracers
T = Field(KernelFunctionOperation{Center, Center, Center}(ThermoShenanigans.height_dependent_functionᶜᶜᶜ, grid,
                                                          in_situ_temperature, model.buoyancy.formulation, model.tracers))

#####
##### Simulation
#####

# Explicit viscous stability, ν Δt (1/Δx² + 1/Δy² + 1/Δz²) ≤ ½; the wizard only enforces the advective CFL
Δx, Δz = 0.07 / Nx, 1 / Nz
max_Δt = 0.5 / (1e-6 * (2 / Δx^2 + 1 / Δz^2))

simulation = Simulation(model; Δt = max_Δt / 2, stop_time)
simulation.callbacks[:wizard] = Callback(TimeStepWizard(cfl = 0.5, max_Δt = max_Δt), IterationInterval(10))

∫θ₀, ∫S₀ = sum(interior(θ)), sum(interior(S))

progress(sim) = @info string("iteration ", iteration(sim), ", t = ", prettytime(sim), ", Δt = ", prettytime(sim.Δt),
                             ", wall time = ", prettytime(sim.run_wall_time), ", max|w| = ", maximum(abs, w),
                             ", ∫θ drift = ", sum(interior(θ)) / ∫θ₀ - 1, ", ∫S drift = ", sum(interior(S)) / ∫S₀ - 1)
add_callback!(simulation, progress, IterationInterval(100))

simulation.output_writers[:slices] = JLD2Writer(model, (; S, w, T); filename = prefix * "_slices",
                                                indices = (:, Nx ÷ 2, :), schedule = TimeInterval(30), overwrite_files = true)

simulation.output_writers[:profiles] = JLD2Writer(model, (; θ = Average(θ, dims = (1, 2)), S = Average(S, dims = (1, 2)));
                                                  filename = prefix * "_profiles", schedule = TimeInterval(30), overwrite_files = true)

simulation.output_writers[:diagnostics] = JLD2Writer(model, (; ∫θ = Integral(θ), ∫S = Integral(S),
                                                               ∫KE = Integral((u^2 + v^2 + w^2) / 2));
                                                     filename = prefix * "_diagnostics", schedule = TimeInterval(10),
                                                     array_type = Array{Float64}, overwrite_files = true)

run!(simulation)
