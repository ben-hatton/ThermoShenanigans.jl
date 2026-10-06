# # Two-layer cabbeling instability
#
# A two-dimensional, low-resolution version of experiment IV of Bisits, Zika & Sohail (2025, JFM 1011 A17):
# cold, fresh water over warm, salty water. The layers are statically stable, but their mixtures are
# denser than either, so diffusion across the interface drives a cabbeling instability.
#
# This example demonstrates:
#
#   * How to set up `CabbelingBoussinesqThermodynamics` from observed layer properties.
#   * How to output thermodynamic diagnostics, such as the temperature, with `JLD2Writer`.
#   * How to monitor tracer conservation with `Integral` and save horizontal means with `Average`.
#
# Run with `julia --project=examples examples/two_layer_cabbeling/two_layer_cabbeling.jl`.
# Output and figures are saved in `examples/two_layer_cabbeling/output`.
# For a 3D version on a GPU, see `gpu/`.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using Printf
using CairoMakie

# ## Thermodynamics
#
# θ is the absolute potential temperature and S is in kg/kg. The equation of state
# b = g [α θ′ + ½ γ θ′² - β S′] is linearised about the deep layer.

θ_deep, S_deep       = 273.15 + 0.5, 0.0347  # Θ = 0.5 °C,  S = 34.7 g/kg
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458 # Θ = -1.5 °C, S = 34.58 g/kg

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027 # b(shallow) - b(deep), from Δρ = -0.0046 kg m⁻³

# We choose α so that b(shallow) - b(deep) = Δb,

Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ

thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                           reference_temperature = θ_deep, reference_salinity = S_deep,
                                           reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

# The shallow layer is lighter than the deep layer, but an even mixture of the two is denser:

b_shallow = buoyancy(θ_shallow, S_shallow, thermo)
b_mixture = buoyancy((θ_shallow + θ_deep) / 2, (S_shallow + S_deep) / 2, thermo)
@printf("b(shallow) - b(deep) = %.2e m s⁻², b(mixture) - b(deep) = %.2e m s⁻²\n", b_shallow, b_mixture)

# ## Model
#
# `thermodynamic_model_components` returns the buoyancy, the consistent closure
# `(ScalarDiffusivity(ν), ThermodynamicDiffusivity(κ_T, κ_S))` and the θ forcing σ_θ.

grid = RectilinearGrid(size = (32, 256), x = (-0.035, 0.035), z = (-1, 0), topology = (Periodic, Flat, Bounded))

components = thermodynamic_model_components(grid, thermo; viscosity = 1e-6, thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7)
model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

# The interface at z = -0.5 m is seeded with small-amplitude salinity noise.

θᵢ(x, z) = z > -0.5 ? θ_shallow : θ_deep
Sᵢ(x, z) = (z > -0.5 ? S_shallow : S_deep) + 2e-7 * randn() * exp(-((z + 0.5) / 0.02)^2)
set!(model, θ = θᵢ, S = Sᵢ)

# ## Simulation

simulation = Simulation(model; Δt = 0.5, stop_time = 10minutes)

# The progress message reports the drift of ∫S dV, which is conserved to round-off, and of ∫θ dV,
# which changes only through the small source σ_θ = (ε - j_θ·∇π - j_S·∇Σ_S) / π.

θ, S = model.tracers
w = model.velocities.w
∫θ, ∫S = compute!(Field(Integral(θ))), compute!(Field(Integral(S)))
∫θ₀, ∫S₀ = ∫θ[1, 1, 1], ∫S[1, 1, 1]

function progress(sim)
    compute!(∫θ)
    compute!(∫S)
    @printf("iteration: %d, time: %s, max|w|: %.2e m s⁻¹, ∫θ drift: %.1e, ∫S drift: %.1e\n",
            iteration(sim), prettytime(sim), maximum(abs, w), ∫θ[1, 1, 1] / ∫θ₀ - 1, ∫S[1, 1, 1] / ∫S₀ - 1)
end

add_callback!(simulation, progress, IterationInterval(100))

# We save θ, S, w and the temperature T, plus the horizontally averaged salinity.

output_dir = joinpath(@__DIR__, "output")
T = thermodynamic_diagnostics(model).T

simulation.output_writers[:fields] = JLD2Writer(model, (; θ, S, T, w); dir = output_dir, filename = "two_layer_cabbeling",
                                                schedule = TimeInterval(15), overwrite_files = true)

simulation.output_writers[:profiles] = JLD2Writer(model, (; S = Average(S, dims = (1, 2))); dir = output_dir,
                                                  filename = "two_layer_cabbeling_profiles",
                                                  schedule = TimeInterval(15), overwrite_files = true)

run!(simulation)

# ## Visualization
#
# We animate the salinity and vertical velocity,

filename = joinpath(output_dir, "two_layer_cabbeling.jld2")
S_timeseries = FieldTimeSeries(filename, "S")
w_timeseries = FieldTimeSeries(filename, "w")
times = S_timeseries.times

n = Observable(1)
title = @lift "t = " * prettytime(times[$n])
wlim = maximum(abs, interior(w_timeseries))

fig = Figure(size = (700, 600))
axS = Axis(fig[2, 1]; title = "S (kg/kg)", xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[2, 3]; title = "w (m s⁻¹)", xlabel = "x (m)")
Colorbar(fig[2, 2], heatmap!(axS, @lift(S_timeseries[$n]); colorrange = (S_shallow, S_deep)))
Colorbar(fig[2, 4], heatmap!(axw, @lift(w_timeseries[$n]); colormap = :balance, colorrange = (-wlim, wlim)))
fig[1, :] = Label(fig, title, tellwidth = false)

record(fig, joinpath(output_dir, "two_layer_cabbeling.mp4"), 1:length(times); framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](two_layer_cabbeling.mp4)

# and plot the horizontally averaged salinity against time, which shows the mixed layer
# spreading from the interface.

S̄ = FieldTimeSeries(joinpath(output_dir, "two_layer_cabbeling_profiles.jld2"), "S")
z = znodes(S̄.grid, Center())

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1]; xlabel = "t (min)", ylabel = "z (m)", title = "Horizontally averaged S (kg/kg)")
Colorbar(fig[1, 2], heatmap!(ax, S̄.times ./ minute, z, interior(S̄, 1, 1, :, :)'))
save(joinpath(output_dir, "two_layer_cabbeling_hovmoller.png"), fig)
nothing #hide

# ![](two_layer_cabbeling_hovmoller.png)
