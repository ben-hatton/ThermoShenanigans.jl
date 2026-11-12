# # Cabbeling energetics I: reference state, available and background potential energy
#
# The two-layer cabbeling instability of `examples/two_layer_cabbeling`, diagnosed with the local energetics of
# Hatton, Tailleux & Dubos: cold, fresh water over warm, salty water, statically stable but with mixtures denser
# than either layer.
#
# The Lorenz reference state is found by sorting the fluid by buoyancy. Each parcel then has a reference height z★,
# and the potential energy 𝒫 = h⁰ - b (z - z₀) splits, cell by cell, into
#
#   * the available potential energy 𝒜 = - b (z - z★) + 𝒲★ ≥ 0, with 𝒲★ = p★(z) - p★(z★) the work against the
#     reference pressure p★ = ∫ b★ dz, and
#   * the background potential energy 𝓑 = ℐ + 𝒢★, with internal energy ℐ = h⁰ and gravitational part
#     𝒢★ = - b (z★ - z₀) - 𝒲★.
#
# This example demonstrates:
#
#   * How to compute the reference state and local energetics with `energetics`.
#   * How to output the reference buoyancy profile b★(z) alongside fields and volume integrals.
#
# Run with `julia --project=examples examples/cabbeling_energetics/reference_state.jl`.
# Output and figures are saved in `examples/cabbeling_energetics/output`.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using Printf
using CairoMakie

# ## Thermodynamics and model
#
# As in `examples/two_layer_cabbeling`, after experiment IV of Bisits, Zika & Sohail (2025). The reference
# values are those of the deep layer, so b = 0 there.

θ_deep, S_deep       = 273.15 + 0.5, 0.0347
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ

thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                                   reference_temperature = θ_deep, reference_salinity = S_deep,
                                                   reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

grid = RectilinearGrid(size = (32, 256), x = (-0.035, 0.035), z = (-1, 0), topology = (Periodic, Flat, Bounded))

components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7)
model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

# Within each layer, S decreases upwards by a tiny amount, so that b increases by δb = 10⁻¹² m s⁻² per grid level.
# Otherwise all the cells of a uniform layer would have the same buoyancy, and where they sit in the sorted reference
# state would be decided by round-off. δb is about 10⁴ times the round-off in b, but is negligible for the dynamics.

δb = 1e-12
∂S∂z = - δb / (g * β * zspacings(grid, Center())[1])

θᵢ(x, z) = z > -0.5 ? θ_shallow : θ_deep
Sᵢ(x, z) = (z > -0.5 ? S_shallow : S_deep) + ∂S∂z * (z + 0.5) + 2e-7 * randn() * exp(-((z + 0.5) / 0.02)^2)
set!(model, θ = θᵢ, S = Sᵢ)

simulation = Simulation(model; Δt = 0.5, stop_time = 10minutes)

u, v, w = model.velocities
progress(sim) = @printf("iteration: %d, time: %s, max|w|: %.2e m s⁻¹\n", iteration(sim), prettytime(sim), maximum(abs, w))
add_callback!(simulation, progress, IterationInterval(200))

# ## Energetics
#
# All the diagnostics of `energetics` share one sort of the buoyancy per output time.

e = energetics(model)

# We save 𝒜 and the displacement z - z★ = - Υ, and the reference buoyancy b★ at the w-faces, where it is the
# discrete gradient of p★.

output_dir = joinpath(@__DIR__, "output")

simulation.output_writers[:fields] = JLD2Writer(model, (; 𝒜 = e.𝒜, displacement = -e.Υ, S = model.tracers.S);
                                                dir = output_dir, filename = "reference_state_fields",
                                                schedule = TimeInterval(15), overwrite_files = true)

simulation.output_writers[:profiles] = JLD2Writer(model, (; b★ = e.b★); dir = output_dir, filename = "reference_state_profiles",
                                                  schedule = TimeInterval(15), overwrite_files = true)

# Volume integrals (per metre in y). Changes in ℐ are differenced cell by cell from the initial state, since
# ℐ = cₚ₀ (θ - T₀) is much larger than its changes. ∫ 𝒲★ dV vanishes, up to round-off.

ℐ₀ = CenterField(grid)
set!(ℐ₀, e.ℐ)

budget = (𝒦  = Integral((u^2 + w^2) / 2),
          𝒜  = Integral(e.𝒜),
          𝒢★ = Integral(e.𝒢★),
          Δℐ = Integral(e.ℐ - ℐ₀),
          𝒲★ = Integral(e.𝒲★))

simulation.output_writers[:budget] = JLD2Writer(model, budget; dir = output_dir, filename = "reference_state_budget",
                                                schedule = TimeInterval(5), array_type = Array{Float64}, overwrite_files = true)

run!(simulation)

# ## The reference buoyancy profile
#
# b★(z, t) starts as two layers. As diffusion mixes them at the interface, it forms water denser than the deep
# layer (b < 0, blue), which settles to the bottom of the reference state: cabbeling.

b★ = FieldTimeSeries(joinpath(output_dir, "reference_state_profiles.jld2"), "b★")
zᶠ = znodes(b★.grid, Face())
blim = maximum(abs, interior(b★))

fig = Figure(size = (800, 450))
ax = Axis(fig[1, 1]; xlabel = "t (min)", ylabel = "z (m)", title = "Reference buoyancy b★ (m s⁻²)")
Colorbar(fig[1, 2], heatmap!(ax, b★.times ./ minute, zᶠ, interior(b★, 1, 1, :, :)'; colormap = :balance, colorrange = (-blim, blim)))
save(joinpath(output_dir, "reference_buoyancy.png"), fig)
nothing #hide

# ![](reference_buoyancy.png)

# ## Available potential energy and displacement
#
# Snapshots of the salinity, the displacement z - z★ of each parcel from its reference height, and 𝒜.
# z★ depends on b alone, so all the cells of a uniform layer share one reference height, the middle of the band
# that layer fills in the sorted column: z - z★ then varies linearly across the layer, while 𝒜 = 0 there.
# 𝒜 is concentrated where the mixed, cabbeled water sits above its reference height.

filename = joinpath(output_dir, "reference_state_fields.jld2")
𝒜_timeseries = FieldTimeSeries(filename, "𝒜")
displacement_timeseries = FieldTimeSeries(filename, "displacement")
S_timeseries = FieldTimeSeries(filename, "S")
times = 𝒜_timeseries.times

snapshots = [findfirst(≥(t), times) for t in (2minutes, 5minutes, 10minutes)]
𝒜max = maximum(interior(𝒜_timeseries))
dmax = maximum(abs, interior(displacement_timeseries))

fig = Figure(size = (1100, 900))
for (column, n) in enumerate(snapshots)
    axS = Axis(fig[1, column]; title = "S (kg/kg), t = " * prettytime(times[n]), ylabel = "z (m)")
    axd = Axis(fig[2, column]; title = "z - z★ (m)", ylabel = "z (m)")
    ax𝒜 = Axis(fig[3, column]; title = "𝒜 (m² s⁻²)", xlabel = "x (m)", ylabel = "z (m)")
    hmS = heatmap!(axS, S_timeseries[n]; colorrange = (S_shallow, S_deep))
    hmd = heatmap!(axd, displacement_timeseries[n]; colormap = :balance, colorrange = (-dmax, dmax))
    hm𝒜 = heatmap!(ax𝒜, 𝒜_timeseries[n]; colormap = :amp, colorrange = (0, 𝒜max))
    column == length(snapshots) && foreach((row, hm) -> Colorbar(fig[row, column + 1], hm), 1:3, (hmS, hmd, hm𝒜))
end
save(joinpath(output_dir, "available_potential_energy.png"), fig)
nothing #hide

# ![](available_potential_energy.png)

# ## Energy reservoirs
#
# The kinetic energy 𝒦, the available potential energy 𝒜, and the changes in the gravitational and internal
# parts of the background potential energy 𝓑 = ℐ + 𝒢★. Their sum, the total energy, changes only through the
# time-stepping error.

series(name) = interior(FieldTimeSeries(joinpath(output_dir, "reference_state_budget.jld2"), name), 1, 1, 1, :)

t = FieldTimeSeries(joinpath(output_dir, "reference_state_budget.jld2"), "𝒦").times ./ minute
𝒦, 𝒜, 𝒢★, Δℐ, 𝒲★ = map(series, ("𝒦", "𝒜", "𝒢★", "Δℐ", "𝒲★"))
Δ𝒦, Δ𝒜, Δ𝒢★ = 𝒦 .- 𝒦[1], 𝒜 .- 𝒜[1], 𝒢★ .- 𝒢★[1]
ΔE = Δ𝒦 .+ Δ𝒜 .+ Δ𝒢★ .+ Δℐ

@printf("max|∫𝒲★ dV| / max ∫𝒜 dV = %.1e\n", maximum(abs, 𝒲★) / maximum(𝒜))
@printf("max|ΔE| / max|Δ𝒢★| = %.1e\n", maximum(abs, ΔE) / maximum(abs, Δ𝒢★))

fig = Figure(size = (1000, 400))
ax = Axis(fig[1, 1]; title = "Energy changes", xlabel = "t (min)", ylabel = "m⁴ s⁻² (per m in y)")
lines!(ax, t, Δ𝒦; label = "Δ𝒦")
lines!(ax, t, Δ𝒜; label = "Δ𝒜")
lines!(ax, t, Δ𝒢★; label = "Δ𝒢★")
lines!(ax, t, Δℐ; label = "Δℐ")
lines!(ax, t, ΔE; label = "ΔE", color = :black, linewidth = 3)
axislegend(ax; position = :lb)

ax = Axis(fig[1, 2]; title = "Kinetic and available potential energy", xlabel = "t (min)", yscale = log10)
lines!(ax, t[2:end], 𝒦[2:end]; label = "∫𝒦 dV")
lines!(ax, t[2:end], 𝒜[2:end]; label = "∫𝒜 dV")
axislegend(ax; position = :rb)
save(joinpath(output_dir, "energy_reservoirs.png"), fig)
nothing #hide

# ![](energy_reservoirs.png)
