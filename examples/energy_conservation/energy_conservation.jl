# # Energy conservation
#
# A deliberately extreme test of energy conservation: a two-dimensional Rayleigh–Taylor instability
# with a strongly nonlinear (cabbeling) equation of state and strong molecular diffusion, so that
# the kinetic energy 𝒦, the static energy 𝒫 = ∫ Σ dV, viscous dissipation and entropy production
# are all large. The total energy 𝒦 + 𝒫 changes only through the RK3 time-stepping error, which
# falls by about 8× when Δt is halved.
#
# This example demonstrates:
#
#   * How to build a thermodynamically consistent model with `thermodynamic_model_components`.
#   * How to output volume integrals of `thermodynamic_diagnostics` with `Integral`.
#
# Run with `julia --project=examples examples/energy_conservation/energy_conservation.jl`.
# Output and figures are saved in `examples/energy_conservation/output`.

using Oceananigans
using ThermoShenanigans
using Printf
using CairoMakie

# ## Thermodynamics
#
# The quadratic term ½ γ θ′² (≈ 1.3 × 10⁻³ for θ′ = 5 K) exceeds the linear term α θ′ (10⁻³).

thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, cabbeling_coefficient = 1e-4,
                                           reference_temperature = 283.15, reference_salinity = 0.035,
                                           reference_heat_capacity = 3991.86795711963)

# ## Grid and initial conditions

L = 0.1
grid = RectilinearGrid(size = (64, 64), x = (0, L), z = (-L, 0), topology = (Periodic, Flat, Bounded))

# Cold, salty, dense water lies over warm, fresh water, and mixtures are denser still (cabbeling).
# A few deterministic modes on the interface seed the instability.

interface(x) = - L / 2 + 5e-4 * (sin(2π * 3x / L) + sin(2π * 5x / L + 1) + sin(2π * 8x / L + 2))
layer(x, z) = tanh((z - interface(x)) / 2e-3) # +1 above, -1 below
θᵢ(x, z) = 283.15 - 5 * layer(x, z)
Sᵢ(x, z) = 0.035 + 1e-3 * layer(x, z)

# ## Simulations
#
# We run the same simulation with two time steps, each saving the volume integrals of the energy budget
# in double precision.
# 𝒦 is integrated at the native locations of `u` and `w`, so that it is exactly the discrete kinetic energy.
# Σ - Σ₀ is differenced cell by cell before integrating, because Σ ≈ cₚ θ is about 10⁶ times larger than
# its changes.

output_dir = joinpath(@__DIR__, "output")

function run_energy_budget(Δt; stop_time = 60, save_fields = false)
    components = thermodynamic_model_components(grid, thermo; viscosity = 2e-5, thermal_diffusivity = 2e-5, salt_diffusivity = 2e-5)
    model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)
    set!(model, θ = θᵢ, S = Sᵢ)

    u, v, w = model.velocities
    diagnostics = thermodynamic_diagnostics(model)
    Σ₀ = CenterField(grid)
    set!(Σ₀, diagnostics.Σ)

    budget = (𝒦ᵘ  = Integral(u^2 / 2),
              𝒦ʷ  = Integral(w^2 / 2),
              Δ𝒫  = Integral(diagnostics.Σ - Σ₀),
              wb  = Integral(diagnostics.wb),
              ε   = Integral(diagnostics.ε),
              σ_η = Integral(diagnostics.σ_η))

    simulation = Simulation(model; Δt, stop_time)

    progress(sim) = @printf("Δt: %.3f, iteration: %d, time: %s, max|w|: %.2e\n",
                            Δt, iteration(sim), prettytime(sim), maximum(abs, w))
    add_callback!(simulation, progress, IterationInterval(1000))

    filename = @sprintf("energy_budget_dt%.3f", Δt)
    simulation.output_writers[:budget] = JLD2Writer(model, budget; dir = output_dir, filename,
                                                    schedule = TimeInterval(0.1), array_type = Array{Float64},
                                                    overwrite_files = true)

    if save_fields
        simulation.output_writers[:fields] = JLD2Writer(model, (; θ = model.tracers.θ, w); dir = output_dir,
                                                        filename = "energy_conservation_fields",
                                                        schedule = TimeInterval(1), overwrite_files = true)
    end

    run!(simulation)

    return joinpath(output_dir, filename * ".jld2")
end

Δts = (0.04, 0.02)
filenames = [run_energy_budget(Δt; save_fields = Δt == first(Δts)) for Δt in Δts]
nothing #hide

# ## The energy budget
#
# We load the integrals with `FieldTimeSeries`. The energy error ΔE = Δ𝒦 + Δ𝒫 should fall by
# about 8× when Δt is halved. Below Δt ≈ 0.01 s it reaches a round-off floor of about 3 × 10⁻¹²,
# set by rounding θ ≈ 283 K at every time step.

series(filename, name) = interior(FieldTimeSeries(filename, name), 1, 1, 1, :)

function energy_changes(filename)
    𝒦 = series(filename, "𝒦ᵘ") .+ series(filename, "𝒦ʷ")
    Δ𝒦 = 𝒦 .- 𝒦[1]
    Δ𝒫 = series(filename, "Δ𝒫")
    return Δ𝒦, Δ𝒫, Δ𝒦 .+ Δ𝒫
end

errors = [maximum(abs, last(energy_changes(filename))) for filename in filenames]
@printf("max|ΔE| = %.3e (Δt = %.3f), %.3e (Δt = %.3f); ratio %.2f\n", errors[1], Δts[1], errors[2], Δts[2], errors[1] / errors[2])

t = FieldTimeSeries(filenames[1], "Δ𝒫").times
Δ𝒦, Δ𝒫, ΔE = energy_changes(filenames[1])

fig = Figure(size = (1200, 800))

ax = Axis(fig[1, 1]; title = "Energy changes (Δt = $(Δts[1]) s)", xlabel = "t (s)", ylabel = "m⁴ s⁻² (per m in y)")
lines!(ax, t, Δ𝒦; label = "Δ𝒦")
lines!(ax, t, Δ𝒫; label = "Δ𝒫")
lines!(ax, t, ΔE; label = "ΔE = Δ𝒦 + Δ𝒫", linewidth = 3, color = :black)
axislegend(ax; position = :lb)

ax = Axis(fig[1, 2]; title = "Relative energy error |ΔE| / max|Δ𝒦|", xlabel = "t (s)", yscale = log10)
for (Δt, filename) in zip(Δts, filenames)
    Δ𝒦ₙ, _, ΔEₙ = energy_changes(filename)
    lines!(ax, t[2:end], abs.(ΔEₙ[2:end]) ./ maximum(abs, Δ𝒦ₙ); label = "Δt = $Δt s")
end
axislegend(ax; position = :rb)

ax = Axis(fig[2, 1]; title = "Energy conversions", xlabel = "t (s)", ylabel = "m⁴ s⁻³ (per m in y)")
lines!(ax, t, series(filenames[1], "wb"); label = "∫ w b̃ dV  (𝒫 → 𝒦)")
lines!(ax, t, series(filenames[1], "ε"); label = "∫ ε dV  (𝒦 → internal)")
axislegend(ax; position = :rt)

ax = Axis(fig[2, 2]; title = "Entropy production ∫ σ_η dV", xlabel = "t (s)", ylabel = "m² s⁻³ K⁻¹ (per m in y)")
lines!(ax, t, series(filenames[1], "σ_η"))

save(joinpath(output_dir, "energy_conservation.png"), fig)
nothing #hide

# ![](energy_conservation.png)

# ## The flow
#
# Finally, we animate θ and w from the first simulation.

fields_filename = joinpath(output_dir, "energy_conservation_fields.jld2")
θ_timeseries = FieldTimeSeries(fields_filename, "θ")
w_timeseries = FieldTimeSeries(fields_filename, "w")
times = θ_timeseries.times

n = Observable(1)
title = @lift "t = " * prettytime(times[$n])

fig = Figure(size = (900, 450))
axθ = Axis(fig[2, 1]; title = "θ (K)", xlabel = "x (m)", ylabel = "z (m)", aspect = 1)
axw = Axis(fig[2, 3]; title = "w (m s⁻¹)", xlabel = "x (m)", aspect = 1)
Colorbar(fig[2, 2], heatmap!(axθ, @lift(θ_timeseries[$n]); colormap = :thermal, colorrange = (278.15, 288.15)))
Colorbar(fig[2, 4], heatmap!(axw, @lift(w_timeseries[$n]); colormap = :balance, colorrange = (-0.03, 0.03)))
fig[1, :] = Label(fig, title, tellwidth = false)

record(fig, joinpath(output_dir, "energy_conservation.mp4"), 1:length(times); framerate = 8) do i
    n[] = i
end
nothing #hide

# ![](energy_conservation.mp4)
