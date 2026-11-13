# # Cabbeling energetics III: the Lorenz energy cycle
#
# The two-layer cabbeling instability of `reference_state.jl` and `ape_budget.jl`, with the budgets of all three energy
# reservoirs: the kinetic energy 𝒦, the available potential energy 𝒜 and the background potential energy ℬ = 𝒫 - 𝒜
# (Hatton, Tailleux & Dubos). In every cell,
#
#   ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + w (b̃ - b★) - ε,
#   ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - w (b̃ - b★) + 𝒩 + R★,
#   ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ,
#
#   * A_𝒦: the advection of 𝒦, which centred second-order advection only redistributes;
#   * j_𝒦 = (p - p★) u - u·τ, j_𝒜 = Υ j_b and j_ℬ = p★ u + π(θ, z★) j_θ + m(z★) j_S: the fluxes of the three
#     reservoirs. j_ℬ holds the conduction of heat, π j_θ ≈ cₚ j_θ, and so the flux of internal energy;
#   * 𝒩 and 𝒩_ℬ: the production of 𝒜 and ℬ by the error of the discrete advection of θ and S. Their sum integrates to
#     zero, so that 𝒩 is a spurious conversion of ℬ into 𝒜: the spurious mixing of Ilıcak et al. (2012).
#
# Integrated over the domain, the flux divergences and R★ vanish, and the three reservoirs exchange energy through the
# conversions of a Lorenz energy cycle:
#
#   * ℬ → 𝒜: the production Υ σ_b by the buoyancy sources, split into heating and cabbeling, and 𝒩;
#   * 𝒜 → 𝒦: the work w (b̃ - b★) of the buoyancy anomaly;
#   * 𝒦 → ℬ: the viscous dissipation ε, which heats the fluid;
#   * 𝒜 → ℬ: the diffusive dissipation εₐ of 𝒜.
#
# This example demonstrates:
#
#   * How to diagnose the three local budgets, sharing one reference state between them.
#   * That the integrated cycle closes, with cabbeling as the source of 𝒜 that drives the instability.
#
# Run with `julia --project=examples examples/cabbeling_energetics/energy_cycle.jl`.
# Output and figures are saved in `examples/cabbeling_energetics/output`.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using Printf
using CairoMakie

# ## Thermodynamics and model
#
# As in `ape_budget.jl`, after experiment IV of Bisits, Zika & Sohail (2025), with κ = ν so that the flow is resolved.

θ_deep, S_deep       = 273.15 + 0.5, 0.0347
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ

thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                                   reference_temperature = θ_deep, reference_salinity = S_deep,
                                                   reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

grid = RectilinearGrid(size = (64, 512), x = (-0.035, 0.035), z = (-1, 0), topology = (Periodic, Flat, Bounded))

output_dir = joinpath(@__DIR__, "output")

components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-6,
                                            salt_diffusivity = 1e-6)
model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

# As in `ape_budget.jl`: a resolved interface with noise confined to it, and S decreasing upwards within each layer so
# that no two cells share a buoyancy.
δb = 1e-12
∂S∂z = - δb / (g * β * zspacings(grid, Center())[1])
interface_thickness = 5e-3
shallow_fraction(z) = (1 + tanh((z + 0.5) / interface_thickness)) / 2
interface_noise(z) = 2e-7 * randn() * 4 * shallow_fraction(z) * (1 - shallow_fraction(z))

θᵢ(x, z) = θ_deep + (θ_shallow - θ_deep) * shallow_fraction(z)
Sᵢ(x, z) = S_deep + (S_shallow - S_deep) * shallow_fraction(z) + ∂S∂z * (z + 0.5) + interface_noise(z)
set!(model, θ = θᵢ, S = Sᵢ)

simulation = Simulation(model; Δt = 0.25, stop_time = 15minutes)
w = model.velocities.w
progress(sim) = @printf("iteration: %d, time: %s, max|w|: %.2e m s⁻¹\n", iteration(sim), prettytime(sim), maximum(abs, w))
add_callback!(simulation, progress, IterationInterval(800))

# ## Diagnostics
#
# All the terms share one reference height z★. ℬ is dominated by the internal energy h⁰, which is much larger than
# its changes, so its change from the initial ℬ₀ is computed cell by cell. ℬ₀ is a plain field set once, rather than a
# `Field` of the operation, which would be recomputed with the others.

z★ = reference_height(model)
ℬ₀ = CenterField(grid)
set!(ℬ₀, background_potential_energy(model, z★))

terms = map(Field, (; 𝒦 = kinetic_energy(model),
                      𝒜 = available_potential_energy(z★),
                      Δℬ = background_potential_energy(model, z★) - ℬ₀,
                      σ_𝒜_heating = heating_ape_production(model, z★),
                      σ_𝒜_nonlin = nonlinear_ape_production(model, z★),
                      ε_𝒜 = ape_dissipation(model, z★),
                      wb_𝒜 = available_buoyancy_work(model, z★),
                      ε = viscous_dissipation(model),
                      𝒩 = spurious_ape_production(model, z★),
                      𝒩_ℬ = spurious_bpe_production(model, z★),
                      A_𝒦 = ke_advection(model)))

simulation.output_writers[:integrals] = JLD2Writer(model, map(Integral, terms); dir = output_dir,
                                                   filename = "energy_cycle_integrals", schedule = TimeInterval(10),
                                                   array_type = Array{Float64}, overwrite_files = true)

fluxes = (; j_𝒜 = Field(ape_flux(model, z★).z), j_ℬ = Field(bpe_flux(model, z★).z))
simulation.output_writers[:fluxes] = JLD2Writer(model, fluxes; dir = output_dir, filename = "energy_cycle_fluxes",
                                                schedule = TimeInterval(1minute), overwrite_files = true)
run!(simulation)

# ## The integrated budgets
#
# Each reservoir changes by the time integral of its conversions, from their values every 10 s. The residuals are the
# errors of this trapezoidal time integration and of the time stepping. ∫A_𝒦 dV vanishes to round-off, and so does
# ∫(𝒩 + 𝒩_ℬ) dV = ∫(π ADV_θ + m ADV_S + w b̃) dV, though to the round-off of π ADV_θ ≈ cₚ ADV_θ, which is much
# larger than 𝒩.

integrals_file = joinpath(output_dir, "energy_cycle_integrals.jld2")
t = FieldTimeSeries(integrals_file, "𝒦").times
series = NamedTuple(name => interior(FieldTimeSeries(integrals_file, string(name)), 1, 1, 1, :) for name in keys(terms))
cumulative(f) = vcat(0, cumsum((f[2:end] .+ f[1:end-1]) ./ 2 .* diff(t)))

conversions = (ℬ_to_𝒜_heating  = series.σ_𝒜_heating,
               ℬ_to_𝒜_cabbeling = series.σ_𝒜_nonlin,
               ℬ_to_𝒜_spurious  = series.𝒩,
               𝒜_to_𝒦           = series.wb_𝒜,
               𝒦_to_ℬ           = series.ε,
               𝒜_to_ℬ           = series.ε_𝒜)

rates = (𝒦 = conversions.𝒜_to_𝒦 .- conversions.𝒦_to_ℬ .+ series.A_𝒦,
         𝒜 = conversions.ℬ_to_𝒜_heating .+ conversions.ℬ_to_𝒜_cabbeling .+ conversions.ℬ_to_𝒜_spurious .-
             conversions.𝒜_to_𝒦 .- conversions.𝒜_to_ℬ,
         Δℬ = .- conversions.ℬ_to_𝒜_heating .- conversions.ℬ_to_𝒜_cabbeling .+ conversions.𝒜_to_ℬ .+
              conversions.𝒦_to_ℬ .+ series.𝒩_ℬ)

for name in keys(rates)
    change = series[name] .- series[name][1]
    @printf("%-2s: max|residual| / max|change| = %.1e\n", name, maximum(abs, change .- cumulative(rates[name])) / maximum(abs, change))
end
@printf("max|∫(𝒩 + 𝒩_ℬ) dV| / max|∫𝒩 dV| = %.1e\n", maximum(abs, series.𝒩 .+ series.𝒩_ℬ) / maximum(abs, series.𝒩))
@printf("max|∫A_𝒦 dV| / max|∫w (b̃ - b★) dV| = %.1e\n", maximum(abs, series.A_𝒦) / maximum(abs, series.wb_𝒜))

# ## The Lorenz energy cycle
#
# The changes of the reservoirs and the conversions between them, integrated over the run. Cabbeling converts ℬ into 𝒜,
# which the flow converts into 𝒦; viscosity and diffusion return it to ℬ.

integrated = map(f -> cumulative(f)[end], conversions)
changes = map(name -> series[name][end] - series[name][1], (𝒦 = :𝒦, 𝒜 = :𝒜, ℬ = :Δℬ))

fig = Figure(size = (900, 650))
ax = Axis(fig[1, 1]; title = @sprintf("Lorenz energy cycle over %s (m⁴ s⁻², per m in y)", prettytime(t[end])),
          aspect = DataAspect())
hidedecorations!(ax)
hidespines!(ax)

centres = (𝒜 = Point2f(0, 2), 𝒦 = Point2f(3, 0), ℬ = Point2f(-3, 0))
for (name, centre) in pairs(centres)
    poly!(ax, Rect2f(centre - Vec2f(0.9, 0.45), Vec2f(1.8, 0.9)); color = (:steelblue, 0.15), strokecolor = :steelblue,
          strokewidth = 2)
    text!(ax, centre; text = @sprintf("%s\nΔ = %.2e", name, changes[name]), align = (:center, :center), fontsize = 18)
end

# An arrow from box `a` to box `b`, shifted to its left so that the arrows of opposite conversions are side by side,
# with its label further to its left
function conversion_arrow!(ax, a, b, label; shift = 0.3, label_distance = 0.35)
    separation = centres[b] - centres[a]
    direction = separation / hypot(separation...)
    left = Vec2f(-direction[2], direction[1])
    start = centres[a] + 1.1 * direction + shift * left
    finish = centres[b] - 1.1 * direction + shift * left
    arrows2d!(ax, [start], [finish - start]; shaftwidth = 3, tipwidth = 14, tiplength = 14, color = :black)
    text!(ax, (start + finish) / 2 + label_distance * left; text = label, align = (:center, :center), fontsize = 13)
end

conversion_arrow!(ax, :ℬ, :𝒜, @sprintf("heating %.2e\ncabbeling %.2e\nspurious 𝒩 %.2e", integrated.ℬ_to_𝒜_heating,
                                       integrated.ℬ_to_𝒜_cabbeling, integrated.ℬ_to_𝒜_spurious); label_distance = 0.6)
conversion_arrow!(ax, :𝒜, :ℬ, @sprintf("εₐ %.2e", integrated.𝒜_to_ℬ))
conversion_arrow!(ax, :𝒜, :𝒦, @sprintf("w (b̃ - b★) %.2e", integrated.𝒜_to_𝒦); label_distance = 0.6)
conversion_arrow!(ax, :𝒦, :ℬ, @sprintf("ε %.2e", integrated.𝒦_to_ℬ))
xlims!(ax, -4.3, 4.3)
ylims!(ax, -1.2, 3.1)
save(joinpath(output_dir, "energy_cycle_diagram.png"), fig)
nothing #hide

# ![](energy_cycle_diagram.png)

# ## Conversion rates
#
# The cabbeling production of 𝒜 grows with the mixed water at the interface, and drives the conversion into 𝒦.

fig = Figure(size = (1100, 450))
ax = Axis(fig[1, 1]; title = "Integrated conversion rates", xlabel = "t (min)", ylabel = "m⁴ s⁻³ (per m in y)")
labels = (ℬ_to_𝒜_heating = "ℬ → 𝒜: ∫Υ σ_b_heating", ℬ_to_𝒜_cabbeling = "ℬ → 𝒜: ∫Υ σ_b_nonlin", ℬ_to_𝒜_spurious = "ℬ → 𝒜: ∫𝒩",
          𝒜_to_𝒦 = "𝒜 → 𝒦: ∫w (b̃ - b★)", 𝒦_to_ℬ = "𝒦 → ℬ: ∫ε", 𝒜_to_ℬ = "𝒜 → ℬ: ∫εₐ")
for name in keys(conversions)
    lines!(ax, t ./ minute, conversions[name]; label = labels[name])
end
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; title = "Changes of the reservoirs", xlabel = "t (min)", ylabel = "m⁴ s⁻² (per m in y)")
for (name, label) in ((:𝒦, "Δ∫𝒦"), (:𝒜, "Δ∫𝒜"), (:Δℬ, "Δ∫ℬ"))
    lines!(ax, t ./ minute, series[name] .- series[name][1]; label)
end
axislegend(ax; position = :lt)
save(joinpath(output_dir, "energy_cycle.png"), fig)
nothing #hide

# ![](energy_cycle.png)

# ## The fluxes of 𝒜 and ℬ
#
# The vertical fluxes j_𝒜 = Υ j_b and j_ℬ at the onset of the instability. j_ℬ is dominated by the conduction of heat
# across the interface.

fluxes_file = joinpath(output_dir, "energy_cycle_fluxes.jld2")
fluxseries = (j_𝒜 = FieldTimeSeries(fluxes_file, "j_𝒜"), j_ℬ = FieldTimeSeries(fluxes_file, "j_ℬ"))
times = fluxseries.j_𝒜.times
snapshots = [findfirst(≥(t), times) for t in (5minutes, 10minutes, 15minutes)]
titles = (j_𝒜 = "j_𝒜 · 𝐤 (m³ s⁻³)", j_ℬ = "j_ℬ · 𝐤 (m³ s⁻³)")

fig = Figure(size = (1100, 650))
for (row, name) in enumerate(keys(fluxseries))
    flux_series = fluxseries[name]
    limit = maximum(abs, interior(flux_series))
    local hm, ax
    for (column, n) in enumerate(snapshots)
        ax = Axis(fig[row, column]; title = titles[name] * ", t = " * prettytime(times[n]), ylabel = "z (m)",
                  xlabel = row == length(fluxseries) ? "x (m)" : "")
        hm = heatmap!(ax, flux_series[n]; colormap = :balance, colorrange = (-limit, limit))
        ylims!(ax, -0.7, -0.3)
    end
    Colorbar(fig[row, length(snapshots) + 1], hm)
end
save(joinpath(output_dir, "energy_cycle_fluxes.png"), fig)
nothing #hide

# ![](energy_cycle_fluxes.png)
