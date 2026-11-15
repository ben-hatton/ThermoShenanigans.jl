# # Cabbeling energetics II: the local available potential energy budget
#
# The two-layer cabbeling instability of `reference_state.jl`, with the budget of the local available potential
# energy 𝒜 = b Υ + 𝒲★ (Hatton, Tailleux & Dubos), where Υ = z★ - z. With Db/Dt = σ_b - ∇·j_b, in every cell
#
#   ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - w (b̃ - b★) + 𝒩 + R★,
#
#   * Υ σ_b: the production of 𝒜 by the buoyancy sources, split into the heating part Υ b_θ σ_θ and the
#     cabbeling part Υ (j_θ·∇b_θ + j_S·∇b_S), which vanishes for a linear equation of state;
#   * εₐ = j_b·∇(z - z★): the diffusive dissipation of 𝒜, positive when j_b points down the gradient of z - z★;
#   * j_𝒜 = Υ j_b: the diffusive flux of 𝒜;
#   * w (b̃ - b★): the conversion of 𝒜 into kinetic energy;
#   * 𝒩: the production of 𝒜 by the error of the discrete advection of θ and S, which vanishes in the continuum.
#     Its integral is minus the change of the background potential energy by advection, the spurious mixing of
#     Ilıcak et al. (2012): negative when advection mixes and positive when its dispersive errors unmix;
#   * R★ = Ṗ★(z) - Ṗ★(z★): the work of the changing reference pressure, which integrates to zero.
#
# This example demonstrates:
#
#   * How to diagnose the terms of the local APE budget, sharing one reference state between them.
#   * That cabbeling produces 𝒜 where mixed water forms at the interface, and that the integrated budget closes,
#     with a small production 𝒩 by the error of the discrete advection when the flow is resolved.
#
# Run with `julia --project=examples examples/cabbeling_energetics/ape_budget.jl`.
# Output and figures are saved in `examples/cabbeling_energetics/output`.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using Printf
using CairoMakie

# ## Thermodynamics and model
#
# As in `reference_state.jl`, after experiment IV of Bisits, Zika & Sohail (2025), with κ = ν so that the flow is
# resolved (see there). The thermal expansion α is set
# so that the shallow layer is lighter than the deep one by Δb, whatever the cabbeling coefficient γ, so that the
# run with γ = 0 has the same initial stratification but no cabbeling.

θ_deep, S_deep       = 273.15 + 0.5, 0.0347
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458

g, β = 9.80665, 0.78
Δb = g * 0.0046 / 1027
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep

grid = RectilinearGrid(size = (64, 512), x = (-0.035, 0.035), z = (-1, 0), topology = (Periodic, Flat, Bounded))

output_dir = joinpath(@__DIR__, "output")

# Within each layer S decreases upwards, so that b increases by δb per level and no two cells share a buoyancy
# (see `reference_state.jl`).
δb = 1e-12
∂S∂z = - δb / (g * β * zspacings(grid, Center())[1])

# As in `reference_state.jl`, the interface is resolved from the start and the noise is confined to it. Either a
# step or noise in the nearly uniform layers would give an initial transient too fast for the budget to be
# closed from its values every 10 s.
interface_thickness = 5e-3
shallow_fraction(z) = (1 + tanh((z + 0.5) / interface_thickness)) / 2
interface_noise(z) = 2e-7 * randn() * 4 * shallow_fraction(z) * (1 - shallow_fraction(z))

function run_ape_budget(cabbeling_coefficient; stop_time = 15minutes)
    γ = cabbeling_coefficient
    α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ
    thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                                       reference_temperature = θ_deep, reference_salinity = S_deep,
                                                       reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

    components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-6,
                                                salt_diffusivity = 1e-6)
    model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

    θᵢ(x, z) = θ_deep + (θ_shallow - θ_deep) * shallow_fraction(z)
    Sᵢ(x, z) = S_deep + (S_shallow - S_deep) * shallow_fraction(z) + ∂S∂z * (z + 0.5) + interface_noise(z)
    set!(model, θ = θᵢ, S = Sᵢ)

    simulation = Simulation(model; Δt = 0.25, stop_time)
    w = model.velocities.w
    progress(sim) = @printf("γ = %.0e, iteration: %d, time: %s, max|w|: %.2e m s⁻¹\n", γ, iteration(sim), prettytime(sim),
                            maximum(abs, w))
    add_callback!(simulation, progress, IterationInterval(800))

    # The terms of the budget are built on one reference height z★, so that they share one computation of the
    # reference state per output time. Each term is wrapped in a Field once and passed to both output writers: the
    # fields are saved every minute for the maps, and their integrals every 10 s for the budget. The flux
    # divergences ∇·j_𝒜 and ∇·(u 𝒜) integrate to zero, so they are left out.
    z★ = reference_height(model)
    terms = map(Field, (; 𝒜 = available_potential_energy(z★),
                          σ_𝒜_heating = heating_ape_production(model, z★),
                          σ_𝒜_nonlin = nonlinear_ape_production(model, z★),
                          ε_𝒜 = ape_dissipation(model, z★),
                          wb_𝒜 = available_buoyancy_work(model, z★),
                          𝒩 = spurious_ape_production(model, z★),
                          R★ = reference_tendency_work(model, z★)))
    prefix = γ == 0 ? "ape_budget_linear" : "ape_budget"

    simulation.output_writers[:fields] = JLD2Writer(model, (; 𝒜 = terms.𝒜, σ_𝒜_nonlin = terms.σ_𝒜_nonlin, ε_𝒜 = terms.ε_𝒜,
                                                            𝒩 = terms.𝒩, R★ = terms.R★);
                                                    dir = output_dir, filename = prefix * "_fields",
                                                    schedule = TimeInterval(1minute), overwrite_files = true)

    integrals = map(Integral, terms)

    simulation.output_writers[:integrals] = JLD2Writer(model, integrals; dir = output_dir, filename = prefix * "_integrals",
                                                       schedule = TimeInterval(10), array_type = Array{Float64},
                                                       overwrite_files = true)
    run!(simulation)

    return joinpath(output_dir, prefix * "_integrals.jld2"), joinpath(output_dir, prefix * "_fields.jld2")
end

integrals_file, fields_file = run_ape_budget(1e-5)
linear_integrals_file, _ = run_ape_budget(0)

# ## Where 𝒜 is produced and destroyed
#
# Snapshots of 𝒜, its production by cabbeling, its diffusive dissipation εₐ, its production 𝒩 by the error of the
# discrete advection, and the reference term R★. Cabbeling produces 𝒜 in the mixed water at the interface, which is
# denser than both layers and so sits above its reference height.

fieldseries = map(name -> FieldTimeSeries(fields_file, name), (𝒜 = "𝒜", σ_𝒜_nonlin = "σ_𝒜_nonlin", ε_𝒜 = "ε_𝒜", 𝒩 = "𝒩",
                                                               R★ = "R★"))
times = fieldseries.𝒜.times
snapshots = [findfirst(≥(t), times) for t in (5minutes, 10minutes, 15minutes)]
titles = (𝒜 = "𝒜 (m² s⁻²)", σ_𝒜_nonlin = "Υ σ_b_nonlin (m² s⁻³)", ε_𝒜 = "εₐ (m² s⁻³)", 𝒩 = "𝒩 (m² s⁻³)",
          R★ = "R★ (m² s⁻³)")

fig = Figure(size = (1100, 1500))
for (row, name) in enumerate(keys(fieldseries))
    series = fieldseries[name]
    limit = maximum(abs, interior(series))
    colormap, colorrange = name == :𝒜 ? (:amp, (0, limit)) : (:balance, (-limit, limit))
    local hm
    for (column, n) in enumerate(snapshots)
        ax = Axis(fig[row, column]; title = titles[name] * ", t = " * prettytime(times[n]), ylabel = "z (m)",
                  xlabel = row == length(fieldseries) ? "x (m)" : "")
        hm = heatmap!(ax, series[n]; colormap, colorrange)
        ylims!(ax, -0.7, -0.3)
    end
    Colorbar(fig[row, length(snapshots) + 1], hm)
end
save(joinpath(output_dir, "ape_budget_maps.png"), fig)
nothing #hide

# ![](ape_budget_maps.png)

# ## The integrated budget
#
# Integrated over the domain, d∫𝒜 dV/dt = ∫ [Υ σ_b - εₐ - w (b̃ - b★) + 𝒩] dV, since ∫ R★ dV and the integrals of the
# flux divergences vanish. This holds exactly for the semi-discrete model. We compare the change in ∫𝒜 dV with the
# time integral of the right-hand side, from its values every 10 s: the residual is the error of this trapezoidal
# time integration, and of the time stepping. Here the flow is resolved, and 𝒩 is small next to the conversion
# w (b̃ - b★). With κ = 10⁻⁷ on a grid twice as coarse, it is as large as the conversion, and the budget misses
# about 15% of the change of ∫𝒜 dV without it. Without cabbeling (γ = 0), mixing at the interface only weakens a
# stable stratification, so ∫𝒜 dV stays five orders of magnitude smaller and no instability develops.

function budget_series(filename)
    series(name) = interior(FieldTimeSeries(filename, name), 1, 1, 1, :)
    t = FieldTimeSeries(filename, "𝒜").times
    terms = map(series, (𝒜 = "𝒜", σ_𝒜_heating = "σ_𝒜_heating", σ_𝒜_nonlin = "σ_𝒜_nonlin", ε_𝒜 = "ε_𝒜", wb_𝒜 = "wb_𝒜",
                         𝒩 = "𝒩", R★ = "R★"))
    cumulative(f) = vcat(0, cumsum((f[2:end] .+ f[1:end-1]) ./ 2 .* diff(t)))
    rhs = terms.σ_𝒜_heating .+ terms.σ_𝒜_nonlin .- terms.ε_𝒜 .- terms.wb_𝒜 .+ terms.𝒩
    Δ𝒜 = terms.𝒜 .- terms.𝒜[1]
    return (; t, terms, Δ𝒜, rhs, residual = Δ𝒜 .- cumulative(rhs), residual_without_𝒩 = Δ𝒜 .- cumulative(rhs .- terms.𝒩),
              cumulative)
end

budget = budget_series(integrals_file)
linear_budget = budget_series(linear_integrals_file)

scale = maximum(abs, budget.Δ𝒜)
@printf("max|∫R★ dV| / max|∫εₐ dV| = %.1e\n", maximum(abs, budget.terms.R★) / maximum(abs, budget.terms.ε_𝒜))
@printf("max|residual| / max|Δ∫𝒜 dV| = %.1e, and without 𝒩: %.1e\n", maximum(abs, budget.residual) / scale,
        maximum(abs, budget.residual_without_𝒩) / scale)
@printf("max|∫Υ σ_b_nonlin dV| with γ = 0: %.1e\n", maximum(abs, linear_budget.terms.σ_𝒜_nonlin))
@printf("max ∫𝒜 dV with γ = 1e-5: %.2e, with γ = 0: %.2e\n", maximum(budget.terms.𝒜), maximum(linear_budget.terms.𝒜))

t = budget.t ./ minute
fig = Figure(size = (1100, 450))
ax = Axis(fig[1, 1]; title = "Integrated APE budget (γ = 10⁻⁵)", xlabel = "t (min)", ylabel = "m⁴ s⁻² (per m in y)")
for (name, label) in ((:σ_𝒜_heating, "∫Υ σ_b_heating"), (:σ_𝒜_nonlin, "∫Υ σ_b_nonlin"), (:ε_𝒜, "-∫εₐ"), (:wb_𝒜, "-∫w (b̃ - b★)"),
                      (:𝒩, "∫𝒩"))
    sign = name in (:ε_𝒜, :wb_𝒜) ? -1 : 1
    lines!(ax, t, budget.cumulative(sign .* budget.terms[name]); label = label * " dt")
end
lines!(ax, t, budget.Δ𝒜; label = "Δ∫𝒜", color = :black, linewidth = 3)
lines!(ax, t, budget.residual; label = "residual", color = :black, linestyle = :dash)
axislegend(ax; position = :lt)

ax = Axis(fig[1, 2]; title = "∫𝒜 dV", xlabel = "t (min)", yscale = log10)
lines!(ax, t[2:end], budget.terms.𝒜[2:end]; label = "γ = 10⁻⁵")
lines!(ax, t[2:end], max.(linear_budget.terms.𝒜[2:end], eps()); label = "γ = 0")
axislegend(ax; position = :rb)
save(joinpath(output_dir, "ape_budget.png"), fig)
nothing #hide

# ![](ape_budget.png)
