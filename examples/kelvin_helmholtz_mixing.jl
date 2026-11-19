# # Kelvin–Helmholtz mixing
#
# A stratified shear layer, U(z) = U₀ tanh(z/h) with buoyancy B(z) = B₀ tanh(z/h), is unstable when its minimum
# Richardson number is below ¼ (Hazel 1972, [doi:10.1017/S0022112072001065](https://doi.org/10.1017/S0022112072001065)).
# The instability rolls the interface into a billow, which stirs the two layers until diffusion mixes them.
# Mixing raises the background potential energy ℬ irreversibly (Winters et al. 1995,
# [doi:10.1017/S002211209500125X](https://doi.org/10.1017/S002211209500125X)).
#
# With the thermodynamically consistent closure, the energy dissipated by viscosity heats the fluid, so
# kinetic plus potential energy, 𝒦 + 𝒫, is conserved exactly, and ℬ gains energy from two sinks: the APE
# dissipation εₐ (mixing) and the viscous dissipation ε (heating). Their ratio is the mixing efficiency
# (Peltier & Caulfield 2003, [doi:10.1146/annurev.fluid.35.101101.161144](https://doi.org/10.1146/annurev.fluid.35.101101.161144)).
#
# This example demonstrates:
#
#   * how to build a model with `thermodynamic_model_components`;
#   * how to diagnose the energy reservoirs 𝒦, 𝒜, ℬ and the conversions between them;
#   * that 𝒦 + 𝒫 is conserved to round-off while the billow mixes the layers.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# Pkg.add(url = "https://github.com/ben-hatton/ThermoShenanigans.jl")
# ```

using Oceananigans
using ThermoShenanigans
using Printf
using CairoMakie

# ## Parameters
#
# The flow is set by the Reynolds number Re = U₀ h / ν = 1000, the Prandtl number Pr = ν / κ_θ = 1 and the bulk
# Richardson number Ri₀ = B₀ h / U₀² = 0.1. We fix ν and the temperature difference across the layer, 
# Δθ = 2B₀ / (g α), which then set h, U₀ and B₀.

Re, Pr, J = 1000, 1, 0.1
g, α, T₀, S₀, cₚ₀ = 9.81, 2e-4, 283.15, 0.035, 3991.87
ν, Δθ = 1e-6, 1.0

h   = (2J * Re^2 * ν^2 / (g * α * Δθ))^(1/3)
U₀  = Re * ν / h
B₀  = J * U₀^2 / h
κ_θ = ν / Pr

# The fastest-growing Taylor–Goldstein mode of these profiles, between walls at z = ±5h, has wavenumber
# k h ≈ 0.45 and growth rate σ h / U₀ ≈ 0.12. The domain holds one wavelength, so neighbouring billows
# cannot pair.

k  = 0.45 / h
Lx = 2π / k
Lz = 10h
Nx, Nz = round(Int, 16Lx / h), round(Int, 16Lz / h)

# The consistent closure differs from a Fickian one by dissipative heating and by conduction down the adiabatic
# temperature gradient. Their sizes are set by the dissipation number Di = g α L_z / cₚ₀ and by Di T₀ / Δθ:

Di = g * α * Lz / cₚ₀
@printf("h = %.3e m, U₀ = %.3e m s⁻¹, Nx × Nz = %d × %d, Di = %.1e, Di T₀ / Δθ = %.1e\n", h, U₀, Nx, Nz, Di, Di * T₀ / Δθ)

# Both are tiny for a laboratory shear layer: the two closures produce the same flow, and they differ in where
# the dissipated energy goes, which is what the diagnostics below show.
#
# ## Model
#
# A linear equation of state (no cabbeling coefficient) and a constant salinity make θ the only tracer, with
# B = g α (θ - T₀).

grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (-Lz/2, Lz/2), topology = (Periodic, Flat, Bounded))

thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = 0.,
                                                   reference_temperature = T₀, reference_salinity = S₀,
                                                   reference_heat_capacity = cₚ₀, gravitational_acceleration = g)

components = thermodynamic_model_components(grid, thermodynamics; viscosity = ν, thermal_diffusivity = κ_θ,
                                            salt_diffusivity = κ_θ, constant_salinity = S₀)

model = NonhydrostaticModel(grid; tracers = :θ, components...)

# ## Initial conditions
#
# The base state is perturbed by a streamfunction ψ = a cos(kx) sech²(z/h), with w = -∂ψ/∂x reaching 10⁻³ U₀.
# `set!` projects the velocity onto a divergence-free field.

a = 1e-3 * U₀ / k

uᵢ(x, z) = U₀ * tanh(z / h) - 2a / h * cos(k * x) * sech(z / h)^2 * tanh(z / h)
wᵢ(x, z) = a * k * sin(k * x) * sech(z / h)^2
θᵢ(x, z) = T₀ + B₀ * tanh(z / h) / (g * α)

set!(model, u = uᵢ, w = wᵢ, θ = θᵢ)

# ## Diagnostics
#
# All the energy diagnostics share one reference height z★, the height each fluid parcel would have in the
# sorted, motionless state. 𝒫 and ℬ contain the internal energy cₚ₀ θ, which is about 10⁶ times larger than their
# changes, so we subtract their initial values cell by cell before integrating.

z★ = reference_height(model)
compute!(z★)

𝒫 = potential_energy(model)
ℬ = background_potential_energy(model, z★)
𝒫₀ = CenterField(grid); set!(𝒫₀, 𝒫)
ℬ₀ = CenterField(grid); set!(ℬ₀, ℬ)

# The volume integrals are written every time step, so the budgets close to the error of the time stepping.
# The flux divergences integrate to zero and are not written. The conversions are
#
#   * w (b̃ - b★), from 𝒜 to 𝒦 (negative while the billow lifts dense fluid);
#   * ε, from 𝒦 to ℬ, by viscous heating;
#   * εₐ, from 𝒜 to ℬ, by diffusion: the irreversible mixing;
#   * 𝒩_ℬ, the spurious change of ℬ by the discrete advection.
#
# The production of 𝒜 by the heating itself is of order Di and is not shown.

∫(term) = Field(Integral(Field(term)))

integrals = (; 𝒦  = ∫(kinetic_energy(model)),
               Δ𝒫 = ∫(𝒫 - 𝒫₀),
               𝒜  = ∫(available_potential_energy(z★)),
               Δℬ = ∫(ℬ - ℬ₀),
               wb = ∫(available_buoyancy_work(model, z★)),
               ε  = ∫(viscous_dissipation(model)),
               εₐ = ∫(ape_dissipation(model, z★)),
               𝒩_ℬ = ∫(spurious_bpe_production(model, z★)))

simulation = Simulation(model; Δt = 0.02h / U₀, stop_time = 150h / U₀)
conjure_time_step_wizard!(simulation, IterationInterval(10), cfl = 0.3)

output_dir = joinpath(@__DIR__, "output")

simulation.output_writers[:integrals] = JLD2Writer(model, integrals; dir = output_dir,
                                                   filename = "kelvin_helmholtz_mixing_integrals",
                                                   schedule = IterationInterval(1), overwrite_files = true)

fields = (; θ = model.tracers.θ, εₐ = Field(ape_dissipation(model, z★)))
simulation.output_writers[:fields] = JLD2Writer(model, fields; dir = output_dir,
                                                filename = "kelvin_helmholtz_mixing_fields",
                                                schedule = TimeInterval(1h / U₀), overwrite_files = true)

progress(sim) = @info @sprintf("t = %5.1f h/U₀, Δt = %s, max|w| = %.1e m s⁻¹, wall time: %s",
                               time(sim) * U₀ / h, prettytime(sim.Δt), maximum(abs, model.velocities.w),
                               prettytime(sim.run_wall_time))
add_callback!(simulation, progress, IterationInterval(1000))

run!(simulation)

# ## Energy pathways

file = joinpath(output_dir, "kelvin_helmholtz_mixing_integrals.jld2")
series = (; (name => vec(interior(FieldTimeSeries(file, string(name)))) for name in keys(integrals))...)
t = FieldTimeSeries(file, "𝒦").times

cumulative(f) = cumsum(vcat(0, (f[2:end] .+ f[1:end-1]) ./ 2 .* diff(t)))

ΔE = series.𝒦 .- series.𝒦[1] .+ series.Δ𝒫
∫ε, ∫εₐ, ∫𝒩_ℬ = cumulative(series.ε), cumulative(series.εₐ), cumulative(series.𝒩_ℬ)
efficiency = ∫εₐ[end] / (∫εₐ[end] + ∫ε[end])

# The billow draws 𝒦 from the shear and stores it as 𝒜 (left), then diffusion converts that 𝒜 into ℬ.
# Viscous dissipation feeds ℬ from the start, through the laminar shear layer; mixing feeds it only once the
# billow has formed (centre). The energy that viscosity removes from 𝒦 reappears as internal energy, so
# 𝒦 + 𝒫 is constant to round-off (right).

τ = h / U₀
energy = (U₀ * h)^2
fig = Figure(size = (1400, 420))

ax = Axis(fig[1, 1]; xlabel = "t U₀ / h", ylabel = "energy / (U₀ h)²", title = "Reservoirs")
lines!(ax, t / τ, (series.𝒦 .- series.𝒦[1]) / energy; label = "Δ𝒦")
lines!(ax, t / τ, series.𝒜 / energy; label = "𝒜")
lines!(ax, t / τ, series.Δℬ / energy; label = "Δℬ")
axislegend(ax; position = :lb)

ax = Axis(fig[1, 2]; xlabel = "t U₀ / h", ylabel = "energy / (U₀ h)²", title = "Sources of Δℬ")
lines!(ax, t / τ, series.Δℬ / energy; label = "Δℬ", color = :black, linewidth = 3)
lines!(ax, t / τ, ∫ε / energy; label = "∫ε dt (heating)")
lines!(ax, t / τ, ∫εₐ / energy; label = "∫εₐ dt (mixing)")
lines!(ax, t / τ, ∫𝒩_ℬ / energy; label = "∫𝒩_ℬ dt (spurious)")
lines!(ax, t / τ, (∫ε .+ ∫εₐ .+ ∫𝒩_ℬ) / energy; label = "sum", color = :black, linestyle = :dash)
axislegend(ax; position = :lt)

ax = Axis(fig[1, 3]; xlabel = "t U₀ / h", ylabel = "energy / (U₀ h)²", yscale = log10, title = "Conservation of 𝒦 + 𝒫")
lines!(ax, t[2:end] / τ, ∫ε[2:end] / energy; label = "∫ε dt")
lines!(ax, t[2:end] / τ, abs.(ΔE[2:end]) / energy .+ eps(); label = "|Δ(𝒦 + 𝒫)|", color = :black)
axislegend(ax; position = :rb)

save(joinpath(output_dir, "kelvin_helmholtz_mixing_energetics.png"), fig)

@printf("max|Δ(𝒦 + 𝒫)| / ∫ε dt = %.1e, cumulative mixing efficiency = %.2f\n", maximum(abs, ΔE) / ∫ε[end], efficiency)
nothing #hide

# ![](kelvin_helmholtz_mixing_energetics.png)
#
# ## The billow
#
# θ and the APE dissipation εₐ. Mixing is concentrated on the braid and on the rim of the billow, where stirring
# has thinned the interface.

θ_series  = FieldTimeSeries(joinpath(output_dir, "kelvin_helmholtz_mixing_fields.jld2"), "θ")
εₐ_series = FieldTimeSeries(joinpath(output_dir, "kelvin_helmholtz_mixing_fields.jld2"), "εₐ")

n = Observable(1)
εₐ_limit = maximum(abs, interior(εₐ_series)) / 4

fig = Figure(size = (1000, 650))
Label(fig[0, :], @lift(@sprintf("t = %.0f h / U₀", θ_series.times[$n] / τ)); fontsize = 20)
ax = Axis(fig[1, 1]; ylabel = "z (m)", aspect = DataAspect())
ylims!(ax, -3h, 3h)
Colorbar(fig[1, 2], heatmap!(ax, @lift(θ_series[$n]); colormap = :thermal); label = "θ (K)")
ax = Axis(fig[2, 1]; xlabel = "x (m)", ylabel = "z (m)", aspect = DataAspect())
ylims!(ax, -3h, 3h)
Colorbar(fig[2, 2], heatmap!(ax, @lift(εₐ_series[$n]); colormap = :balance, colorrange = (-εₐ_limit, εₐ_limit));
         label = "εₐ (m² s⁻³)")

record(fig, joinpath(output_dir, "kelvin_helmholtz_mixing.mp4"), 1:length(θ_series.times); framerate = 16) do i
    n[] = i
end
nothing #hide

# ![](kelvin_helmholtz_mixing.mp4)
