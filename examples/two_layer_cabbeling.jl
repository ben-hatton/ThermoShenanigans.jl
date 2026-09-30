# Two-layer cabbeling instability in 2D at low resolution, after experiment IV of
# Bisits, Zika & Sohail (2025, JFM 1011 A17): cold, fresh water over warm, salty water,
# statically stable but denser when mixed.
#
# Run with `julia --project=examples examples/two_layer_cabbeling.jl`.

using Oceananigans
using Oceananigans.Units
using ThermoShenanigans
using GLMakie
using Statistics: mean

live = true # show the flow while the simulation runs
gpu  = false # run on an NVIDIA GPU

if gpu
    using CUDA
end

arch = gpu ? GPU() : CPU()

#####
##### Thermodynamics, linearised about the deep layer (θ absolute, S in kg/kg)
#####

θ_deep, S_deep       = 273.15 + 0.5, 0.0347     # Θ = 0.5 °C,  S = 34.7 g/kg
θ_shallow, S_shallow = 273.15 - 1.5, 0.03458    # Θ = -1.5 °C, S = 34.58 g/kg

g, β, γ = 9.80665, 0.78, 1e-5
Δb = g * 0.0046 / 1027 # shallow - deep, from Δρ = -0.0046 kg m⁻³

# Choose α so that b(shallow) - b(deep) = Δb, with b = g [α θ′ + ½ γ θ′² - β S′]
Δθ, ΔS = θ_shallow - θ_deep, S_shallow - S_deep
α = (Δb / g - γ * Δθ^2 / 2 + β * ΔS) / Δθ

thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = α, haline_contraction = β, cabbeling_coefficient = γ,
                                           reference_temperature = θ_deep, reference_salinity = S_deep,
                                           reference_heat_capacity = 3991.86795711963, gravitational_acceleration = g)

b_mix = buoyancy((θ_shallow + θ_deep) / 2, (S_shallow + S_deep) / 2, thermo)
@info "α = $α; b(shallow) - b(deep) = $(buoyancy(θ_shallow, S_shallow, thermo)) (> 0: stable); b(mix) - b(deep) = $b_mix (< 0: cabbeling-unstable)"

#####
##### Model
#####

grid = RectilinearGrid(arch; size = (32, 256), x = (-0.035, 0.035), z = (-1, 0), topology = (Periodic, Flat, Bounded))

closure = (ScalarDiffusivity(ν = 1e-6), ThermodynamicDiffusivity(thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7))
model = NonhydrostaticModel(grid; tracers = (:θ, :S), buoyancy = ThermodynamicBuoyancy(grid, thermo), closure)

θᵢ(x, z) = z > -0.5 ? θ_shallow : θ_deep
Sᵢ(x, z) = (z > -0.5 ? S_shallow : S_deep) + 2e-7 * randn() * exp(-((z + 0.5) / 0.02)^2)
set!(model, θ = θᵢ, S = Sᵢ)

θ, S = model.tracers
w = model.velocities.w
T = Field(KernelFunctionOperation{Center, Center, Center}(ThermoShenanigans.height_dependent_functionᶜᶜᶜ, grid,
                                                          in_situ_temperature, model.buoyancy.formulation, model.tracers))

#####
##### Simulation, with progress, live plotting and output
#####

simulation = Simulation(model; Δt = 0.5, stop_time = 10minutes)

∫θ₀, ∫S₀ = sum(interior(θ)), sum(interior(S))

progress(sim) = @info string("iteration ", iteration(sim), ", t = ", prettytime(sim), ", max|w| = ", maximum(abs, w),
                             ", Σθ drift = ", sum(interior(θ)) / ∫θ₀ - 1, ", ΣS drift = ", sum(interior(S)) / ∫S₀ - 1)
add_callback!(simulation, progress, IterationInterval(100))

if live
    θₒ, Sₒ, Tₒ = Observable(θ), Observable(S), Observable(compute!(T))
    title = Observable("t = 0")
    fig = Figure(size = (900, 600))
    for (n, (field, label)) in enumerate(((θₒ, "θ (K)"), (Sₒ, "S (kg/kg)"), (Tₒ, "T (K)")))
        ax = Axis(fig[1, 2n - 1]; title = label, xlabel = "x (m)", ylabel = "z (m)")
        hm = heatmap!(ax, field)
        Colorbar(fig[1, 2n], hm)
    end
    Label(fig[0, :], title)
    display(fig)

    function update_plot(sim)
        compute!(T)
        notify(θₒ); notify(Sₒ); notify(Tₒ)
        title[] = "t = " * prettytime(sim)
        yield() # let GLMakie render
    end
    add_callback!(simulation, update_plot, IterationInterval(20))
end

filename = joinpath(@__DIR__, "two_layer_cabbeling")
simulation.output_writers[:fields] = JLD2Writer(model, (; θ, S, T, w); filename, schedule = TimeInterval(15), overwrite_files = true)

run!(simulation)

#####
##### Plots from the saved output
#####

S_ts = FieldTimeSeries(filename * ".jld2", "S")
w_ts = FieldTimeSeries(filename * ".jld2", "w")
times, Nt = S_ts.times, length(S_ts.times)

n = Observable(1)
fig = Figure(size = (700, 600))
axS = Axis(fig[1, 1]; title = "S (kg/kg)", xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[1, 3]; title = "w (m/s)", xlabel = "x (m)")
Colorbar(fig[1, 2], heatmap!(axS, @lift(S_ts[$n]); colorrange = (S_shallow, S_deep)))
wmax = maximum(abs, interior(w_ts))
Colorbar(fig[1, 4], heatmap!(axw, @lift(w_ts[$n]); colormap = :balance, colorrange = (-wmax, wmax)))
Label(fig[0, :], @lift("t = " * prettytime(times[$n])))
record(fig, filename * ".mp4", 1:Nt; framerate = 8) do i
    n[] = i
end

# Hovmöller plot of the horizontally averaged salinity
S̄ = dropdims(mean(interior(S_ts), dims = (1, 2)), dims = (1, 2)) # (z, t)
z = znodes(grid, Center())
fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1]; xlabel = "t (min)", ylabel = "z (m)", title = "Horizontally averaged S (kg/kg)")
Colorbar(fig[1, 2], heatmap!(ax, times ./ 60, z, S̄'))
save(filename * "_hovmoller.png", fig)

#####
##### Variant: θ only, with salinity held constant
#####

model = NonhydrostaticModel(grid; tracers = :θ, closure,
                            buoyancy = ThermodynamicBuoyancy(grid, thermo; constant_salinity = S_deep))
set!(model, θ = (x, z) -> θᵢ(x, z) + 1e-3 * randn())
for _ in 1:10
    time_step!(model, 0.5)
end
@info "Constant-salinity model: max|w| = $(maximum(abs, model.velocities.w)) after 10 steps"
