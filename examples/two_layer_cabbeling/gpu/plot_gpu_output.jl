# # Plots of the 3D two-layer cabbeling output
#
# Offline plots of the output of `two_layer_cabbeling_gpu.jl`.
#
# Run with `julia --project=examples examples/two_layer_cabbeling/gpu/plot_gpu_output.jl [output_directory]`.
# Saves `two_layer_cabbeling_fields.png` and `two_layer_cabbeling_integrals.png` in the output directory.

using Oceananigans
using Oceananigans.Units
using CairoMakie

dir = get(ARGS, 1, pwd())
prefix = joinpath(dir, "two_layer_cabbeling")

# ## Fields
#
# Slices of S and w at the final time, and the horizontally averaged S against time.

S = FieldTimeSeries(prefix * "_slices.jld2", "S")
w = FieldTimeSeries(prefix * "_slices.jld2", "w")
S̄ = FieldTimeSeries(prefix * "_profiles.jld2", "S")
Nt = length(S.times)
wlim = maximum(abs, interior(w[Nt]))

fig = Figure(size = (1100, 500))
axS = Axis(fig[1, 1]; title = "S at t = " * prettytime(S.times[Nt]), xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[1, 3]; title = "w at t = " * prettytime(w.times[Nt]), xlabel = "x (m)")
axS̄ = Axis(fig[1, 5]; title = "Horizontally averaged S", xlabel = "t (min)", ylabel = "z (m)")
Colorbar(fig[1, 2], heatmap!(axS, S[Nt]))
Colorbar(fig[1, 4], heatmap!(axw, w[Nt]; colormap = :balance, colorrange = (-wlim, wlim)))
Colorbar(fig[1, 6], heatmap!(axS̄, S̄.times ./ minute, znodes(S̄.grid, Center()), interior(S̄, 1, 1, :, :)'))
save(prefix * "_fields.png", fig)

# ## Volume integrals
#
# The drift of ∫θ dV and ∫S dV, and the kinetic energy.

integrals = prefix * "_integrals.jld2"
series(name) = interior(FieldTimeSeries(integrals, name), 1, 1, 1, :)
t = FieldTimeSeries(integrals, "∫KE").times ./ minute
∫θ, ∫S = series("∫θ"), series("∫S")

fig = Figure(size = (1100, 350))
lines(fig[1, 1], t, ∫θ ./ ∫θ[1] .- 1; axis = (; title = "∫θ dV relative drift", xlabel = "t (min)"))
lines(fig[1, 2], t, ∫S ./ ∫S[1] .- 1; axis = (; title = "∫S dV relative drift", xlabel = "t (min)"))
lines(fig[1, 3], t, series("∫KE"); axis = (; title = "∫½|u|² dV (m⁵ s⁻²)", xlabel = "t (min)"))
save(prefix * "_integrals.png", fig)
