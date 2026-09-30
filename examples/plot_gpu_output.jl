# Offline plots of the output of `examples/gpu/two_layer_cabbeling_gpu.jl`.
#
# Run with `julia --project=examples examples/plot_gpu_output.jl [output_directory]`.
# Saves `two_layer_cabbeling_fields.png` and `two_layer_cabbeling_diagnostics.png` in the output directory.

using Oceananigans
using Oceananigans.Units
using GLMakie

dir = get(ARGS, 1, pwd())
prefix = joinpath(dir, "two_layer_cabbeling")

S = FieldTimeSeries(prefix * "_slices.jld2", "S")
w = FieldTimeSeries(prefix * "_slices.jld2", "w")
S̄ = FieldTimeSeries(prefix * "_profiles.jld2", "S")
Nt = length(S.times)

# Slices of S and w at the final time, and the horizontally averaged S over time
fig = Figure(size = (1100, 500))
axS = Axis(fig[1, 1]; title = "S at t = " * prettytime(S.times[Nt]), xlabel = "x (m)", ylabel = "z (m)")
axw = Axis(fig[1, 3]; title = "w at t = " * prettytime(w.times[Nt]), xlabel = "x (m)")
axH = Axis(fig[1, 5]; title = "Horizontally averaged S", xlabel = "t (min)", ylabel = "z (m)")
Colorbar(fig[1, 2], heatmap!(axS, S[Nt]))
wmax = maximum(abs, interior(w[Nt]))
Colorbar(fig[1, 4], heatmap!(axw, w[Nt]; colormap = :balance, colorrange = (-wmax, wmax)))
Colorbar(fig[1, 6], heatmap!(axH, S̄.times ./ minute, znodes(S̄.grid, Center()), interior(S̄)[1, 1, :, :]'))
save(prefix * "_fields.png", fig)

# Global diagnostics: conservation of ∫θ and ∫S, and the kinetic energy
series(name) = interior(FieldTimeSeries(prefix * "_diagnostics.jld2", name))[1, 1, 1, :]
t = FieldTimeSeries(prefix * "_diagnostics.jld2", "∫KE").times ./ minute
∫θ, ∫S, ∫KE = series("∫θ"), series("∫S"), series("∫KE")

fig = Figure(size = (1100, 350))
lines(fig[1, 1], t, ∫θ ./ ∫θ[1] .- 1; axis = (; title = "∫θ dV relative drift", xlabel = "t (min)"))
lines(fig[1, 2], t, ∫S ./ ∫S[1] .- 1; axis = (; title = "∫S dV relative drift", xlabel = "t (min)"))
lines(fig[1, 3], t, ∫KE;              axis = (; title = "∫½|u|² dV (m⁵ s⁻²)", xlabel = "t (min)"))
save(prefix * "_diagnostics.png", fig)

@info "Saved $(prefix)_fields.png and $(prefix)_diagnostics.png"
