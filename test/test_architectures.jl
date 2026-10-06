# A deterministic state, so that models on different architectures can be compared
function set_smooth_state!(model)
    U(x, y, z) = 1e-2 * sin(2π * x) * cos(π * y) * cos(π * z)
    θᵢ(x, y, z) = 283 + sin(2π * x) * sin(π * y) + z
    Sᵢ(x, y, z) = 0.035 + 1e-3 * cos(2π * x) * cos(π * z)
    haskey(model.tracers, :S) ? set!(model, u = U, v = U, θ = θᵢ, S = Sᵢ) : set!(model, u = U, v = U, θ = θᵢ)
    return nothing
end

function stepped_model(arch, FT = Float64; kw...)
    grid = RectilinearGrid(arch, FT, size = (4, 4, 4), extent = (1, 1, 1), topology = (Bounded, Bounded, Bounded))
    model = component_model(grid, CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...); kw...)
    set_smooth_state!(model)
    time_step!(model, 1e-3)
    return model
end

# Every diagnostic on Float64 models and GPUs (one face component each, since all three come from the same code),
# and a few on the CPU in Float32, which keeps compilation short
@testset "Architectures [$(summary(arch)), $FT]" for arch in archs, FT in (Float64, Float32)
    model = stepped_model(arch, FT)
    @test model.closure[2] isa ThermodynamicDiffusivity{FT}
    full = FT == Float64 || arch isa GPU

    d = thermodynamic_diagnostics(model)
    centred = full ? (:Σ, :T, :π, :Σ_S, :μ, :ε, :σ_θ, :σ_b, :σ_b_nonlin, :σ_η, :conversion, :wb) : (:T, :σ_θ, :σ_η)
    for name in centred
        @test all(isfinite, computed(d[name]))
    end
    for name in (full ? (:j_θ, :j_S, :j_b) : (:j_θ,))
        @test all(isfinite, computed(d[name].z))
    end

    if arch isa GPU
        reference = stepped_model(CPU(), FT)
        for name in (:θ, :S)
            @test Array(interior(model.tracers[name])) ≈ interior(reference.tracers[name])
        end
    end

    if full
        model = stepped_model(arch, FT; tracers = :θ, constant_salinity = 0.035)
        d = thermodynamic_diagnostics(model)
        @test all(isfinite, computed(d.σ_θ)) && all(isfinite, computed(d.σ_η))
    end
end
