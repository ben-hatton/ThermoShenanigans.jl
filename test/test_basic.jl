# A quick core set on one model: the first and second laws, and a time step.
# The full suite (TEST_GROUP=all) repeats these for more thermodynamics, grids and architectures.
@testset "Basic model" begin
    model = component_model(test_grid(), cabbeling)
    set_random_state!(model)

    @testset "Semi-discrete energy identity" begin
        dEdt, round_off, conversions = energy_tendency(model)
        @info "dE/dt = $dEdt for conversions of $conversions [basic]"
        @test abs(dEdt) < 1e-13 * round_off
    end

    @testset "σ_η ≥ 0" begin
        σ_η = computed(thermodynamic_diagnostics(model).σ_η)
        @test all(≥(0), σ_η) && any(>(0), σ_η)
    end

    @testset "𝒜 + 𝓑 = 𝒫 and 𝒜 ≥ 0" begin
        e = energetics(model)
        𝒜, 𝓑 = computed(e.𝒜), computed(e.𝓑)
        @test 𝒜 .+ 𝓑 ≈ computed(thermodynamic_diagnostics(model).𝒫) rtol = 1e-14
        @test all(≥(0), 𝒜)
    end

    @testset "Time step" begin
        time_step!(model, 1e-3)
        @test all(isfinite, interior(model.tracers.θ)) && all(isfinite, interior(model.tracers.S))
    end
end
