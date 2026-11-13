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
        σ_η = computed(entropy_production(model))
        @test all(≥(0), σ_η) && any(>(0), σ_η)
    end

    @testset "𝒜 + ℬ = 𝒫 and 𝒜 ≥ 0" begin
        z★ = reference_height(model)
        𝒜, ℬ = computed(available_potential_energy(z★)), computed(background_potential_energy(model, z★))
        @test 𝒜 .+ ℬ ≈ computed(thermodynamic_operation(model, potential_energy)) rtol = 1e-14
        @test all(≥(0), 𝒜)
    end

    @testset "∫ R★ dV = 0" begin
        R★ = computed(reference_tendency_work(model, reference_height(model)))
        V = cell_volumes(CenterField(model.grid))
        @test abs(sum(R★ .* V)) < 1e-13 * sum(abs.(R★) .* V)
    end

    @testset "∫ (𝒩 + 𝒩_ℬ) dV = 0" begin
        # The spurious mixing by the discrete advection converts ℬ into 𝒜 with no net change of 𝒫
        _, round_off, _ = energy_tendency(model)
        z★ = reference_height(model)
        𝒩, 𝒩_ℬ = computed(spurious_ape_production(model, z★)), computed(spurious_bpe_production(model, z★))
        V = cell_volumes(CenterField(model.grid))
        @test abs(sum((𝒩 .+ 𝒩_ℬ) .* V)) < 1e-13 * round_off
    end

    @testset "Time step" begin
        time_step!(model, 1e-3)
        @test all(isfinite, interior(model.tracers.θ)) && all(isfinite, interior(model.tracers.S))
    end
end
