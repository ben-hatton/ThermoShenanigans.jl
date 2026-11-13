using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using ThermoShenanigans: flux_contractionᶜᶜᶜ, thermodynamic_functionᶜᶜᶜ, θ_value, HeightIndependent

@testset "Irreversible production" begin
    grid = test_grid()

    @testset "ε is the discrete kinetic energy sink" begin
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        u, v, w = model.velocities
        args = (model.closure, model.closure_fields, model.clock, merge(model.velocities, model.tracers), model.buoyancy)

        # ∫ uᵢ ∂ⱼτᵢⱼ dV over all velocity points (wall-normal velocities vanish at walls)
        sink = integral(*, u, KernelFunctionOperation{Face, Center, Center}(∂ⱼ_τ₁ⱼ, grid, args...)) +
               integral(*, v, KernelFunctionOperation{Center, Face, Center}(∂ⱼ_τ₂ⱼ, grid, args...)) +
               integral(*, w, KernelFunctionOperation{Center, Center, Face}(∂ⱼ_τ₃ⱼ, grid, args...))

        ε = viscous_dissipation(model)
        @test integral(ε) ≈ sink rtol = 1e-12
        @test all(≥(0), computed(ε))
    end

    # The coupled thermodynamics is covered by the semi-discrete energy identity in test_energy_conservation.jl
    @testset "Closure and σ_θ conserve 𝒦 + 𝒫 [$(summary(thermodynamics))]" for thermodynamics in (cabbeling,)
        model = component_model(grid, thermodynamics)
        set_random_state!(model)
        σ_θ = KernelFunctionOperation{Center, Center, Center}(model.forcing.θ, grid, model.clock, merge(model.velocities, model.tracers))
        π, m, σ_θ, ∇_dot_j_θ, ∇_dot_j_S = map(field, (thermodynamic_operation(model, exner), thermodynamic_operation(model, chemical_potential_analogue), σ_θ, flux_divergence(model, 1), flux_divergence(model, 2)))

        # ∫ (π ∂ₜθ + m ∂ₜS) dV from the closure and σ_θ must equal ∫ ε dV, the kinetic energy lost to viscosity
        potential_energy_tendency = integral((π, σ_θ, ∇_dot_j_θ, m, ∇_dot_j_S) -> π * (σ_θ - ∇_dot_j_θ) - m * ∇_dot_j_S,
                                          π, σ_θ, ∇_dot_j_θ, m, ∇_dot_j_S)
        dissipation = integral(viscous_dissipation(model))
        scale = integral((a, b) -> abs(a * b), π, ∇_dot_j_θ)

        @test abs(potential_energy_tendency - dissipation) < 1e-12 * scale
        @test dissipation > 0
    end

    @testset "σ_η ≥ 0 [$(summary(thermodynamics))]" for thermodynamics in (cabbeling, coupled)
        model = component_model(grid, thermodynamics)
        set_random_state!(model)
        σ_η = computed(entropy_production(model))
        @test all(≥(0), σ_η) && any(>(0), σ_η)

        model = component_model(test_grid(size = (1, 1, 16), extent = (1, 1, 1000)), thermodynamics)
        set_resting_equilibrium!(model, thermodynamics)
        @test maximum(abs, computed(entropy_production(model))) < 1e-20
    end

    @testset "σ_b = b_θ σ_θ + σ_b_nonlin [$(summary(thermodynamics))]" for thermodynamics in (linear, cabbeling)
        model = component_model(grid, thermodynamics)
        set_random_state!(model)
        C, buoyancy_formulation, closure = merge(model.velocities, model.tracers), model.buoyancy.formulation, model.closure[2]
        σ_b_nonlin = computed(nonlinear_buoyancy_production(model))

        # σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S = g γ j_θ·∇θ for the cabbeling equation of state
        zero_value(θ, S, z, thermodynamics) = zero(θ)
        j_θ_dot_∇θ = KernelFunctionOperation{Center, Center, Center}(flux_contractionᶜᶜᶜ, grid, θ_value, zero_value, closure, buoyancy_formulation, C)
        @test σ_b_nonlin ≈ thermodynamics.gravitational_acceleration * thermodynamics.cabbeling_coefficient .* computed(j_θ_dot_∇θ) rtol = 1e-12 atol = 1e-20

        b_θ = computed(KernelFunctionOperation{Center, Center, Center}(thermodynamic_functionᶜᶜᶜ, grid, HeightIndependent(∂b∂θ), buoyancy_formulation, C))
        @test computed(buoyancy_production(model)) ≈ b_θ .* computed(θ_production(model)) .+ σ_b_nonlin
    end

    @testset "consistent = false is the standard closure" begin
        components = thermodynamic_model_components(grid, cabbeling; viscosity = 1e-3, thermal_diffusivity = 1e-3,
                                                    salt_diffusivity = 1e-4, consistent = false)
        @test components.forcing == NamedTuple() && components.closure isa ScalarDiffusivity
        @test components.closure.ν == 1e-3 && components.closure.κ == (θ = 1e-3, S = 1e-4)
    end
end
