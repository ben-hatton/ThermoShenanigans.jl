using Oceananigans.TurbulenceClosures: ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∂ⱼ_τ₃ⱼ
using ThermoShenanigans: flux_contractionᶜᶜᶜ, thermodynamic_functionᶜᶜᶜ, production_parameters, center_operation,
                         model_tendencies, wall_conditions, θ_tendencyᶜᶜᶜ, salt_tendencyᶜᶜᶜ,
                         potential_energy_tendency

function test_production()
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
            integrated_pe_tendency = integral((π, σ_θ, ∇_dot_j_θ, m, ∇_dot_j_S) -> π * (σ_θ - ∇_dot_j_θ) - m * ∇_dot_j_S,
                                              π, σ_θ, ∇_dot_j_θ, m, ∇_dot_j_S)
            dissipation = integral(viscous_dissipation(model))
            scale = integral((a, b) -> abs(a * b), π, ∇_dot_j_θ)

            @test abs(integrated_pe_tendency - dissipation) < 1e-12 * scale
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

        @testset "σ_b = b_θ σ_θ + σ_b_nonlin [$label, $(summary(thermodynamics))]" for (label, build) in
                (("consistent", component_model), ("Fickian", fickian_model)), thermodynamics in (linear, cabbeling)
            model = build(grid, thermodynamics)
            set_random_state!(model)
            C, buoyancy_formulation = merge(model.velocities, model.tracers), model.buoyancy.formulation
            closure = production_parameters(model).diffusivity
            σ_b_nonlin = computed(nonlinear_buoyancy_production(model))

            # σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S = g γ j_θ·∇θ for the cabbeling equation of state, which checks the
            # face fluxes of either closure in flux_contractionᶜᶜᶜ
            θ_itself(θ, S, z, thermodynamics) = θ
            zero_value(θ, S, z, thermodynamics) = zero(θ)
            j_θ_dot_∇θ = KernelFunctionOperation{Center, Center, Center}(flux_contractionᶜᶜᶜ, grid, θ_itself, zero_value, closure, buoyancy_formulation, C)
            @test σ_b_nonlin ≈ thermodynamics.gravitational_acceleration * thermodynamics.cabbeling_coefficient .* computed(j_θ_dot_∇θ) rtol = 1e-12 atol = 1e-20

            b_θ = computed(KernelFunctionOperation{Center, Center, Center}(thermodynamic_functionᶜᶜᶜ, grid, ∂b∂θ, buoyancy_formulation, C))
            @test computed(buoyancy_production(model)) ≈ b_θ .* computed(θ_production(model)) .+ σ_b_nonlin
            closure isa ScalarDiffusivity && @test computed(buoyancy_production(model)) == σ_b_nonlin
        end

        @testset "Fickian model: zero θ production, and refusals" begin
            model = fickian_model(grid, cabbeling)
            set_random_state!(model)
            @test all(iszero, computed(θ_production(model)))
            @test all(iszero, computed(heating_buoyancy_production(model)))

            test_refused(entropy_production, () -> entropy_production(model))
            test_refused(potential_energy_conversion, () -> potential_energy_conversion(model))
            test_refused(thermodynamic_operation, () -> thermodynamic_operation(model, potential_energy))
            test_refused(potential_energy, () -> potential_energy(model))
            test_refused(internal_energy, () -> internal_energy(model))
            test_refused(potential_energy_tendency, () -> potential_energy_tendency(model))
        end

        #=
        The energy reservoirs 𝒫, ℐ and 𝒢 of `diagnostics.jl` and `bpe_budget.jl`. `shifted` has z₀ = -40, so that
        g₀ z₀ ≠ 0 and z₀ enters 𝒢 = - b (z - z₀) as the origin of z.
        =#
        @testset "Energy reservoirs 𝒫 = ℐ + 𝒢 + g₀ z₀ [$label]" for (label, thermodynamics) in (("cabbeling", cabbeling),
                                                                                                 ("z₀ = -40", shifted))
            z = reshape(znodes(grid, Center()), 1, 1, :)
            g₀, z₀ = gravitational_acceleration(thermodynamics), reference_pressure_height(thermodynamics)

            model = component_model(grid, thermodynamics)
            set_random_state!(model)
            𝒫, ℐ, 𝒢 = map(computed, (potential_energy(model), internal_energy(model), gravitational_energy(model)))
            b = computed(KernelFunctionOperation{Center, Center, Center}(thermodynamic_functionᶜᶜᶜ, grid, buoyancy,
                                                                         model.buoyancy.formulation, model.tracers))

            @test 𝒫 == computed(thermodynamic_operation(model, potential_energy))
            @test ℐ == computed(thermodynamic_operation(model, potential_enthalpy))
            @test 𝒢 ≈ .- b .* (z .- z₀) rtol = 1e-14
            @test ℐ .+ 𝒢 .+ g₀ * z₀ ≈ 𝒫 rtol = 1e-14

            # A Fickian model has 𝒢, with the same b and z₀
            fickian = fickian_model(grid, thermodynamics)
            set!(fickian; map(field -> Array(interior(field)), model.tracers)...)
            @test computed(gravitational_energy(fickian)) == 𝒢
        end

        #=
        ∂ₜθ = G_θ + F_θ and ∂ₜS = G_S + F_S, with F the tendency of the `FluxBoundaryCondition`s, against the Gⁿ to which
        `compute_flux_bc_tendencies!` adds F: the only direct test of F. A Fickian closure's Value and Gradient wall
        fluxes are in G already, and not in F.
        =#
        @testset "Prognostic tendencies with the wall-flux tendency F [$label]" for (label, build) in
                (("Flux conditions", flux_bc_model), ("Fickian wall conditions", fickian_wall_model))
            model = build(grid)
            set_random_state!(model)
            tendencies, walls = model_tendencies(model), wall_conditions(model)
            ∂ₜθ = computed(center_operation(θ_tendencyᶜᶜᶜ, grid, tendencies, walls))
            ∂ₜS = computed(center_operation(salt_tendencyᶜᶜᶜ, grid, model.buoyancy.formulation, tendencies, walls))

            compute_flux_bc_tendencies!(model)
            G = model.timestepper.Gⁿ
            @test ∂ₜθ ≈ Array(interior(G.θ)) rtol = 1e-14
            @test ∂ₜS ≈ Array(interior(G.S)) rtol = 1e-14
        end

        @testset "consistent = false is the standard closure" begin
            components = thermodynamic_model_components(grid, cabbeling; viscosity = 1e-3, thermal_diffusivity = 1e-3,
                                                        salt_diffusivity = 1e-4, consistent = false)
            @test components.forcing == NamedTuple() && components.closure isa ScalarDiffusivity
            @test components.closure.ν == 1e-3 && components.closure.κ == (θ = 1e-3, S = 1e-4)
        end
    end
end
