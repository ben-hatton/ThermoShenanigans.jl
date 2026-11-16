using Oceananigans.Forcings: MultipleForcings
using ThermoShenanigans: production_parameters, dissipationᶜᶜᶜ, θ_productionᶜᶜᶜ

function test_model_components()
    @testset "Model components" begin
        grid = test_grid()
        coefficients = (viscosity = 1e-3, thermal_diffusivity = 1e-3, salt_diffusivity = 1e-4)

        @testset "σ_θ and the diagnostics evaluate ε from the same arguments" begin
            model = component_model(grid, cabbeling)
            set_random_state!(model)
            σ_θ = model.forcing.θ
            diagnostic_parameters = production_parameters(model)

            # The arguments of ε after (i, j, k, grid, clock, fields)
            for name in (:viscous_closure, :closure_fields, :buoyancy)
                @test diagnostic_parameters[name] === σ_θ.parameters[name]
            end
            # The diffusivity of the model also records its tracer names, which select the θ and S fluxes
            for name in (:thermal_diffusivity, :salt_diffusivity)
                @test getproperty(diagnostic_parameters.diffusivity, name) === getproperty(σ_θ.parameters.diffusivity, name)
            end

            fields = merge(model.velocities, model.tracers)
            ε_of_σ_θ = KernelFunctionOperation{Center, Center, Center}(dissipationᶜᶜᶜ, grid, model.clock, fields, σ_θ.parameters)
            forcing_σ_θ = KernelFunctionOperation{Center, Center, Center}(σ_θ, grid, model.clock, fields)
            @test computed(viscous_dissipation(model)) == computed(ε_of_σ_θ)
            @test computed(θ_production(model)) == computed(forcing_σ_θ)
        end

        @testset "A user θ forcing is combined with σ_θ" begin
            heating = θ_forcing_by_heating(grid, cabbeling)
            u_forcing(x, y, z, t) = 1e-3 * z

            components = thermodynamic_model_components(grid, cabbeling; coefficients..., forcing = (; u = u_forcing, θ = heating))
            @test keys(components.forcing) == (:u, :θ)
            @test components.forcing.u === u_forcing
            @test components.forcing.θ[1] === heating && components.forcing.θ[2].func === θ_productionᶜᶜᶜ

            # Tuples and MultipleForcings of user θ forcings are followed by σ_θ
            for θ_forcing in ((heating, heating), MultipleForcings(heating, heating))
                components = thermodynamic_model_components(grid, cabbeling; coefficients..., forcing = (; θ = θ_forcing))
                @test length(components.forcing.θ) == 3 && components.forcing.θ[3].func === θ_productionᶜᶜᶜ
            end

            # A model type of its own, for its θ forcing
            heated, unheated = component_model(grid, cabbeling; forcing = (; θ = heating)), component_model(grid, cabbeling)
            set_random_state!(unheated)
            set!(heated; map(field -> Array(interior(field)), merge(unheated.velocities, unheated.tracers))...)
            @test heated.forcing.θ isa MultipleForcings{2}

            F_θ = computed(KernelFunctionOperation{Center, Center, Center}(heating, grid, heated.clock,
                                                                           merge(heated.velocities, heated.tracers)))
            G_θ = θ_tendency(unheated)
            # To the round-off of the sums of the tendency terms, which add F_θ in a different order
            @test θ_tendency(heated) ≈ G_θ .+ F_θ rtol = 1e-12 atol = 1e-12 * maximum(abs, G_θ)
            @test maximum(abs, F_θ) > 1e-3 * maximum(abs, G_θ)

            # Without σ_θ, the user forcing is passed through
            components = thermodynamic_model_components(grid, cabbeling; coefficients..., consistent = false, forcing = (; θ = heating))
            @test components.forcing === (; θ = heating)
        end

        @testset "A model built by hand with θ_production_forcing" begin
            closure = (ScalarDiffusivity(ν = coefficients.viscosity),
                       ThermodynamicDiffusivity(thermal_diffusivity = coefficients.thermal_diffusivity,
                                                salt_diffusivity = coefficients.salt_diffusivity))
            buoyancy = ThermodynamicBuoyancy(grid, cabbeling)
            σ_θ = θ_production_forcing(closure, buoyancy)
            components = thermodynamic_model_components(grid, cabbeling; coefficients...)
            @test σ_θ.func === components.forcing.θ.func && σ_θ.parameters === components.forcing.θ.parameters

            # θ_production_forcing has no method for a Fickian closure, which has no θ source
            @test_throws MethodError θ_production_forcing(ScalarDiffusivity(ν = 1e-3, κ = (θ = 1e-3, S = 1e-4)), buoyancy)

            # Only constant κ has production parameters: a Function κ reads the clock, which the diagnostics do not pass
            κ_θ(x, y, z, t) = 1e-3
            for κ in ((θ = κ_θ, S = 1e-4), κ_θ)
                @test_throws MethodError production_parameters(ScalarDiffusivity(ν = 1e-3; κ), buoyancy)
            end

            # A model type of its own, with σ_θ before the user θ forcing, as in the docstring of θ_production_forcing
            heating = θ_forcing_by_heating(grid, cabbeling)
            by_hand = NonhydrostaticModel(grid; tracers = (:θ, :S), momentum_advection = Centered(), buoyancy, closure,
                                          forcing = (; θ = (σ_θ, heating)))
            heated = component_model(grid, cabbeling; forcing = (; θ = heating))
            set_random_state!(heated)
            # The same velocities, which a second projection onto zero divergence would change at round-off
            set!(by_hand; enforce_incompressibility = false,
                 map(field -> Array(interior(field)), merge(heated.velocities, heated.tracers))...)
            @test by_hand.forcing.θ.forcings[1] === σ_θ
            # σ_θ + F_θ and F_θ + σ_θ are equal in floating point
            @test θ_tendency(by_hand) == θ_tendency(heated)
        end
    end
end
