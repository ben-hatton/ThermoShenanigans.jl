using Oceananigans.TimeSteppers: update_state!, compute_tendencies!, compute_flux_bc_tendencies!

function test_diffusivity()
    @testset "ThermodynamicDiffusivity" begin
        @testset "Isothermal derivatives [$(summary(thermodynamics))]" for thermodynamics in (cabbeling, coupled)
            for (θ, S, z) in states
                T_isotherm = temperature(θ, S, z, thermodynamics)
                @test - ∂T∂S(θ, S, z, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics) ≈ derivative(S -> isothermal_θ(T_isotherm, S, z, thermodynamics; θ), S) atol=1e-12
                @test isothermal_∂b∂S(θ, S, z, thermodynamics) ≈ derivative(S -> buoyancy(isothermal_θ(T_isotherm, S, z, thermodynamics; θ), S, z, thermodynamics), S)
                @test isothermal_∂μ∂S(θ, S, z, thermodynamics) ≈ derivative(S -> chemical_potential(isothermal_θ(T_isotherm, S, z, thermodynamics; θ), S, z, thermodynamics), S)
                # Maxwell relation (∂μ/∂z)_T = - (∂b/∂S)_T
                @test derivative(z -> chemical_potential(isothermal_θ(T_isotherm, S, z, thermodynamics; θ), S, z, thermodynamics), z) ≈ - isothermal_∂b∂S(θ, S, z, thermodynamics)
            end

            R_S, g, β = cabbeling.saline_entropy_constant, cabbeling.gravitational_acceleration, cabbeling.haline_contraction
            θ, S, z = states[2]
            T = temperature(θ, S, z, cabbeling)
            @test barodiffusion_coefficient(θ, S, z, cabbeling) ≈ g * β * S / (R_S * T)
        end

        @testset "θ diffusivity is exactly κ_θ [$(summary(thermodynamics))]" for thermodynamics in (linear, cabbeling, coupled)
            for (θ, S, z) in states
                # π j_θ = - κ_θ cₚ ∇T + ..., and since cₚ / π = 1 / T_θ the ∇θ part of j_θ is - κ_θ ∇θ
                @test θ_flux_prefactor(θ, S, z, thermodynamics) ≈ heat_capacity(θ, S, z, thermodynamics) / exner(θ, S, z, thermodynamics)
                @test θ_flux_prefactor(θ, S, z, thermodynamics) * ∂T∂θ(θ, S, z, thermodynamics) ≈ 1
            end
        end

        @testset "Resting equilibrium has zero fluxes [γ = $(thermodynamics.cabbeling_coefficient)]" for thermodynamics in (linear, cabbeling)
            model = component_model(test_grid(size = (1, 1, 16), extent = (1, 1, 1000)), thermodynamics)
            S, Γ₁, Δz = set_resting_equilibrium!(model, thermodynamics)

            heat_scale = 1e-3 * 1 / Δz        # κ_θ × 1 K / Δz
            salt_scale = 1e-4 * abs(Γ₁)       # κ_S × Γ_S
            @test maximum(abs, z_flux(model, 1)) < 1e-10 * heat_scale
            @test maximum(abs, z_flux(model, 2)) < 1e-10 * salt_scale
            @test S[end] < S[1] # barodiffusion: saltier at depth
        end

        grid = test_grid(size = (4, 4, 8), extent = (1, 1, 100)) # min(Δx, Δy, Δz) = 0.25

        @testset "g → 0 reduces to ScalarDiffusivity" begin
            thermo = CabbelingBoussinesqThermodynamics(; parameters..., gravitational_acceleration = 0)
            model = component_model(grid, thermo)
            reference = component_model(grid, thermo; consistent = false)
            set!(model, θ = (x, y, z) -> 283 + rand(), S = (x, y, z) -> 0.035 + 1e-3 * rand())
            set!(reference, θ = model.tracers.θ, S = model.tracers.S)

            for id in 1:2
                @test computed(flux_divergence(model, id)) ≈ computed(flux_divergence(reference, id)) rtol=1e-12
            end
        end

        @testset "Fickian fluxes are the model's own fluxes" begin
            # At rest and with no forcing, the tendencies are - ∇·j of the closure's fluxes in every cell, wall cells
            # included, with the default no-flux walls
            θᵢ(x, y, z) = 283 + 2 * (1 + z / 100) + 1e-2 * rand()
            Sᵢ(x, y, z) = 0.035 + 1e-4 * rand()

            model = fickian_model(grid, cabbeling)
            set!(model, θ = θᵢ, S = Sᵢ)
            G_θ = θ_tendency(model)
            G_S = Array(interior(model.timestepper.Gⁿ.S))
            @test G_θ ≈ .- divergence(θ_flux(model), grid) rtol = 1e-12
            @test G_S ≈ .- divergence(salt_flux(model), grid) rtol = 1e-12

            model = fickian_constant_salinity_model(grid)
            set!(model, θ = θᵢ)
            @test θ_tendency(model) ≈ .- divergence(θ_flux(model), grid) rtol = 1e-12
            @test all(component -> all(iszero, computed(component)), salt_flux(model))

            # A model type of its own, for its ValueBoundaryCondition, which only a Fickian model accepts: the model's
            # closure carries the wall flux, and so does the diagnostic θ flux
            boundary_conditions = (; θ = FieldBoundaryConditions(top = ValueBoundaryCondition(283.5)))
            model = fickian_model(grid, cabbeling; boundary_conditions)
            set!(model, θ = θᵢ, S = Sᵢ)
            G_θ = θ_tendency(model)
            wall_flux = z_flux(model, 1)[:, :, end]
            @test G_θ ≈ .- divergence(θ_flux(model), grid) rtol = 1e-12
            @test computed(θ_flux(model).z)[:, :, end] == wall_flux
            @test all(!iszero, wall_flux)
        end

        @testset "Walls and conservation" begin
            model = component_model(grid, cabbeling)
            set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100) + 1e-2 * rand(), S = (x, y, z) -> 0.035 + 1e-4 * rand())

            for id in 1:2
                Fx, Fy, Fz = fluxes(model, id)
                @test all(iszero, Fx[[1, end], :, :]) && all(iszero, Fy[:, [1, end], :]) && all(iszero, Fz[:, :, [1, end]])
                @test any(!iszero, Fx[2:end-1, :, :]) && any(!iszero, Fz[:, :, 2:end-1])
                @test abs(integral(flux_divergence(model, id))) < 1e-12 * sum(abs, Fz)
            end
        end

        @testset "Constant salinity" begin
            model = component_model(grid, cabbeling; tracers = :θ, constant_salinity = 0.035)
            set!(model, θ = (x, y, z) -> 283 + z / 100)
            @test any(!iszero, z_flux(model, 1))
        end

        @testset "Flux boundary conditions" begin
            model = flux_bc_model(grid)
            set!(model, θ = (x, y, z) -> 283 + z / 100, S = 0.035, c = (x, y, z) -> z)

            # At rest, a passive tracer has no tendency: the closure gives it no flux
            update_state!(model)
            compute_tendencies!(model, [])
            @test all(iszero, interior(model.timestepper.Gⁿ.c))

            # The prescribed fluxes change only the boundary-cell tendencies, by + Q_west / Δx, - Q_east / Δx,
            # + Q_bottom / Δz and - Q_top / Δz, summed in the cells at two walls; with the energy identity on a bounded
            # domain, this means the energy input is π Q_θ + m Q_S at the boundary
            G₀ = map(name -> Array(interior(model.timestepper.Gⁿ[name])), (θ = :θ, S = :S))
            compute_flux_bc_tendencies!(model)
            Nx, Nz = size(grid, 1), size(grid, 3)
            Δx, Δz = xspacings(grid, Center())[1], zspacings(grid, Center())[1]

            for name in (:θ, :S)
                ΔG = Array(interior(model.timestepper.Gⁿ[name])) .- G₀[name]
                Q = flux_bc_values[name]
                expected_ΔG = zeros(size(ΔG))
                expected_ΔG[1, :, :] .+= Q.west / Δx
                expected_ΔG[Nx, :, :] .-= Q.east / Δx
                expected_ΔG[:, :, 1] .+= Q.bottom / Δz
                expected_ΔG[:, :, Nz] .-= Q.top / Δz
                @test all(ΔG .≈ expected_ΔG) # exactly zero away from the walls
            end
        end

        @testset "The diagnostic fluxes are the model's total fluxes [$label]" for (label, build, z_wall_flux_values) in
                (("consistent", flux_bc_model, name -> flux_bc_values[name]),
                 ("Fickian", fickian_wall_model, name -> (bottom = 0, top = 0)))
            # At rest, the tendencies with the Flux conditions applied are σ - ∇·j of the diagnostic fluxes (θ_flux and
            # salt_flux of thermodynamic_diffusivity.jl) in every cell, wall cells included. Their wall-face values are
            # the closure's own wall flux (nonzero only for the Fickian closure, from its Value and Gradient conditions)
            # plus the flux of the Flux conditions
            model = build(grid)
            set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100) + 1e-2 * rand(), S = (x, y, z) -> 0.035 + 1e-4 * rand())
            update_state!(model)
            compute_tendencies!(model, [])
            compute_flux_bc_tendencies!(model)
            G_θ = Array(interior(model.timestepper.Gⁿ.θ))
            G_S = Array(interior(model.timestepper.Gⁿ.S))
            σ_θ = computed(θ_production(model))

            @test G_θ .- σ_θ ≈ .- divergence(θ_flux(model), grid) rtol = 1e-12
            @test G_S ≈ .- divergence(salt_flux(model), grid) rtol = 1e-12

            for (id, name, diagnostic_flux) in ((1, :θ, θ_flux), (2, :S, salt_flux))
                jx, jy, jz = map(computed, diagnostic_flux(model))
                closure_x, closure_y, closure_z = fluxes(model, id)
                Q_x, Q_z = flux_bc_values[name], z_wall_flux_values(name)
                @test jx[1, :, :] == closure_x[1, :, :] .+ Q_x.west
                @test jx[end, :, :] == closure_x[end, :, :] .+ Q_x.east
                @test jy[:, [1, end], :] == closure_y[:, [1, end], :] # no Flux conditions on the y walls
                @test jz[:, :, 1] == closure_z[:, :, 1] .+ Q_z.bottom
                @test jz[:, :, end] == closure_z[:, :, end] .+ Q_z.top
            end
        end

        @testset "Diffusive time-step constraint" begin
            model = component_model(grid, cabbeling; κ_θ = 2e-3) # ν = 1e-3 < κ_θ
            @test DiffusiveCFL(0.1)(model) ≈ 0.1 * 2e-3 / 0.25^2
        end
    end
end
