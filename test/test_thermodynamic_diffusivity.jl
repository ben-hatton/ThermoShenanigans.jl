using Oceananigans.TimeSteppers: update_state!, compute_tendencies!, compute_flux_bc_tendencies!

@testset "ThermodynamicDiffusivity" begin
    @testset "Isothermal derivatives [$(summary(t))]" for t in (cabbeling, coupled)
        for (θ, S, z) in states
            T₀ = temperature(θ, S, z, t)
            @test - ∂T∂S(θ, S, z, t) / ∂T∂θ(θ, S, z, t) ≈ derivative(S -> isothermal_θ(T₀, S, z, t; θ), S) atol=1e-12
            @test isothermal_∂b∂S(θ, S, z, t) ≈ derivative(S -> buoyancy(isothermal_θ(T₀, S, z, t; θ), S, t), S)
            @test isothermal_∂μ∂S(θ, S, z, t) ≈ derivative(S -> chemical_potential(isothermal_θ(T₀, S, z, t; θ), S, z, t), S)
            # Maxwell relation (∂μ/∂z)_T = - (∂b/∂S)_T
            @test derivative(z -> chemical_potential(isothermal_θ(T₀, S, z, t; θ), S, z, t), z) ≈ - isothermal_∂b∂S(θ, S, z, t)
        end

        R_η, g, β = cabbeling.saline_entropy_constant, cabbeling.gravitational_acceleration, cabbeling.haline_contraction
        θ, S, z = states[2]
        T = temperature(θ, S, z, cabbeling)
        @test barodiffusion_coefficient(θ, S, z, cabbeling) ≈ - g * β * S / (R_η * T)
    end

    @testset "Resting equilibrium has zero fluxes [γ = $(t.cabbeling_coefficient)]" for t in (linear, cabbeling)
        model = component_model(test_grid(size = (1, 1, 16), extent = (1, 1, 1000)), t)
        S, Γ₁, Δz = set_resting_equilibrium!(model, t)

        heat_scale = 1e-3 * 1 / Δz        # κ_T × 1 K / Δz
        salt_scale = 1e-4 * abs(Γ₁)       # κ_S × Γ
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

        # The prescribed fluxes change only the boundary-cell tendencies, by - Q_top / Δz and + Q_bottom / Δz;
        # with the energy identity on a bounded domain, this means the energy input is π Q_θ + Σ_S Q_S at the boundary
        G₀ = map(name -> Array(interior(model.timestepper.Gⁿ[name])), (θ = :θ, S = :S))
        compute_flux_bc_tendencies!(model)
        Nz, Δz = size(grid, 3), zspacings(grid, Center())[1]

        for name in (:θ, :S)
            ΔG = Array(interior(model.timestepper.Gⁿ[name])) .- G₀[name]
            Q = flux_bc_values[name]
            @test all(ΔG[:, :, Nz] .≈ - Q.top / Δz) && all(ΔG[:, :, 1] .≈ Q.bottom / Δz)
            @test all(iszero, ΔG[:, :, 2:Nz-1])
        end
    end

    @testset "Diffusive time-step constraint" begin
        model = component_model(grid, cabbeling; κ_T = 2e-3) # ν = 1e-3 < κ_T
        @test DiffusiveCFL(0.1)(model) ≈ 0.1 * 2e-3 / 0.25^2
    end
end
