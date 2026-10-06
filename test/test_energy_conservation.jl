using Oceananigans.TimeSteppers: update_state!, compute_tendencies!
using Oceananigans.BuoyancyFormulations: z_dot_g_bᶜᶜᶠ
using Oceananigans.Operators: Vᶜᶜᶠ
using ThermoShenanigans: consistent_buoyancyᶜᶜᶠ

# dE/dt = ∫ uᵢ Gᵢ dV + ∫ (π G_θ + Σ_S G_S) dV from the model's own tendencies, with two scales:
# the round-off scale ∫ |π G_θ| + |Σ_S G_S| dV (π ≈ cₚ makes this the largest term), and the size of the conversions ∫ |w b̃| + ε dV
function energy_tendency(model)
    update_state!(model)
    compute_tendencies!(model, [])
    G = model.timestepper.Gⁿ
    u, v, w = model.velocities
    d = thermodynamic_diagnostics(model)
    π, Σ_S, wb, ε = map(field, (d.π, d.Σ_S, d.wb, d.ε))

    kinetic = integral(*, u, G.u) + integral(*, v, G.v) + integral(*, w, G.w)
    static, round_off = integral(*, π, G.θ), integral((a, b) -> abs(a * b), π, G.θ)

    if haskey(G, :S)
        static += integral(*, Σ_S, G.S)
        round_off += integral((a, b) -> abs(a * b), Σ_S, G.S)
    end

    conversions = integral(abs, wb) + integral(ε)
    return kinetic + static, round_off, conversions
end

@testset "Energy conservation" begin
    grid = test_grid()
    stretched_grid = RectilinearGrid(size = (6, 6, 6), x = (0, 1), y = (0, 1), z = [-1, -0.7, -0.45, -0.25, -0.12, -0.04, 0],
                                     topology = (Bounded, Bounded, Bounded))

    # Oceananigans' Centered momentum advection does not conserve kinetic energy when Δz varies (an Oceananigans bug):
    # the w-momentum fluxes use Axᶠᶜᶠ ℑz(u) and Ayᶜᶠᶠ ℑz(v) instead of the volume fluxes ℑz(Ax u) and ℑz(Ay v).
    # The stretched case therefore omits momentum advection, and tests only the buoyancy work, closure and σ_θ.
    cases = (("linear",                  linear,    grid,           (;)),
             ("cabbeling",               cabbeling, grid,           (;)),
             ("coupled",                 coupled,   grid,           (;)),
             ("constant salinity",       cabbeling, grid,           (; tracers = :θ, constant_salinity = 0.035)),
             ("cabbeling, stretched Δz", cabbeling, stretched_grid, (; momentum_advection = nothing)))

    @testset "Consistent buoyancy b̃ [$label]" for (label, t, model_grid, kw) in cases[[1, 2, 3, 5]]
        model = component_model(model_grid, t; kw...)
        set_random_state!(model)
        C, tb = merge(model.velocities, model.tracers), model.buoyancy.formulation
        θ, S = model.tracers
        ℑb(i, j, k) = (buoyancy(θ[i, j, k], S[i, j, k], t) + buoyancy(θ[i, j, k-1], S[i, j, k-1], t)) / 2
        gγ = t isa CabbelingBoussinesqThermodynamics ? t.gravitational_acceleration * t.cabbeling_coefficient : 0.0

        for i in 1:6, j in 1:6, k in 2:6
            @test consistent_buoyancyᶜᶜᶠ(i, j, k, model_grid, tb, C) ≈ ℑb(i, j, k) - gγ * (θ[i, j, k] - θ[i, j, k-1])^2 / 4 atol = 1e-14
            @test z_dot_g_bᶜᶜᶠ(i, j, k, model_grid, model.buoyancy, C) == consistent_buoyancyᶜᶜᶠ(i, j, k, model_grid, tb, C)
        end
    end

    @testset "Semi-discrete energy identity [$label]" for (label, t, model_grid, kw) in cases
        model = component_model(model_grid, t; kw...)
        set_random_state!(model; S = haskey(model.tracers, :S))
        dEdt, round_off, conversions = energy_tendency(model)
        @info "dE/dt = $dEdt for conversions of $conversions [$label]"
        @test abs(dEdt) < 1e-13 * round_off
    end

    @testset "Default ℑz b would leak energy for cabbeling" begin
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        w, θ = model.velocities.w, model.tracers.θ
        gγ = cabbeling.gravitational_acceleration * cabbeling.cabbeling_coefficient
        leak = sum(w[i, j, k] * gγ * (θ[i, j, k] - θ[i, j, k-1])^2 / 4 for i in 1:6, j in 1:6, k in 2:6) * Vᶜᶜᶠ(1, 1, 1, grid)
        dEdt, _, conversions = energy_tendency(model)
        @test abs(leak) > 1e-6 * conversions && abs(leak) > 100 * abs(dEdt) # far above the round-off residual
    end

    @testset "RK3 energy error converges with Δt" begin
        section = test_grid(size = (16, 1, 16), extent = (1, 1, 1))
        thermo = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-2, parameters...)

        function energy_drift(Δt; stop_time = 8.0)
            model = component_model(section, thermo)
            set!(model, u = (x, y, z) -> 0.1 * sin(2π * x) * cos(π * z), θ = (x, y, z) -> 283 + 2 * tanh(10 * (z + 0.5)),
                        S = (x, y, z) -> 0.035 - 1e-3 * tanh(10 * (z + 0.5)))
            u, v, w = model.velocities
            Σ = thermodynamic_diagnostics(model).Σ
            Σ₀ = CenterField(section)
            set!(Σ₀, Σ)

            # Σ - Σ₀ is differenced cell by cell, since Σ ≈ cₚ θ is much larger than its changes
            energy() = integral(u -> u^2 / 2, u) + integral(w -> w^2 / 2, w) + integral(-, Σ, Σ₀)
            E₀ = energy()
            for _ in 1:round(Int, stop_time / Δt)
                time_step!(model, Δt)
            end
            return abs(energy() - E₀)
        end

        drifts = [energy_drift(Δt) for Δt in (0.2, 0.1, 0.05)]
        @info "RK3 energy drift for Δt = 0.2, 0.1, 0.05: $drifts; ratios $(drifts[1:2] ./ drifts[2:3])"
        @test all(drifts[1:2] ./ drifts[2:3] .> 6) # third order: 8
    end
end
