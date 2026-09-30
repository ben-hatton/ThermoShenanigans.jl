using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ
using Oceananigans.Operators: Vᶜᶜᶜ
using ThermoShenanigans: barodiffusion_coefficient, diffusive_flux_x, diffusive_flux_y, diffusive_flux_z

# θ on the isotherm T(θ, S, z) = T₀ (Newton; differentiable with ForwardDiff)
function isothermal_θ(T₀, S, z, t; θ = T₀ + zero(S * z))
    for _ in 1:30
        θ -= (in_situ_temperature(θ, S, z, t) - T₀) / ∂T∂θ(θ, S, z, t)
    end
    return θ
end

m(θ, S, z, t) = chemical_potential(θ, S, z, t) - in_situ_temperature(θ, S, z, t) * ∂η⁰∂S(θ, S, t) # m = μ - T η⁰_S

closure_model(grid, thermo; tracers = (:θ, :S), κ_T = 1e-3, κ_S = 1e-4, kw...) =
    NonhydrostaticModel(grid; tracers, buoyancy = ThermodynamicBuoyancy(grid, thermo; kw...),
                        closure = (ScalarDiffusivity(eltype(grid); ν = 1e-3),
                                   ThermodynamicDiffusivity(eltype(grid); thermal_diffusivity = κ_T, salt_diffusivity = κ_S)))

fluxes(model, id, c) = map(f -> [f(i, j, k, model.grid, model.closure, model.closure_fields, Val(id), c, model.clock,
                                   merge(model.velocities, model.tracers), model.buoyancy)
                                 for i in 1:size(model.grid, 1) + 1, j in 1:size(model.grid, 2) + 1, k in 1:size(model.grid, 3) + 1],
                           (diffusive_flux_x, diffusive_flux_y, diffusive_flux_z))

@testset "ThermodynamicDiffusivity" begin
    @testset "Isothermal derivatives [$(summary(t))]" for t in (cabbeling, cross)
        for (θ, S, z) in states
            T₀ = in_situ_temperature(θ, S, z, t)
            @test isothermal_∂θ∂S(θ, S, z, t) ≈ derivative(S -> isothermal_θ(T₀, S, z, t; θ), S)             atol=1e-12
            @test isothermal_∂b∂S(θ, S, z, t) ≈ derivative(S -> buoyancy(isothermal_θ(T₀, S, z, t; θ), S, t), S)
            @test isothermal_∂m∂S(θ, S, z, t) ≈ derivative(S -> m(isothermal_θ(T₀, S, z, t; θ), S, z, t), S)
            # Maxwell relation (∂m/∂z)_T = - (∂b/∂S)_T
            @test derivative(z -> m(isothermal_θ(T₀, S, z, t; θ), S, z, t), z) ≈ - isothermal_∂b∂S(θ, S, z, t)
        end

        R_η, g, β = cabbeling.saline_entropy_constant, cabbeling.gravitational_acceleration, cabbeling.haline_contraction
        θ, S, z = states[2]
        T = in_situ_temperature(θ, S, z, cabbeling)
        @test barodiffusion_coefficient(θ, S, z, cabbeling) ≈ - g * β * S / (R_η * T)
    end

    @testset "Resting equilibrium has zero fluxes [γ = $(t.cabbeling_coefficient)]" for t in (linear, cabbeling)
        grid = RectilinearGrid(size = (1, 1, 16), extent = (1, 1, 1000), topology = (Periodic, Periodic, Bounded))
        model = closure_model(grid, t)
        z = znodes(grid, Center())

        # Uniform T₀, and S marched upwards so that δS / Δz = ℑΓ on every interior face
        T₀, Nz, Δz = 283.0, size(grid, 3), zspacings(grid, Center())[1]
        S, θ = zeros(Nz), zeros(Nz)
        S[1] = 0.035
        θ[1] = isothermal_θ(T₀, S[1], z[1], t)
        Γ(k) = barodiffusion_coefficient(θ[k], S[k], z[k], t)
        for k in 2:Nz
            S[k], θ[k] = S[k-1], θ[k-1]
            for _ in 1:50
                θ[k] = isothermal_θ(T₀, S[k], z[k], t)
                S[k] = S[k-1] + Δz * (Γ(k-1) + Γ(k)) / 2
            end
        end
        set!(model, θ = reshape(θ, 1, 1, Nz), S = reshape(S, 1, 1, Nz))

        heat_scale = 1e-3 * 1 / Δz        # κ_T × 1 K / Δz
        salt_scale = 1e-4 * abs(Γ(1))     # κ_S × Γ
        @test maximum(abs, fluxes(model, 1, model.tracers.θ)[3]) < 1e-10 * heat_scale
        @test maximum(abs, fluxes(model, 2, model.tracers.S)[3]) < 1e-10 * salt_scale
        @test S[end] < S[1] # barodiffusion: saltier at depth
    end

    @testset "g → 0 reduces to ScalarDiffusivity" begin
        grid = RectilinearGrid(size = (4, 4, 4), extent = (1, 1, 1))
        thermo = CabbelingBoussinesqThermodynamics(; parameters..., gravitational_acceleration = 0)
        model = closure_model(grid, thermo)
        reference = NonhydrostaticModel(grid; tracers = (:θ, :S), buoyancy = ThermodynamicBuoyancy(grid, thermo),
                                        closure = ScalarDiffusivity(κ = (θ = 1e-3, S = 1e-4)))

        θᵢ(x, y, z) = 283 + rand()
        Sᵢ(x, y, z) = 0.035 + 1e-3 * rand()
        set!(model, θ = θᵢ, S = Sᵢ)
        set!(reference, θ = model.tracers.θ, S = model.tracers.S)

        for (id, name) in enumerate((:θ, :S)), i in 1:4, j in 1:4, k in 1:4
            args(m) = (i, j, k, grid, m.closure, m.closure_fields, Val(id), m.tracers[name], m.clock,
                       merge(m.velocities, m.tracers), m.buoyancy)
            @test ∇_dot_qᶜ(args(model)...) ≈ ∇_dot_qᶜ(args(reference)...) rtol=1e-12 atol=1e-18
        end
    end

    @testset "Walls and conservation" begin
        grid = RectilinearGrid(size = (4, 4, 8), extent = (1, 1, 100), topology = (Periodic, Bounded, Bounded))
        model = closure_model(grid, cabbeling)
        set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100) + 1e-2 * rand(), S = (x, y, z) -> 0.035 + 1e-4 * rand())

        for (id, c) in enumerate(model.tracers)
            Fx, Fy, Fz = fluxes(model, id, c)
            @test all(iszero, Fy[:, [1, end], :]) && all(iszero, Fz[:, :, [1, end]])
            @test any(!iszero, Fz[:, :, 2:end-1])

            total = sum(∇_dot_qᶜ(i, j, k, grid, model.closure, model.closure_fields, Val(id), c, model.clock,
                                 merge(model.velocities, model.tracers), model.buoyancy) * Vᶜᶜᶜ(i, j, k, grid)
                        for i in 1:4, j in 1:4, k in 1:8)
            @test abs(total) < 1e-12 * sum(abs, Fz)
        end
    end

    @testset "Passive tracer and constant salinity" begin
        grid = RectilinearGrid(size = (2, 2, 4), extent = (1, 1, 100))
        model = closure_model(grid, cabbeling; tracers = (:θ, :S, :c))
        set!(model, θ = (x, y, z) -> 283 + z / 100, S = 0.035, c = (x, y, z) -> z)
        @test all(F -> all(iszero, F), fluxes(model, 3, model.tracers.c))

        model = closure_model(grid, cabbeling; tracers = :θ, constant_salinity = 0.035)
        set!(model, θ = (x, y, z) -> 283 + z / 100)
        @test any(!iszero, fluxes(model, 1, model.tracers.θ)[3])
        time_step!(model, 1)
        @test all(isfinite, interior(model.tracers.θ))
    end

    @testset "Model steps [$FT]" for FT in (Float64, Float32)
        grid = RectilinearGrid(FT, size = (4, 4, 8), extent = (1, 1, 100))
        model = closure_model(grid, CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...))
        @test model.closure[2] isa ThermodynamicDiffusivity{FT}
        set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100), S = 0.035)
        time_step!(model, 1)
        @test all(isfinite, interior(model.tracers.θ)) && all(isfinite, interior(model.tracers.S))
    end
end
