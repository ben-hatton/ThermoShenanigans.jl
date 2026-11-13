using Oceananigans.Models: buoyancy_operation
using Oceananigans.BuoyancyFormulations: ∂z_b

@testset "ThermodynamicBuoyancy" begin
    @testset "Construction [$FT]" for FT in (Float64, Float32)
        grid = RectilinearGrid(FT, size = (4, 4, 8), extent = (1, 1, 100))
        thermo = CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...)
        @test thermo.saline_entropy_constant isa FT
        @test thermo.reference_pressure_height === zero(FT)
        @test CabbelingBoussinesqThermodynamics(FT; reference_pressure_height = -10, parameters...).reference_pressure_height === FT(-10)
        @test ThermodynamicBuoyancy(grid, thermo).thermodynamics === thermo
    end

    grid = test_grid(size = (4, 4, 8), extent = (1, 1, 100))
    i, j, k = 2, 3, 5
    z = znodes(grid, Center())[k]

    @testset "b, T and N² at cell centres" begin
        model = component_model(grid, cabbeling)
        set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100) + 1e-2 * randn(), S = (x, y, z) -> 0.035 - 1e-4 * z / 100)
        θ, S = model.tracers

        # b at (C, C, C) matches the point-wise buoyancy
        b = field(buoyancy_operation(model))
        @test b[i, j, k] ≈ buoyancy(θ[i, j, k], S[i, j, k], cabbeling)

        # T at (C, C, C) uses the height of the cell centre
        T = field(thermodynamic_operation(model, temperature))
        @test T[i, j, k] ≈ temperature(θ[i, j, k], S[i, j, k], z, cabbeling)

        # N² is the difference of centred b at (C, C, F)
        @test ∂z_b(i, j, k, grid, model.buoyancy, model.tracers) ≈ (b[i, j, k] - b[i, j, k-1]) / zspacings(grid, Center())[k]
    end

    @testset "Constant salinity" begin
        model = component_model(grid, cabbeling; tracers = :θ, constant_salinity = 0.035)
        buoyancy_formulation = model.buoyancy.formulation
        @test ThermoShenanigans.required_tracers(buoyancy_formulation) == (:θ,)
        @test sprint(show, buoyancy_formulation) == """
            ThermodynamicBuoyancy:
            ├── constant_salinity: 0.035
            └── thermodynamics: CabbelingBoussinesqThermodynamics{Float64}"""

        set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100))
        θ = model.tracers.θ
        @test field(buoyancy_operation(model))[i, j, k] ≈ buoyancy(θ[i, j, k], 0.035, cabbeling)
        @test field(thermodynamic_operation(model, temperature))[i, j, k] ≈ temperature(θ[i, j, k], 0.035, z, cabbeling)
    end

    @testset "Translating the grid and the reference pressure height together" begin
        # z enters the model only through z - z₀, so the translated model with z₀ translated too has the same dynamics
        reference, translated, translated_without_z₀ = translated_models()
        T, T_translated, T_without_z₀ = map(model -> computed(thermodynamic_operation(model, temperature)),
                                            (reference, translated, translated_without_z₀))
        @test T_translated ≈ T
        @test !(T_without_z₀ ≈ T)
        @test computed(thermodynamic_operation(translated, exner)) ≈ computed(thermodynamic_operation(reference, exner))

        for model in (reference, translated)
            time_step!(model, 1e-3)
        end
        for name in (:θ, :S)
            @test interior(translated.tracers[name]) ≈ interior(reference.tracers[name])
        end
        @test interior(translated.velocities.w) ≈ interior(reference.velocities.w)
        @test any(!iszero, interior(reference.velocities.w))
    end
end
