using Oceananigans.Models: buoyancy_operation
using Oceananigans.BuoyancyFormulations: ∂z_b

@testset "ThermodynamicBuoyancy" begin
    @testset "Construction [$FT]" for FT in (Float64, Float32)
        grid = RectilinearGrid(FT, size = (4, 4, 8), extent = (1, 1, 100))
        thermo = CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...)
        @test thermo.saline_entropy_constant isa FT
        @test ThermodynamicBuoyancy(grid, thermo).surface_height === zero(FT) # grid top
        @test ThermodynamicBuoyancy(grid, thermo; surface_height = -10).surface_height === FT(-10)
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

        # T at (C, C, C) uses the height above the surface
        T = field(thermodynamic_diagnostics(model).T)
        @test T[i, j, k] ≈ temperature(θ[i, j, k], S[i, j, k], z, cabbeling)

        # N² is the difference of centred b at (C, C, F)
        @test ∂z_b(i, j, k, grid, model.buoyancy, model.tracers) ≈ (b[i, j, k] - b[i, j, k-1]) / zspacings(grid, Center())[k]
    end

    @testset "Constant salinity" begin
        model = component_model(grid, cabbeling; tracers = :θ, constant_salinity = 0.035)
        tb = model.buoyancy.formulation
        @test ThermoShenanigans.required_tracers(tb) == (:θ,)
        @test sprint(show, tb) == "ThermodynamicBuoyancy with CabbelingBoussinesqThermodynamics{Float64} and surface_height=0.0, constant_salinity=0.035"

        set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100))
        θ = model.tracers.θ
        @test field(buoyancy_operation(model))[i, j, k] ≈ buoyancy(θ[i, j, k], 0.035, cabbeling)
        @test field(thermodynamic_diagnostics(model).T)[i, j, k] ≈ temperature(θ[i, j, k], 0.035, z, cabbeling)
    end
end
