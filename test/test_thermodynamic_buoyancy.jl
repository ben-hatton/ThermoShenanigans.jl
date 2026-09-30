using Oceananigans.Models: buoyancy_operation
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.AbstractOperations: KernelFunctionOperation
using ThermoShenanigans: height_dependent_functionᶜᶜᶜ

@testset "ThermodynamicBuoyancy [$FT]" for FT in (Float64, Float32)
    grid = RectilinearGrid(FT, size=(4, 4, 8), extent=(1, 1, 100))
    thermo = CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...)
    @test thermo.saline_entropy_constant isa FT
    tb = ThermodynamicBuoyancy(grid, thermo)

    @test tb.surface_height === zero(FT) # grid top
    @test ThermodynamicBuoyancy(grid, thermo; surface_height = -10).surface_height === FT(-10)

    model = NonhydrostaticModel(grid; tracers = (:θ, :S), buoyancy = tb, closure = ScalarDiffusivity(ν = 1e-2, κ = 1e-2))

    θᵢ(x, y, z) = 283 + 2 * (1 + z / 100) + 1e-2 * randn()
    Sᵢ(x, y, z) = 0.035 - 1e-4 * z / 100
    set!(model, θ = θᵢ, S = Sᵢ)

    # b at (C, C, C) matches the point-wise buoyancy
    b = Field(buoyancy_operation(model))
    compute!(b)
    θ, S = model.tracers
    i, j, k = 2, 3, 5
    @test b[i, j, k] ≈ buoyancy(θ[i, j, k], S[i, j, k], thermo)

    # T at (C, C, C) uses the height above the surface
    T = Field(KernelFunctionOperation{Center, Center, Center}(height_dependent_functionᶜᶜᶜ, grid,
                                                              in_situ_temperature, tb, model.tracers))
    compute!(T)
    z = znodes(grid, Center())[k]
    @test T[i, j, k] ≈ in_situ_temperature(θ[i, j, k], S[i, j, k], z, thermo)

    # N² is the difference of centred b at (C, C, F)
    @test ∂z_b(i, j, k, grid, model.buoyancy, model.tracers) ≈ (b[i, j, k] - b[i, j, k-1]) / zspacings(grid, Center())[k]

    time_step!(model, 1)
    @test all(isfinite, interior(model.velocities.w))
end

@testset "ThermodynamicBuoyancy with constant salinity" begin
    grid = RectilinearGrid(size=(4, 4, 8), extent=(1, 1, 100))
    thermo = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, parameters...)
    tb = ThermodynamicBuoyancy(grid, thermo; constant_salinity = 0.035)
    @test ThermoShenanigans.required_tracers(tb) == (:θ,)

    model = NonhydrostaticModel(grid; tracers = :θ, buoyancy = tb, closure = ScalarDiffusivity(ν = 1e-2, κ = 1e-2))
    set!(model, θ = (x, y, z) -> 283 + 2 * (1 + z / 100))

    b = Field(buoyancy_operation(model))
    compute!(b)
    θ = model.tracers.θ
    i, j, k = 2, 3, 5
    @test b[i, j, k] ≈ buoyancy(θ[i, j, k], 0.035, thermo)

    T = Field(KernelFunctionOperation{Center, Center, Center}(height_dependent_functionᶜᶜᶜ, grid,
                                                              in_situ_temperature, tb, model.tracers))
    compute!(T)
    @test T[i, j, k] ≈ in_situ_temperature(θ[i, j, k], 0.035, znodes(grid, Center())[k], thermo)

    time_step!(model, 1)
    @test all(isfinite, interior(model.velocities.w))
end
