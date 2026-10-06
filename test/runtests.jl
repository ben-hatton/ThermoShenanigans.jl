using Test
using Oceananigans
using ThermoShenanigans
using Documenter: doctest
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports

include("utils_for_runtests.jl")

include("test_thermodynamics.jl")
include("test_thermodynamic_buoyancy.jl")
include("test_thermodynamic_diffusivity.jl")
include("test_irreversible_production.jl")
include("test_energy_conservation.jl")
include("test_architectures.jl")

@testset "Explicit imports" begin
    @test isnothing(check_no_implicit_imports(ThermoShenanigans))
    # diffusive_flux_x/y/z are extended inside an @eval loop through interpolated Symbols, which ExplicitImports cannot see
    @test isnothing(check_no_stale_explicit_imports(ThermoShenanigans; ignore = (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)))
end

doctest(ThermoShenanigans; manual = false)
