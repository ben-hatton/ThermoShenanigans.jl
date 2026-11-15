using Documenter: doctest
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports

# Run after a change to a docstring, an export or an import; the full suite (TEST_GROUP=all) includes them
function test_doctests_and_imports()
    @testset "Explicit imports" begin
        @test isnothing(check_no_implicit_imports(ThermoShenanigans))
        # diffusive_flux_x/y/z are extended inside an @eval loop through interpolated Symbols, which ExplicitImports cannot see
        @test isnothing(check_no_stale_explicit_imports(ThermoShenanigans; ignore = (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)))
    end

    doctest(ThermoShenanigans; manual = false)
end
