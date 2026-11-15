#=
A Julia session that keeps the compiled tests between runs. From the repository root,

    julia --project -i test/session.jl

then, as often as needed,

    julia> run_tests()                                         # the group set by TEST_GROUP, "basic" by default
    julia> run_tests("all")
    julia> run_tests("test_buoyancy.jl", "test_energetics.jl")   # test files
    julia> run_doctests_and_imports()

The first call of each test compiles it, as `Pkg.test` does; later calls reuse the compiled code. Revise tracks `src/`
and the test files, so that a later call recompiles only what an edit changed. A change to a struct, or to the test
environment, needs a new session.
=#

using Revise # from the default (global) environment, loaded before ThermoShenanigans so that `src/` is tracked

# The test dependencies of `test/Project.toml`, after the package environment so that its versions take precedence
insert!(LOAD_PATH, 2, @__DIR__)

using Test
using Oceananigans
using ThermoShenanigans
using Documenter: doctest
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports

Revise.includet(joinpath(@__DIR__, "utils_for_runtests.jl"))

foreach(file -> Revise.includet(joinpath(@__DIR__, file)), unique([test_files("all"); test_files("basic")]))

# Each argument is a test group, "basic" or "all", or a test file
function run_tests(names::AbstractString...)
    groups_and_files = isempty(names) ? (test_group,) : names
    files = mapreduce(name -> name in ("basic", "all") ? test_files(name) : [name], vcat, groups_and_files)
    foreach(run_test_file, files)
    return nothing
end
