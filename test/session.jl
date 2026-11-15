#=
A Julia session that keeps the compiled tests between runs. From the repository root,

    julia --project -i test/session.jl

then, as often as needed,

    julia> run_tests()                                         # the groups set by TEST_GROUP, "basic" by default
    julia> run_tests("all")
    julia> run_tests("energetics", "diffusivity")              # test groups, see test/test_groups.jl
    julia> run_tests("test_buoyancy.jl", "test_energetics.jl")   # test files
    julia> run_doctests_and_imports()                          # the group doctests_and_imports

The first call of each test compiles it, as `Pkg.test` does; later calls reuse the compiled code. Revise tracks `src/`
and the test files, so that a later call recompiles only what an edit changed. A change to a struct, or to the test
environment, needs a new session.
=#

using Revise # from the default (global) environment, loaded before ThermoShenanigans so that `src/` is tracked

# The test dependencies of `test/Project.toml`, after the package environment so that its versions take precedence
insert!(LOAD_PATH, 2, @__DIR__)

Revise.includet(joinpath(@__DIR__, "utils_for_runtests.jl"))
Revise.includet(joinpath(@__DIR__, "test_groups.jl"))

foreach(file -> Revise.includet(joinpath(@__DIR__, file)), test_files(["all", "basic"]))

# Each argument is a test group or a test file
function run_tests(names::AbstractString...)
    test_groups()
    foreach(run_test_file, test_files(isempty(names) ? selected_names(String[]) : collect(names)))
    return nothing
end

run_doctests_and_imports() = run_tests("doctests_and_imports")
