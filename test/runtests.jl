using Test
using Oceananigans
using ThermoShenanigans
using Documenter: doctest
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports

include("utils_for_runtests.jl")

files = test_files(test_group)
foreach(include, files)
foreach(run_test_file, files)

if test_group == "all" || test_doctests_and_imports
    run_doctests_and_imports()
else
    @info "Doctests and explicit imports are skipped: set TEST_DOCTESTS_AND_IMPORTS=true, or TEST_GROUP=all, to run them"
end
