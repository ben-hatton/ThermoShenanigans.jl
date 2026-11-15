#=
Runs the test groups or files named by `test_args`, or else by TEST_GROUP ("basic" by default); see test/test_groups.jl.

    Pkg.test(test_args = ["energetics"])
    Pkg.test(test_args = ["thermodynamics", "test_buoyancy.jl"])
    TEST_GROUP=all julia --project -e 'using Pkg; Pkg.test()'
=#

include("utils_for_runtests.jl")
include("test_groups.jl")

files = test_files(selected_names())
@info "Running $(join(files, ", "))"

test_groups()
foreach(include, files)
foreach(run_test_file, files)

"test_doctests_and_imports.jl" in files ||
    @info "Doctests and explicit imports are skipped: select the group doctests_and_imports or all, or set TEST_DOCTESTS_AND_IMPORTS=true, to run them"
