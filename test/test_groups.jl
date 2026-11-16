#=
Test groups, selected by name with `Pkg.test(test_args = ["energetics"])`, `TEST_GROUP=energetics`, or
`run_tests("energetics")` in `test/session.jl`. A name is a group below, or a test file (`test_ape_budget.jl`).

Each source group lists the files in `src/` and the test files that cover them: after a change to a source file, run the
groups that list it. `test_groups()` checks that every file in `src/` belongs to at least one group and every test file
to `all`, so a source or test file that is added, split, renamed or removed must be entered here in the same change.
=#

using Test

const source_groups = (
    thermodynamics = (sources = ["abstract_boussinesq_thermodynamics.jl", "cabbeling_boussinesq_thermodynamics.jl"],
                      tests   = ["test_thermodynamics.jl"]),

    # The energy-conservation tests check that the consistent buoyancy b̃ and the closure's fluxes conserve 𝒦 + 𝒫
    buoyancy       = (sources = ["thermodynamic_buoyancy.jl", "model_components.jl"],
                      tests   = ["test_buoyancy.jl", "test_energy_conservation.jl"]),

    # The architecture tests hold the GPU comparison of the Fickian fluxes
    diffusivity    = (sources = ["thermodynamic_diffusivity.jl", "model_components.jl"],
                      tests   = ["test_diffusivity.jl", "test_energy_conservation.jl", "test_architectures.jl"]),

    # The architecture tests compare every exported diagnostic between architectures and float types
    production     = (sources = ["production.jl", "diagnostics.jl"],
                      tests   = ["test_production.jl", "test_architectures.jl"]),

    reference_state = (sources = ["reference_state.jl", "sorting.jl"],
                       tests   = ["test_reference_state.jl", "test_architectures.jl"]),

    # The budgets of 𝒜, 𝒦 and ℬ share terms, and the tests of ℬ close the budgets of all three
    ape_budget     = (sources = ["ape_budget.jl"],
                      tests   = ["test_ape_budget.jl", "test_bpe_budget.jl", "test_architectures.jl"]),

    ke_budget      = (sources = ["ke_budget.jl"],
                      tests   = ["test_ke_budget.jl", "test_bpe_budget.jl", "test_architectures.jl"]),

    bpe_budget     = (sources = ["bpe_budget.jl"],
                      tests   = ["test_bpe_budget.jl", "test_architectures.jl"]),

    # σ_θ of `θ_production_forcing`, with the production parameters it shares with the diagnostics, and a user θ forcing
    # combined with σ_θ
    model_components = (sources = ["model_components.jl", "production.jl", "diagnostics.jl"],
                        tests   = ["test_model_components.jl"]),

    # The exports and imports of the module file, and the docstrings of every file
    doctests_and_imports = (sources = ["ThermoShenanigans.jl"],
                            tests   = ["test_doctests_and_imports.jl"]))

# Groups of test files that are not tied to source files, in the order they run
const suite_groups = (
    # A quick core set (the default): the thermodynamic derivatives, and the first and second laws and a time step on one model
    basic = ["test_thermodynamics.jl", "test_basic.jl"],

    # The reference state and the local energetics, whose sources are all of the groups reference_state and *_budget
    energetics = ["test_reference_state.jl", "test_ape_budget.jl", "test_ke_budget.jl", "test_bpe_budget.jl",
                  "test_architectures.jl"],

    # The full suite, with every test file but `test_basic.jl`, whose tests it repeats for more thermodynamics and grids
    all = ["test_thermodynamics.jl", "test_buoyancy.jl", "test_diffusivity.jl", "test_production.jl",
           "test_energy_conservation.jl", "test_model_components.jl", "test_reference_state.jl", "test_ape_budget.jl",
           "test_ke_budget.jl", "test_bpe_budget.jl", "test_architectures.jl", "test_doctests_and_imports.jl"])

group_tests(name::Symbol) = haskey(suite_groups, name) ? suite_groups[name] : source_groups[name].tests

is_test_file(name) = isfile(joinpath(@__DIR__, name)) && startswith(name, "test_") && endswith(name, ".jl")

# Test files of `names`, groups (such as "energetics" or "all") or test files (such as "test_ape_budget.jl"), in the
# order they run and without repeats
function test_files(names)
    files = String[]
    for name in names
        if Symbol(name) in (keys(suite_groups)..., keys(source_groups)...)
            append!(files, group_tests(Symbol(name)))
        elseif is_test_file(name)
            push!(files, name)
        else
            valid = join(string.((keys(suite_groups)..., keys(source_groups)...)), ", ")
            throw(ArgumentError("unknown test group or file \"$name\": use one of $valid, or a test_*.jl file in test/"))
        end
    end
    return unique(files)
end

# Groups or files named by `ARGS` (`Pkg.test(test_args = [...])`), or else by `TEST_GROUP` (space-separated, "basic" by
# default), with the doctests and the explicit-imports check added when TEST_DOCTESTS_AND_IMPORTS=true
function selected_names(args = ARGS)
    names = isempty(args) ? split(get(ENV, "TEST_GROUP", "basic")) : args
    doctests_and_imports = get(ENV, "TEST_DOCTESTS_AND_IMPORTS", "false") == "true"
    return doctests_and_imports ? [names; "doctests_and_imports"] : collect(names)
end

function test_groups()
    @testset "Test groups" begin
        @testset "$file in src/ belongs to a group" for file in readdir(joinpath(dirname(@__DIR__), "src"))
            @test any(group -> file in group.sources, source_groups)
        end

        all_tests = suite_groups.all ∪ suite_groups.basic
        test_dir_files = filter(file -> startswith(file, "test_") && endswith(file, ".jl") && file != "test_groups.jl",
                                readdir(@__DIR__))
        @testset "$file in test/ belongs to a group" for file in test_dir_files
            @test file in all_tests
        end

        @testset "Groups name existing files" begin
            for group in source_groups
                @test all(file -> isfile(joinpath(dirname(@__DIR__), "src", file)), group.sources)
            end
            @test all(is_test_file, all_tests)
        end

        # `all` runs every test of the source groups, so that a change to any source file is covered by the full suite
        @test issetequal(suite_groups.all, mapreduce(group -> group.tests, vcat, source_groups))
    end
end
