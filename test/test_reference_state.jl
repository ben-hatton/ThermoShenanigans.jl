using ThermoShenanigans: AbstractReferenceStateMethod, Sorting
import ThermoShenanigans: validate_reference_state_grid, reference_state_workspace, compute_reference_state!,
                          reference_tendency_workspace, compute_reference_tendency!
using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ

# A stand-in for a new reference state method, which reaches Sorting only through the method interface
struct DelegatedSorting <: AbstractReferenceStateMethod end
validate_reference_state_grid(::DelegatedSorting, grid) = validate_reference_state_grid(Sorting(), grid)
reference_state_workspace(::DelegatedSorting, grid) = reference_state_workspace(Sorting(), grid)
compute_reference_state!(z★, state, ::DelegatedSorting) = compute_reference_state!(z★, state, Sorting())
reference_tendency_workspace(::DelegatedSorting, grid) = reference_tendency_workspace(Sorting(), grid)
compute_reference_tendency!(Ṗ★, tendency, ::DelegatedSorting) = compute_reference_tendency!(Ṗ★, tendency, Sorting())

function test_reference_state()
    @testset "Reference state" begin
        grid = test_grid()
        Nx, Ny, Nz = size(grid)
        z = reshape(znodes(grid, Center()), 1, 1, Nz)
        V = cell_volumes(CenterField(grid))

        @testset "Random state [$label]" for (label, thermodynamics) in (("linear", linear), ("cabbeling", cabbeling))
            model = component_model(grid, thermodynamics)
            set_random_state!(model)
            z★_field = reference_height(model)
            z★, 𝒜, ℬ, 𝒲★, Υ, 𝒢★ = map(computed, (z★_field, available_potential_energy(z★_field),
                                                   background_potential_energy(model, z★_field), reference_work(z★_field),
                                                   reference_displacement(z★_field), background_gravitational_energy(model, z★_field)))
            𝒫 = computed(potential_energy(model))
            ℐ = computed(internal_energy(model))
            b = computed(KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, model.buoyancy.formulation,
                                                                         model.tracers))

            @test all(-1 .< z★ .< 0)
            @test sum(z★ .* V) ≈ sum(z .* V)
            @test Υ ≈ z★ .- z
            @test all(≥(0), 𝒜) && any(>(0), 𝒜)
            @test abs(sum(𝒲★ .* V)) < 1e-14 * sum(abs.(𝒲★) .* V)
            @test 𝒜 .+ ℬ ≈ 𝒫 rtol = 1e-14
            @test ℐ .+ 𝒢★ ≈ ℬ rtol = 1e-14
            @test 𝒢★ ≈ .- b .* z★ .- 𝒲★ # z₀ = 0

            # Denser parcels sit lower in the reference state, and b★ increases upwards
            order = sortperm(vec(b))
            @test issorted(vec(z★)[order])
            @test issorted(vec(interior(compute!(reference_buoyancy(z★_field)))))
        end

        @testset "Translating the grid and the reference pressure height together" begin
            # 𝒜 and 𝒢★ are unchanged, and ℬ = ℐ + 𝒢★ + g z₀ changes by the constant g z₀ of 𝒫
            reference, translated, _ = translated_models()
            function reservoirs(model)
                z★ = reference_height(model)
                return map(computed, (available_potential_energy(z★), background_gravitational_energy(model, z★),
                                      background_potential_energy(model, z★),
                                      internal_energy(model)))
            end
            𝒜, 𝒢★, ℬ, ℐ = reservoirs(reference)
            𝒜′, 𝒢★′, ℬ′, ℐ′ = reservoirs(translated)
            thermodynamics = translated.buoyancy.formulation.thermodynamics
            g, z₀ = gravitational_acceleration(thermodynamics), reference_pressure_height(thermodynamics)

            @test any(>(0), 𝒜)
            @test 𝒜′ ≈ 𝒜
            @test 𝒢★′ ≈ 𝒢★
            @test ℬ′ .- g * z₀ ≈ ℬ
            @test ℐ′ .+ 𝒢★′ .+ g * z₀ ≈ ℬ′ rtol = 1e-14
        end

        #=
        The identities of the energy reservoirs in every cell: 𝒫 = ℐ + 𝒢 + g₀ z₀, ℬ = ℐ + 𝒢★ + g₀ z₀ (b and h⁰ are
        independent of z) and 𝒜 = 𝒢 - 𝒢★ for the consistent model, and 𝒜 = 𝒢 - 𝒢★ for the Fickian model, the only one
        whose terms it defines. `shifted` has z₀ = -40, so that g₀ z₀ ≠ 0 and 𝒢 and 𝒢★ are each about 40 b, much larger
        than 𝒜: the round-off of 𝒢 - 𝒢★ is measured against 𝒢.
        =#
        @testset "Energy reservoir identities [$label, $closure]" for (label, thermodynamics) in (("linear", linear),
                                                                                                 ("cabbeling", cabbeling),
                                                                                                 ("z₀ = -40", shifted)),
                                                                     closure in ("consistent", "Fickian")
            g₀, z₀ = gravitational_acceleration(thermodynamics), reference_pressure_height(thermodynamics)
            model = closure == "consistent" ? component_model(grid, thermodynamics) : fickian_model(grid, thermodynamics)
            set_random_state!(model)
            z★ = reference_height(model)
            𝒜, 𝒢, 𝒢★ = map(computed, (available_potential_energy(z★), gravitational_energy(model),
                                       background_gravitational_energy(model, z★)))

            @test any(>(0), 𝒜)
            @test maximum(abs, 𝒜 .- (𝒢 .- 𝒢★)) < 1e-14 * maximum(abs, 𝒢)

            if closure == "consistent"
                𝒫, ℐ, ℬ = map(computed, (potential_energy(model), internal_energy(model), background_potential_energy(model, z★)))
                @test ℐ .+ 𝒢 .+ g₀ * z₀ ≈ 𝒫 rtol = 1e-14
                @test ℐ .+ 𝒢★ .+ g₀ * z₀ ≈ ℬ rtol = 1e-14
                @test 𝒜 .+ ℬ ≈ 𝒫 rtol = 1e-14
            end
        end

        @testset "Stable, horizontally uniform column" begin
            model = component_model(grid, cabbeling)
            set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
            z★ = reference_height(model)
            @test computed(z★) ≈ repeat(z, Nx, Ny, 1)
            @test maximum(abs, computed(available_potential_energy(z★))) <
                  1e-15 * maximum(abs, interior(compute!(reference_potential(z★))))
        end

        @testset "Tied buoyancies share the mid-height of their band" begin
            model = component_model(grid, linear)
            # Two classes, each filling a random half of the cells: the dense class fills the bottom half of the sorted column
            dense = rand(Bool, Nx, Ny, Nz)
            set!(model, θ = ifelse.(dense, 283.0, 284.0), S = 0.035)
            z★ = computed(reference_height(model))
            n_dense = count(dense)
            @test all(z★[dense] .≈ -1 + (n_dense / (Nx * Ny * Nz)) / 2)
            @test all(z★[.!dense] .≈ -1 + (n_dense / (Nx * Ny * Nz) + 1) / 2)

            # R★ still integrates to zero, through the averaging over each band of ties
            R★ = computed(reference_tendency_work(model, reference_height(model)))
            @test any(!iszero, R★)
            @test abs(sum(R★ .* V)) < 1e-13 * sum(abs.(R★) .* V)
        end

        @testset "Reference profiles recompute the sort" begin
            model = component_model(grid, linear)
            set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
            b★ = reference_buoyancy(reference_height(model))
            b★_stable = Array(interior(compute!(b★)))
            set!(model, θ = (x, y, z) -> 283 + 4z)
            @test Array(interior(compute!(b★))) ≈ 2 .* b★_stable # b = gα (θ - 283) doubles
        end

        @testset "Reference state methods" begin
            model = component_model(grid, cabbeling)
            set_random_state!(model)
            @test reference_height(model).operand.method isa Sorting

            # The diagnostics depend on the method only through the fields it fills, which are compared directly (rather than
            # through the diagnostics, whose kernels would be compiled again for the new method)
            function filled_fields(method)
                z★ = compute!(reference_height(model; method))
                Ṗ★ = compute!(ThermoShenanigans.reference_tendency(model, z★))
                state = z★.operand
                return (Array(interior(z★)), interior_data(state.parcel_reference_potential, Center, Center, Center, grid),
                        interior_data(state.reference_potential, Nothing, Nothing, Center, grid),
                        interior_data(state.reference_buoyancy, Nothing, Nothing, Face, grid), Array(interior(Ṗ★)),
                        interior_data(Ṗ★.operand.reference_tendency, Nothing, Nothing, Center, grid))
            end
            @test filled_fields(Sorting()) == filled_fields(DelegatedSorting())
        end

        @testset "Stretched grids are rejected" begin
            stretched = RectilinearGrid(size = (2, 2, 3), x = (0, 1), y = (0, 1), z = [-1, -0.5, -0.2, 0], topology = (Bounded, Bounded, Bounded))
            model = component_model(stretched, linear)
            @test_throws ArgumentError reference_height(model)
        end
    end
end
