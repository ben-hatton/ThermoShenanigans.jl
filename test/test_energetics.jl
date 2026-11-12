using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ

@testset "Energetics" begin
    grid = test_grid()
    Nx, Ny, Nz = size(grid)
    z = reshape(znodes(grid, Center()), 1, 1, Nz)
    V = cell_volumes(CenterField(grid))

    @testset "Random state [$label]" for (label, thermodynamics) in (("linear", linear), ("cabbeling", cabbeling))
        model = component_model(grid, thermodynamics)
        set_random_state!(model)
        e = energetics(model)
        z★, 𝒜, 𝓑, 𝒲★, Υ = map(computed, (e.z★, e.𝒜, e.𝓑, e.𝒲★, e.Υ))
        𝒫 = computed(thermodynamic_diagnostics(model).𝒫)
        b = computed(KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, model.buoyancy.formulation,
                                                                     model.tracers))

        @test all(-1 .< z★ .< 0)
        @test sum(z★ .* V) ≈ sum(z .* V)
        @test Υ ≈ z★ .- z
        @test all(≥(0), 𝒜) && any(>(0), 𝒜)
        @test abs(sum(𝒲★ .* V)) < 1e-14 * sum(abs.(𝒲★) .* V)
        @test 𝒜 .+ 𝓑 ≈ 𝒫 rtol = 1e-14
        @test computed(e.ℐ) .+ computed(e.𝒢★) ≈ 𝓑 rtol = 1e-14
        @test computed(e.𝒢★) ≈ .- b .* z★ .- 𝒲★ # z₀ = 0 at the top of the grid

        # Denser parcels sit lower in the reference state, and b★ increases upwards
        order = sortperm(vec(b))
        @test issorted(vec(z★)[order])
        @test issorted(vec(interior(compute!(e.b★))))
    end

    @testset "Stable, horizontally uniform column" begin
        model = component_model(grid, cabbeling)
        set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
        e = energetics(model)
        @test computed(e.z★) ≈ repeat(z, Nx, Ny, 1)
        @test maximum(abs, computed(e.𝒜)) < 1e-15 * maximum(abs, interior(compute!(e.p★)))
    end

    @testset "Tied buoyancies share the mid-height of their band" begin
        model = component_model(grid, linear)
        # Two classes, each filling a random half of the cells: the dense class fills the bottom half of the sorted column
        dense = rand(Bool, Nx, Ny, Nz)
        set!(model, θ = ifelse.(dense, 283.0, 284.0), S = 0.035)
        z★ = computed(energetics(model).z★)
        n_dense = count(dense)
        @test all(z★[dense] .≈ -1 + (n_dense / (Nx * Ny * Nz)) / 2)
        @test all(z★[.!dense] .≈ -1 + (n_dense / (Nx * Ny * Nz) + 1) / 2)
    end

    @testset "Reference profiles recompute the sort" begin
        model = component_model(grid, linear)
        set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
        e = energetics(model)
        b★ = compute!(e.b★)
        b★_stable = Array(interior(b★))
        set!(model, θ = (x, y, z) -> 283 + 4z)
        @test Array(interior(compute!(e.b★))) ≈ 2 .* b★_stable # b = gα (θ - 283) doubles
    end

    @testset "Buoyancy production split" begin
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        d = thermodynamic_diagnostics(model)
        @test computed(d.σ_b_heating) .+ computed(d.σ_b_nonlin) ≈ computed(d.σ_b)
    end

    @testset "Stretched grids are rejected" begin
        stretched = RectilinearGrid(size = (2, 2, 3), x = (0, 1), y = (0, 1), z = [-1, -0.5, -0.2, 0], topology = (Bounded, Bounded, Bounded))
        model = component_model(stretched, linear)
        @test_throws ArgumentError energetics(model)
    end
end
