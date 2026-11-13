using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using ThermoShenanigans: AbstractReferenceStateMethod, Sorting
import ThermoShenanigans: validate_reference_state_grid, reference_state_workspace, compute_reference_state!,
                          reference_tendency_workspace, compute_reference_tendency!

# A stand-in for a new reference state method, which reaches Sorting only through the method interface
struct DelegatedSorting <: AbstractReferenceStateMethod end
validate_reference_state_grid(::DelegatedSorting, grid) = validate_reference_state_grid(Sorting(), grid)
reference_state_workspace(::DelegatedSorting, grid) = reference_state_workspace(Sorting(), grid)
compute_reference_state!(z★, state, ::DelegatedSorting) = compute_reference_state!(z★, state, Sorting())
reference_tendency_workspace(::DelegatedSorting, grid) = reference_tendency_workspace(Sorting(), grid)
compute_reference_tendency!(Ṗ★, tendency, ::DelegatedSorting) = compute_reference_tendency!(Ṗ★, tendency, Sorting())

# Bytes allocated by evaluating `operation` in one cell, behind a function barrier so that the result is not boxed
function cell_allocations(operation)
    operation[3, 3, 3]
    return @allocated operation[3, 3, 3]
end

# The interior of a data array of the reference state at (LX, LY, LZ)
interior_data(data, LX, LY, LZ, grid) = Array(interior(Field{LX, LY, LZ}(grid; data)))

# ∇·j of face fluxes j = (x, y, z) on a regular, Bounded grid, from the interiors including the wall faces
function divergence(j, grid)
    Δx, Δy, Δz = xspacings(grid, Center())[1], yspacings(grid, Center())[1], zspacings(grid, Center())[1]
    jx, jy, jz = map(computed, (j.x, j.y, j.z))
    return diff(jx, dims = 1) ./ Δx .+ diff(jy, dims = 2) ./ Δy .+ diff(jz, dims = 3) ./ Δz
end

# δp / Δ at the faces of the Bounded direction `dims` of the cell-centred p, zero on the walls
function face_gradient(p, Δ, dims)
    wall = zeros(size(selectdim(p, dims, 1:1)))
    return cat(wall, diff(p; dims) ./ Δ, wall; dims)
end

# ℑ to the cell centres of values on the faces of direction `dims`, including the walls
center_average(a, dims) = (selectdim(a, dims, 1:size(a, dims)-1) .+ selectdim(a, dims, 2:size(a, dims))) ./ 2

# Whether the components of the face flux j normal to the walls vanish there
function zero_at_walls(j)
    jx, jy, jz = map(computed, (j.x, j.y, j.z))
    return all(iszero, jx[[1, end], :, :]) && all(iszero, jy[:, [1, end], :]) && all(iszero, jz[:, :, [1, end]])
end

@testset "Energetics" begin
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
        𝒫 = computed(thermodynamic_operation(model, potential_energy))
        ℐ = computed(thermodynamic_operation(model, HeightIndependent(potential_enthalpy)))
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
                                  thermodynamic_operation(model, HeightIndependent(potential_enthalpy))))
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

    @testset "APE budget [$label]" for (label, thermodynamics) in (("linear", linear), ("cabbeling", cabbeling))
        model = component_model(grid, thermodynamics)
        set_random_state!(model)
        z★ = reference_height(model)

        Υ, σ_𝒜, ε_𝒜, R★ = map(computed, (reference_displacement(z★), ape_production(model, z★), ape_dissipation(model, z★),
                                          reference_tendency_work(model, z★)))
        σ_b, Ḃ = computed(buoyancy_production(model)), computed(buoyancy_tendency(model))
        σ_𝒜_heating, σ_𝒜_nonlin = computed(heating_ape_production(model, z★)), computed(nonlinear_ape_production(model, z★))

        @test σ_𝒜 ≈ σ_𝒜_heating .+ σ_𝒜_nonlin
        label == "linear" && @test all(iszero, σ_𝒜_nonlin)

        # R★ integrates to zero, and summation by parts: Σ V (Υ σ_b - εₐ) = Σ V Υ (σ_b - ∇·j_b)
        @test abs(sum(R★ .* V)) < 1e-13 * sum(abs.(R★) .* V)
        @test sum((σ_𝒜 .- ε_𝒜) .* V) ≈ sum(Υ .* Ḃ .* V) rtol = 1e-10

        # Locally, - Υ ∇·j_b = - ∇·j_𝒜 - εₐ, with ∇·j_b = σ_b - Ḃ
        @test .- Υ .* (σ_b .- Ḃ) ≈ .- divergence(ape_flux(model, z★), grid) .- ε_𝒜 rtol = 1e-8

        # Σ V w b★ = 0 for a divergence-free velocity, so 𝒜 and 𝒫 convert the same total energy into kinetic energy
        wb_𝒜 = field(available_buoyancy_work(model, z★))
        Vᶠ = cell_volumes(wb_𝒜)
        wb = computed(buoyancy_work(model))
        @test abs(sum(interior_values(wb_𝒜) .* Vᶠ) - sum(wb .* Vᶠ)) < 1e-10 * sum(abs.(wb) .* Vᶠ)

        # ∇·(u 𝒜) integrates to zero, and the advection error 𝒩 integrates to Σ V z★ ADV_b, the spurious change of
        # the background potential energy, since Σ V z ADV_b = Σ V w b̃ with centred advection and b̃
        ADV_b, 𝒜_advection, 𝒩 = map(computed, (advective_buoyancy_tendency(model), ape_advection(model, z★),
                                                spurious_ape_production(model, z★)))
        z★_values = computed(z★)
        @test abs(sum(𝒜_advection .* V)) < 1e-13 * sum(abs.(𝒜_advection) .* V)
        @test sum(z .* ADV_b .* V) ≈ sum(wb .* Vᶠ) rtol = 1e-10
        @test sum(𝒩 .* V) ≈ sum(z★_values .* ADV_b .* V) rtol = 1e-8
    end

    @testset "APE and BPE budgets close [$label]" for (label, velocity) in (("at rest", 0), ("in motion", nothing))
        # ∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★ and
        # ∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ in every cell, and ∂p★/∂t = Ṗ★,
        # checked by finite differences along the model's tendencies of θ and S
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        isnothing(velocity) || set!(model, u = velocity, v = velocity, w = velocity)
        update_state!(model)
        compute_tendencies!(model, [])
        G = model.timestepper.Gⁿ
        Gθ, GS = Array(interior(G.θ)), Array(interior(G.S))

        z★ = reference_height(model)
        𝒜, ℬ = available_potential_energy(z★), background_potential_energy(model, z★)
        R★ = reference_tendency_work(model, z★)
        𝒜₀, ℬ₀ = computed(𝒜), computed(ℬ)
        conversion = center_average(computed(available_buoyancy_work(model, z★)), 3) # from the w-faces, including the walls
        σ_𝒜, ε_𝒜, R★_values = computed(ape_production(model, z★)), computed(ape_dissipation(model, z★)), computed(R★)
        budget = σ_𝒜 .- ε_𝒜 .- divergence(ape_flux(model, z★), grid) .+ computed(ape_advection(model, z★)) .- conversion .+
                 computed(spurious_ape_production(model, z★)) .+ R★_values
        bpe_budget = .- σ_𝒜 .+ ε_𝒜 .- divergence(bpe_flux(model, z★), grid) .+ computed(bpe_advection(model, z★)) .+
                     computed(viscous_dissipation(model)) .- R★_values .+ computed(spurious_bpe_production(model, z★))
        parcel_tendency = R★.arguments[2] # Ṗ★(z★), whose operand holds Ṗ★ at each level
        state = z★.operand
        reference_potentials() = (interior_data(state.reference_potential, Nothing, Nothing, Center, grid),
                                  interior_data(state.parcel_reference_potential, Center, Center, Center, grid))
        p★₀, parcel_p★₀ = reference_potentials()
        parcel_Ṗ★ = Array(interior(compute!(parcel_tendency)))
        Ṗ★ = interior_data(parcel_tendency.operand.reference_tendency, Nothing, Nothing, Center, grid)

        # A step short of the first time at which two cells swap order, over which z★ is constant
        b = vec(computed(KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid,
                                                                         model.buoyancy.formulation, model.tracers)))
        Ḃ = vec(computed(thermodynamic_operation(model, HeightIndependent(∂b∂θ))) .* Gθ .+
                computed(thermodynamic_operation(model, HeightIndependent(∂b∂S))) .* GS)
        order = sortperm(b)
        δt = min(1e-4, minimum(diff(b[order]) ./ abs.(diff(Ḃ[order]))) / 10)
        θ, S = Array(interior(model.tracers.θ)), Array(interior(model.tracers.S))
        set!(model, θ = θ .+ δt .* Gθ, S = S .+ δt .* GS)
        𝒜₁, ℬ₁ = computed(𝒜), computed(ℬ)
        p★₁, parcel_p★₁ = reference_potentials()

        scale = maximum(abs, budget)
        @info "APE and BPE budgets close [$label] with δt = $δt"
        @test maximum(abs, (𝒜₁ .- 𝒜₀) ./ δt .- budget) < 1e-4 * scale
        @test maximum(abs, (ℬ₁ .- ℬ₀) ./ δt .- bpe_budget) < 1e-4 * maximum(abs, bpe_budget)
        @test (p★₁ .- p★₀) ./ δt ≈ Ṗ★ rtol = 1e-4
        @test (parcel_p★₁ .- parcel_p★₀) ./ δt ≈ parcel_Ṗ★ rtol = 1e-4
    end

    @testset "KE budget in every cell" begin
        # ℑ(u·(G - ∇p_NHS)) = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε for a divergence-free velocity and any p_NHS, with the
        # model's tendencies G, which hold - ∇p_HY′ in u and v and neither b̃ nor - ∂z p_HY′ in w
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        update_state!(model)
        compute_tendencies!(model, [])
        G = model.timestepper.Gⁿ
        set!(model.pressures.pNHS, (x, y, z) -> randn())
        z★ = reference_height(model)

        Δ = (xspacings(grid, Center())[1], yspacings(grid, Center())[1], zspacings(grid, Center())[1])
        p = Array(interior(model.pressures.pNHS))
        velocities = map(name -> Array(interior(model.velocities[name])), (:u, :v, :w))
        tendencies = map(name -> Array(interior(G[name])), (:u, :v, :w))
        ke_tendency = sum(center_average(velocities[d] .* (tendencies[d] .- face_gradient(p, Δ[d], d)), d) for d in 1:3)

        A_𝒦 = computed(ke_advection(model))
        budget = A_𝒦 .- divergence(ke_flux(model, z★), grid) .+
                 center_average(computed(available_buoyancy_work(model, z★)), 3) .- computed(viscous_dissipation(model))
        @test ke_tendency ≈ budget rtol = 1e-12

        # Centred second-order advection only redistributes 𝒦; 𝒦 integrates to the energy at the velocity points
        @test abs(sum(A_𝒦 .* V)) < 1e-13 * sum(abs.(A_𝒦) .* V)
        @test sum(computed(kinetic_energy(model)) .* V) ≈ sum(sum(velocities[d] .^ 2 ./ 2) for d in 1:3) * V[1]
        @test zero_at_walls(ke_flux(model, z★))
    end

    @testset "PE budget splits into the APE and BPE budgets in every cell" begin
        # π G_θ + m G_S = ∂𝒜/∂t + ∂ℬ/∂t, with both from their budgets, and Σ V 𝒩_ℬ = - Σ V 𝒩
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        update_state!(model)
        compute_tendencies!(model, [])
        G = model.timestepper.Gⁿ
        z★ = reference_height(model)

        π_, m = computed(thermodynamic_operation(model, exner)), computed(thermodynamic_operation(model, chemical_potential_analogue))
        pe_tendency = π_ .* Array(interior(G.θ)) .+ m .* Array(interior(G.S))
        round_off = sum((abs.(π_ .* Array(interior(G.θ))) .+ abs.(m .* Array(interior(G.S)))) .* V)

        σ_𝒜, ε_𝒜, R★, 𝒩, 𝒩_ℬ = map(computed, (ape_production(model, z★), ape_dissipation(model, z★),
                                                reference_tendency_work(model, z★), spurious_ape_production(model, z★),
                                                spurious_bpe_production(model, z★)))
        conversion = center_average(computed(available_buoyancy_work(model, z★)), 3)
        ape_tendency = σ_𝒜 .- ε_𝒜 .- divergence(ape_flux(model, z★), grid) .+ computed(ape_advection(model, z★)) .-
                       conversion .+ 𝒩 .+ R★
        bpe_tendency = .- σ_𝒜 .+ ε_𝒜 .- divergence(bpe_flux(model, z★), grid) .+ computed(bpe_advection(model, z★)) .+
                       computed(viscous_dissipation(model)) .- R★ .+ 𝒩_ℬ

        @test pe_tendency ≈ ape_tendency .+ bpe_tendency rtol = 1e-12
        # The scale of Σ V (𝒩 + 𝒩_ℬ) = Σ V (π ADV_θ + m ADV_S + w b̃) is that of π ADV_θ ≈ cₚ ADV_θ, not of 𝒩
        @test abs(sum((𝒩 .+ 𝒩_ℬ) .* V)) < 1e-13 * round_off
        @test zero_at_walls(bpe_flux(model, z★))
    end

    @testset "Integrated Lorenz energy cycle" begin
        # Σ V u·G and Σ V (π G_θ + m G_S) from the model's tendencies equal the integrated KE budget and the sum of the
        # integrated APE and BPE budgets, in which R★ and the flux divergences integrate to zero
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        update_state!(model)
        compute_tendencies!(model, [])
        G = model.timestepper.Gⁿ
        u, v, w = model.velocities
        π_, m = field(thermodynamic_operation(model, exner)), field(thermodynamic_operation(model, chemical_potential_analogue))
        kinetic = integral(*, u, G.u) + integral(*, v, G.v) + integral(*, w, G.w)
        potential = integral(*, π_, G.θ) + integral(*, m, G.S)
        round_off = integral((a, b) -> abs(a * b), π_, G.θ) + integral((a, b) -> abs(a * b), m, G.S)

        z★ = reference_height(model)
        heating, cabbeling_production, ε_𝒜, wb_𝒜, ε, 𝒩, 𝒩_ℬ, A_𝒦 =
            map(integral, (heating_ape_production(model, z★), nonlinear_ape_production(model, z★), ape_dissipation(model, z★),
                           available_buoyancy_work(model, z★), viscous_dissipation(model), spurious_ape_production(model, z★),
                           spurious_bpe_production(model, z★), ke_advection(model)))

        d𝒦 = A_𝒦 + wb_𝒜 - ε                                         # 𝒜 → 𝒦 and 𝒦 → ℬ
        d𝒜 = heating + cabbeling_production - ε_𝒜 - wb_𝒜 + 𝒩         # ℬ → 𝒜, 𝒜 → ℬ and 𝒜 → 𝒦
        dℬ = - heating - cabbeling_production + ε_𝒜 + ε + 𝒩_ℬ
        conversions = sum(abs, (heating, cabbeling_production, ε_𝒜, wb_𝒜, ε, 𝒩))
        @info "Lorenz energy cycle: d∫𝒦/dt = $d𝒦, d∫𝒜/dt = $d𝒜, d∫ℬ/dt = $dℬ"

        @test cabbeling_production != 0
        @test abs(kinetic - d𝒦) < 1e-12 * conversions
        @test abs(potential - (d𝒜 + dℬ)) < 1e-13 * round_off
        @test abs(d𝒦 + d𝒜 + dℬ) < 1e-13 * round_off
    end

    @testset "APE budget of a stable, horizontally uniform column" begin
        model = component_model(grid, cabbeling)
        set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
        j_b = computed(buoyancy_flux(model).z)
        @test maximum(abs, computed(ape_dissipation(model, reference_height(model)))) < 1e-10 * maximum(abs, j_b)
    end

    @testset "Type-stable, allocation-free evaluation in a cell" begin
        # After Oceanostics.jl's performance invariants: an accidental boxing or `Any` shows up here
        model = component_model(grid, cabbeling)
        set_random_state!(model)
        z★ = reference_height(model)
        compute!(z★)
        operations = (available_potential_energy(z★), ape_production(model, z★), ape_dissipation(model, z★),
                      buoyancy_tendency(model), available_buoyancy_work(model, z★), ape_flux(model, z★).x,
                      advective_buoyancy_tendency(model), spurious_ape_production(model, z★), θ_production(model),
                      entropy_production(model), kinetic_energy(model), ke_advection(model), ke_flux(model, z★).x,
                      bpe_flux(model, z★).x, bpe_advection(model, z★), spurious_bpe_production(model, z★))
        for operation in operations
            @inferred operation[3, 3, 3]
            @test cell_allocations(operation) == 0
        end
    end

    @testset "Stretched grids are rejected" begin
        stretched = RectilinearGrid(size = (2, 2, 3), x = (0, 1), y = (0, 1), z = [-1, -0.5, -0.2, 0], topology = (Bounded, Bounded, Bounded))
        model = component_model(stretched, linear)
        @test_throws ArgumentError reference_height(model)
    end
end
