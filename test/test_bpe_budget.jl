using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ
using ThermoShenanigans: ape_tendency, bpe_tendency, potential_energy_tendency

#=
∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★ and, with `bpe = true`,
∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ in every cell, and ∂p★/∂t = Ṗ★, checked by finite differences
along the model's tendencies of θ and S, which include the tendencies of its FluxBoundaryConditions
=#
function test_budgets_close(model, label; bpe = true)
    grid = model.grid
    update_state!(model)
    compute_tendencies!(model, [])

    z★ = reference_height(model)
    ape_tendency_values = computed(ape_tendency(model, z★))
    if bpe
        bpe_tendency_values = computed(bpe_tendency(model, z★))
        pe_tendency_values = computed(potential_energy_tendency(model))
    end
    compute_flux_bc_tendencies!(model)
    G = model.timestepper.Gⁿ
    Gθ, GS = Array(interior(G.θ)), Array(interior(G.S))
    𝒜 = available_potential_energy(z★)
    R★ = reference_tendency_work(model, z★)
    𝒜₀ = computed(𝒜)
    conversion = center_average(computed(available_buoyancy_work(model, z★)), 3) # from the w-faces, including the walls
    σ_𝒜, ε_𝒜, R★_values = computed(ape_production(model, z★)), computed(ape_dissipation(model, z★)), computed(R★)
    budget = σ_𝒜 .- ε_𝒜 .- divergence(ape_flux(model, z★), grid) .+ computed(ape_advection(model, z★)) .- conversion .+
             computed(spurious_ape_production(model, z★)) .+ R★_values
    if bpe
        ℬ = background_potential_energy(model, z★)
        ℬ₀ = computed(ℬ)
        bpe_budget = .- σ_𝒜 .+ ε_𝒜 .- divergence(bpe_flux(model, z★), grid) .+ computed(bpe_advection(model, z★)) .+
                     computed(viscous_dissipation(model)) .- R★_values .+ computed(spurious_bpe_production(model, z★))
    end
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
    Ḃ = vec(computed(thermodynamic_operation(model, ∂b∂θ)) .* Gθ .+
            computed(thermodynamic_operation(model, ∂b∂S)) .* GS)
    order = sortperm(b)
    δt = min(1e-4, minimum(diff(b[order]) ./ abs.(diff(Ḃ[order]))) / 10)
    θ, S = Array(interior(model.tracers.θ)), Array(interior(model.tracers.S))
    set!(model, θ = θ .+ δt .* Gθ, S = S .+ δt .* GS)
    𝒜₁ = computed(𝒜)
    p★₁, parcel_p★₁ = reference_potentials()

    @info "Budgets close [$label] with δt = $δt"
    @test maximum(abs, (𝒜₁ .- 𝒜₀) ./ δt .- budget) < 1e-4 * maximum(abs, budget)
    @test ape_tendency_values ≈ budget rtol = 1e-8
    if bpe
        ℬ₁ = computed(ℬ)
        @test maximum(abs, (ℬ₁ .- ℬ₀) ./ δt .- bpe_budget) < 1e-4 * maximum(abs, bpe_budget)
        # π ≈ cₚ in ∂ₜ𝒫 sets the round-off scale of ∂ₜℬ
        @test bpe_tendency_values ≈ bpe_budget atol = 1e-12 * maximum(abs, pe_tendency_values)
    end
    @test (p★₁ .- p★₀) ./ δt ≈ Ṗ★ rtol = 1e-4
    @test (parcel_p★₁ .- parcel_p★₀) ./ δt ≈ parcel_Ṗ★ rtol = 1e-4
end

function test_bpe_budget()
    @testset "BPE budget" begin
        grid = test_grid()
        V = cell_volumes(CenterField(grid))

        @testset "APE and BPE budgets close [$label]" for (label, velocity) in (("at rest", 0), ("in motion", nothing))
            model = component_model(grid, cabbeling)
            set_random_state!(model)
            isnothing(velocity) || set!(model, u = velocity, v = velocity, w = velocity)
            test_budgets_close(model, label)
        end

        @testset "APE budget closes for a Fickian model [$label, $(summary(thermodynamics))]" for
                thermodynamics in (linear, cabbeling), (label, velocity) in (("at rest", 0), ("in motion", nothing))
            model = fickian_model(grid, thermodynamics)
            set_random_state!(model)
            isnothing(velocity) || set!(model, u = velocity, v = velocity, w = velocity)
            test_budgets_close(model, "Fickian, $label"; bpe = false)
        end

        #=
        The wall fluxes in the diagnostic fluxes (`total_θ_flux_*`, `total_salt_flux_*`), the wall values of
        `buoyancy_flux_*`, `ape_flux_*` and `bpe_flux_*`, and Ḃ of R★ and Ṗ★ (`total_buoyancy_tendencyᶜᶜᶜ`) with them:
        the budgets of 𝒜 and ℬ and the reference potential follow the model's tendencies, including those of its wall
        fluxes. For the Fickian model, the closure's wall flux (`closure_wall_flux_*`) of its Value and Gradient
        conditions as well.
        =#
        @testset "Budgets close with boundary fluxes [$label, $motion]" for
                (label, build, bpe) in (("Flux conditions", flux_bc_model, true), ("Fickian, wall conditions", fickian_wall_model, false)),
                (motion, velocity) in (("at rest", 0), ("in motion", nothing))
            model = build(grid)
            set_random_state!(model)
            isnothing(velocity) || set!(model, u = velocity, v = velocity, w = velocity)
            test_budgets_close(model, "$label, $motion"; bpe)
        end

        @testset "Equal heat fluxes at the top and bottom" begin
            model = equal_heat_flux_model(grid)
            set_random_state!(model)
            model.clock.time = 1 # f(t) = 2, so that the diagnostics must read the model's clock

            # The model `equal_heat_flux_model`, Q_θ = Q_H f(t) / (ρ₀ π₁): no net energy input, as the README states
            dEdt, round_off, _ = energy_tendency(model)
            @test abs(dEdt) < 1e-13 * round_off

            # The field- and time-dependent wall fluxes of `vertical_boundary_flux`, evaluated with the clock and the
            # field data of `wall_conditions(model)`, in the budgets
            test_budgets_close(model, "equal heat fluxes")
        end

        @testset "Fickian model refuses the consistent-only diagnostics" begin
            model = fickian_model(grid, cabbeling)
            z★ = reference_height(model)
            test_refused(background_potential_energy, () -> background_potential_energy(model, z★))
            test_refused(bpe_flux, () -> bpe_flux(model, z★))
            test_refused(bpe_advection, () -> bpe_advection(model, z★))
            test_refused(spurious_bpe_production, () -> spurious_bpe_production(model, z★))
            test_refused(bpe_tendency, () -> bpe_tendency(model, z★))
        end

        @testset "𝒜 + 𝒢★ = - b (z - z₀) in every cell" begin
            # For a Fickian model z₀ only sets the origin of z; `shifted` has z₀ = -40
            model = fickian_model(grid, shifted)
            set_random_state!(model)
            z★ = reference_height(model)
            𝒜, 𝒢★ = computed(available_potential_energy(z★)), computed(background_gravitational_energy(model, z★))
            b = computed(KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid,
                                                                         model.buoyancy.formulation, model.tracers))
            z = reshape(znodes(grid, Center()), 1, 1, size(grid, 3))
            @test 𝒜 .+ 𝒢★ ≈ .- b .* (z .- reference_pressure_height(shifted))
        end

        @testset "PE budget splits into the APE and BPE budgets in every cell [$label]" for
                (label, build) in (("no flux", grid -> component_model(grid, cabbeling)), ("Flux conditions", flux_bc_model))
            # π G_θ + m G_S = ∂𝒜/∂t + ∂ℬ/∂t, with both from their budgets, and Σ V 𝒩_ℬ = - Σ V 𝒩. With Flux conditions,
            # this checks the wall values of `ape_flux_*` and `bpe_flux_*` against the model's wall-flux tendencies.
            model = build(grid)
            set_random_state!(model)
            update_state!(model)
            compute_tendencies!(model, [])
            z★ = reference_height(model)
            pe_tendency_values = computed(potential_energy_tendency(model))
            ape_tendency_values = computed(ape_tendency(model, z★))
            bpe_tendency_values = computed(bpe_tendency(model, z★))
            compute_flux_bc_tendencies!(model)
            G = model.timestepper.Gⁿ

            π_, m = computed(thermodynamic_operation(model, exner)), computed(thermodynamic_operation(model, chemical_potential_analogue))
            pe_tendency = π_ .* Array(interior(G.θ)) .+ m .* Array(interior(G.S))
            round_off = sum((abs.(π_ .* Array(interior(G.θ))) .+ abs.(m .* Array(interior(G.S)))) .* V)

            σ_𝒜, ε_𝒜, R★, 𝒩, 𝒩_ℬ = map(computed, (ape_production(model, z★), ape_dissipation(model, z★),
                                                    reference_tendency_work(model, z★), spurious_ape_production(model, z★),
                                                    spurious_bpe_production(model, z★)))
            conversion = center_average(computed(available_buoyancy_work(model, z★)), 3)
            ape_budget = σ_𝒜 .- ε_𝒜 .- divergence(ape_flux(model, z★), grid) .+ computed(ape_advection(model, z★)) .-
                           conversion .+ 𝒩 .+ R★
            bpe_budget = .- σ_𝒜 .+ ε_𝒜 .- divergence(bpe_flux(model, z★), grid) .+ computed(bpe_advection(model, z★)) .+
                           computed(viscous_dissipation(model)) .- R★ .+ 𝒩_ℬ

            @test pe_tendency ≈ ape_budget .+ bpe_budget rtol = 1e-12
            @test pe_tendency_values ≈ pe_tendency rtol = 1e-12
            # π ≈ cₚ in ∂ₜ𝒫 sets the round-off scale
            @test pe_tendency_values ≈ ape_tendency_values .+ bpe_tendency_values atol = 1e-12 * maximum(abs, pe_tendency_values)
            # The scale of Σ V (𝒩 + 𝒩_ℬ) = Σ V (π ADV_θ + m ADV_S + w b̃) is that of π ADV_θ ≈ cₚ ADV_θ, not of 𝒩
            @test abs(sum((𝒩 .+ 𝒩_ℬ) .* V)) < 1e-13 * round_off
            label == "no flux" && @test zero_at_walls(bpe_flux(model, z★))
        end

        @testset "Wall values of bpe_flux" begin
            #=
            `bpe_flux_{x,z}` at the walls, from `potential_energy_flux_*` and `ape_flux_*` with `boundary_cell_value_*`
            and the wall fluxes `wall_θ_flux_*` and `wall_salt_flux_*`: (π₁ - Υ₁ b_θ₁) Q_θ + (m₁ - Υ₁ b_S₁) Q_S, which
            equals π(θ₁, S₁, z★₁) Q_θ + m(θ₁, S₁, z★₁) Q_S since b does not depend on z
            =#
            model = flux_bc_model(grid)
            set_random_state!(model)
            update_state!(model)
            z★ = reference_height(model)
            j_ℬ = bpe_flux(model, z★)

            π_, m, b_θ, b_S = map(f -> computed(thermodynamic_operation(model, f)), (exner, chemical_potential_analogue, ∂b∂θ, ∂b∂S))
            Υ, z★_values = computed(reference_displacement(z★)), computed(z★)
            θ, S = Array(interior(model.tracers.θ)), Array(interior(model.tracers.S))
            general_form = (π_ .- Υ .* b_θ, m .- Υ .* b_S)
            reference_form = (exner.(θ, S, z★_values, Ref(cabbeling)), chemical_potential_analogue.(θ, S, z★_values, Ref(cabbeling)))
            boundary_cell_input((c, d), dims, index, wall) =
                selectdim(c, dims, index) .* flux_bc_values.θ[wall] .+ selectdim(d, dims, index) .* flux_bc_values.S[wall]

            Nx, Nz = size(grid, 1), size(grid, 3)
            @testset "$wall" for (component, dims, wall, index) in ((:x, 1, :west, 1), (:x, 1, :east, Nx),
                                                                    (:z, 3, :bottom, 1), (:z, 3, :top, Nz))
                j_values = computed(j_ℬ[component])
                face = index == 1 ? 1 : index + 1
                @test selectdim(j_values, dims, face) ≈ boundary_cell_input(general_form, dims, index, wall) rtol = 1e-12
                @test selectdim(j_values, dims, face) ≈ boundary_cell_input(reference_form, dims, index, wall) rtol = 1e-12
            end
        end

        @testset "Integrated Lorenz energy cycle [$label]" for
                (label, build, Q) in (("no flux", grid -> component_model(grid, cabbeling), map(Q -> map(zero, Q), flux_bc_values)),
                                      ("Flux conditions", flux_bc_model, flux_bc_values))
            #=
            Σ V u·G and Σ V (π G_θ + m G_S) from the model's tendencies equal the integrated KE budget and the sum of the
            integrated APE and BPE budgets, in which R★ integrates to zero and the flux divergences to the inputs through
            the walls: Σ_walls A Υ₁ j_b·n for 𝒜, and Σ_walls A (π₁ Q_θ + m₁ Q_S) for 𝒫, with inward normals, zero
            without wall fluxes. With Flux conditions, this checks the wall values of `ape_flux_*` (Υ₁ j_b·n, with
            j_b·n = b_θ₁ Q_θ + b_S₁ Q_S of `wall_buoyancy_flux_*`) and of `bpe_flux_*` (j_𝒫·n - j_𝒜·n).
            =#
            model = build(grid)
            set_random_state!(model)
            update_state!(model)
            compute_tendencies!(model, [])
            compute_flux_bc_tendencies!(model)
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

            ∇_dot_j_𝒜, ∇_dot_j_ℬ = divergence(ape_flux(model, z★), grid), divergence(bpe_flux(model, z★), grid)
            ape_input, bpe_input = - sum(∇_dot_j_𝒜 .* V), - sum(∇_dot_j_ℬ .* V)
            Υ, b_θ, b_S = computed(reference_displacement(z★)), computed(thermodynamic_operation(model, ∂b∂θ)),
                          computed(thermodynamic_operation(model, ∂b∂S))
            wall_ape_input = wall_input(Υ .* b_θ, Υ .* b_S, grid; Q)
            wall_pe_input = wall_input(interior_values(π_), interior_values(m), grid; Q)

            d𝒦 = A_𝒦 + wb_𝒜 - ε                                                 # 𝒜 → 𝒦 and 𝒦 → ℬ
            d𝒜 = heating + cabbeling_production - ε_𝒜 - wb_𝒜 + 𝒩 + ape_input     # ℬ → 𝒜, 𝒜 → ℬ and 𝒜 → 𝒦
            dℬ = - heating - cabbeling_production + ε_𝒜 + ε + 𝒩_ℬ + bpe_input
            conversions = sum(abs, (heating, cabbeling_production, ε_𝒜, wb_𝒜, ε, 𝒩))
            @info "Lorenz energy cycle [$label]: d∫𝒦/dt = $d𝒦, d∫𝒜/dt = $d𝒜, d∫ℬ/dt = $dℬ, wall inputs to 𝒜 and 𝒫: " *
                  "$wall_ape_input, $wall_pe_input"

            @test cabbeling_production != 0
            @test abs(ape_input - wall_ape_input) < 1e-12 * sum(abs.(∇_dot_j_𝒜) .* V)
            @test abs(bpe_input - (wall_pe_input - wall_ape_input)) < 1e-12 * sum(abs.(∇_dot_j_ℬ) .* V)
            @test abs(kinetic - d𝒦) < 1e-12 * conversions
            @test abs(potential - (d𝒜 + dℬ)) < 1e-13 * round_off
            @test abs(d𝒦 + d𝒜 + dℬ - wall_pe_input) < 1e-13 * round_off
        end

        #=
        After Oceanostics.jl's performance invariants: an accidental boxing or `Any` shows up here. Every exported
        diagnostic of both closures is checked in Float64 and in Float32, as CLAUDE.md requires of every public
        diagnostic on the CPU. The model with Flux conditions checks the wall fluxes of `wall_conditions(model)` in the
        diagnostic kernels (`lateral_boundary_flux`, `vertical_boundary_flux`, `boundary_cell_value_*`).
        =#
        @testset "Type-stable, allocation-free evaluation in a cell [$label, $FT]" for
                (label, build, refused) in (("consistent", component_model, ()), ("Fickian", fickian_model, fickian_refused),
                                            ("Flux conditions", flux_bc_model, ())),
                FT in (Float64, Float32)
            model = build(FT == Float64 ? test_grid() : architecture_grid(CPU(), FT), architecture_thermodynamics(FT))
            set_random_state!(model)
            diagnostics = exported_diagnostics(model; refused)
            z★ = last(first(diagnostics))
            compute!(z★)

            @testset "$diagnostic_label" for (diagnostic_label, operation) in diagnostics
                i, j, k = cell_indices(operation)
                @inferred operation[i, j, k]
                @test cell_allocations(operation) == 0
            end
        end
    end
end
