using Oceananigans.BuoyancyFormulations: buoyancy_perturbationᶜᶜᶜ

#=
∂𝒜/∂t = Υ σ_b - εₐ - ∇·j_𝒜 - ∇·(u 𝒜) - ℑz[w (b̃ - b★)] + 𝒩 + R★ and, with `bpe = true`,
∂ℬ/∂t = - Υ σ_b + εₐ - ∇·j_ℬ - ∇·(u ℬ) + ε - R★ + 𝒩_ℬ in every cell, and ∂p★/∂t = Ṗ★, checked by finite differences
along the model's tendencies of θ and S
=#
function test_budgets_close(model, label; bpe = true)
    grid = model.grid
    update_state!(model)
    compute_tendencies!(model, [])
    G = model.timestepper.Gⁿ
    Gθ, GS = Array(interior(G.θ)), Array(interior(G.S))

    z★ = reference_height(model)
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
    if bpe
        ℬ₁ = computed(ℬ)
        @test maximum(abs, (ℬ₁ .- ℬ₀) ./ δt .- bpe_budget) < 1e-4 * maximum(abs, bpe_budget)
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

        @testset "Fickian model refuses the consistent-only diagnostics" begin
            model = fickian_model(grid, cabbeling)
            z★ = reference_height(model)
            test_refused(background_potential_energy, () -> background_potential_energy(model, z★))
            test_refused(bpe_flux, () -> bpe_flux(model, z★))
            test_refused(bpe_advection, () -> bpe_advection(model, z★))
            test_refused(spurious_bpe_production, () -> spurious_bpe_production(model, z★))
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

        #=
        After Oceanostics.jl's performance invariants: an accidental boxing or `Any` shows up here. Every exported
        diagnostic of both closures is checked in Float64 and in Float32, as CLAUDE.md requires of every public
        diagnostic on the CPU.
        =#
        @testset "Type-stable, allocation-free evaluation in a cell [$label, $FT]" for
                (label, build, refused) in (("consistent", component_model, ()), ("Fickian", fickian_model, fickian_refused)),
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
