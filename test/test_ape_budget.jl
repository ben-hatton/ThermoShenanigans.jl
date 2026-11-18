function test_ape_budget()
    @testset "APE budget" begin
        grid = test_grid()
        Nz = size(grid, 3)
        z = reshape(znodes(grid, Center()), 1, 1, Nz)
        V = cell_volumes(CenterField(grid))

        cases = (("linear", linear, component_model), ("cabbeling", cabbeling, component_model),
                 ("Fickian, linear", linear, fickian_model), ("Fickian, cabbeling", cabbeling, fickian_model),
                 ("Fickian, constant salinity", cabbeling, (grid, _) -> fickian_constant_salinity_model(grid)))

        @testset "APE budget [$label]" for (label, thermodynamics, build) in cases
            model = build(grid, thermodynamics)
            set_random_state!(model; S = haskey(model.tracers, :S))
            z★ = reference_height(model)

            Υ, σ_𝒜, ε_𝒜, R★ = map(computed, (reference_displacement(z★), ape_production(model, z★), ape_dissipation(model, z★),
                                              reference_tendency_work(model, z★)))
            σ_b, Ḃ = computed(buoyancy_production(model)), computed(buoyancy_tendency(model))
            σ_𝒜_heating, σ_𝒜_nonlin = computed(heating_ape_production(model, z★)), computed(nonlinear_ape_production(model, z★))

            @test σ_𝒜 ≈ σ_𝒜_heating .+ σ_𝒜_nonlin
            thermodynamics === linear && @test all(iszero, σ_𝒜_nonlin)
            # A Fickian model has no θ source
            model.closure isa ScalarDiffusivity && @test all(iszero, σ_𝒜_heating)

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
            # Both identities assume ∇·u = 0. The pressure solve leaves ∇·u at round-off, and the flux form ∇·(u θ) multiplies
            # it by θ itself, so the residual is measured against the sum of the absolute values of the terms, not the result
            @test abs(sum(z .* ADV_b .* V) - sum(wb .* Vᶠ)) <
                  2e-13 * (sum(abs.(z .* ADV_b) .* V) + sum(abs.(wb) .* Vᶠ))
            @test abs(sum(𝒩 .* V) - sum(z★_values .* ADV_b .* V)) <
                  2e-13 * (sum(abs.(𝒩) .* V) + sum(abs.(z★_values .* ADV_b) .* V))
        end

        #=
        The wall values of `buoyancy_flux_*` (b_θ₁ Q_θ + b_S₁ Q_S) and `ape_flux_*` (Υ₁ j_b·n), the face contractions of
        εₐ with no wall term (`face_ape_dissipation_*`), and Ḃ of `buoyancy_tendencyᶜᶜᶜ` and of R★ with the wall fluxes,
        for Flux conditions and, for the Fickian model, the closure's wall flux of its Value and Gradient conditions
        =#
        @testset "APE budget with boundary fluxes [$label]" for (label, build) in (("Flux conditions", flux_bc_model),
                                                                                  ("Fickian, wall conditions", fickian_wall_model))
            model = build(grid)
            set_random_state!(model)
            z★ = reference_height(model)

            Υ, σ_𝒜, ε_𝒜, R★ = map(computed, (reference_displacement(z★), ape_production(model, z★), ape_dissipation(model, z★),
                                              reference_tendency_work(model, z★)))
            σ_b, Ḃ = computed(buoyancy_production(model)), computed(buoyancy_tendency(model))
            ∇_dot_j_𝒜 = divergence(ape_flux(model, z★), grid)

            # R★ integrates to zero; locally, - Υ ∇·j_b = - ∇·j_𝒜 - εₐ, boundary cells included; and Σ V ∇·j_𝒜, the APE
            # input by the wall fluxes, closes Σ V (Υ σ_b - εₐ - ∇·j_𝒜) = Σ V Υ Ḃ
            @test abs(sum(R★ .* V)) < 1e-13 * sum(abs.(R★) .* V)
            @test .- Υ .* (σ_b .- Ḃ) ≈ .- ∇_dot_j_𝒜 .- ε_𝒜 rtol = 1e-8
            @test sum((σ_𝒜 .- ε_𝒜 .- ∇_dot_j_𝒜) .* V) ≈ sum(Υ .* Ḃ .* V) rtol = 1e-10
        end

        @testset "APE budget of a stable, horizontally uniform column" begin
            model = component_model(grid, cabbeling)
            set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
            j_b = computed(buoyancy_flux(model).z)
            @test maximum(abs, computed(ape_dissipation(model, reference_height(model)))) < 1e-10 * maximum(abs, j_b)

            # R★ of `reference_tendency_work`, with the wall fluxes in its Ḃ (`total_buoyancy_tendencyᶜᶜᶜ`), is zero in
            # a column whose cells are at their reference heights, with top, bottom and x-wall fluxes: zero to the
            # round-off of z★ - z, relative to the scale Lz max |Ḃ| of Ṗ★
            model = flux_bc_model(grid)
            set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
            R★ = computed(reference_tendency_work(model, reference_height(model)))
            @test maximum(abs, R★) < 1e-12 * grid.Lz * maximum(abs, computed(buoyancy_tendency(model)))
        end
    end
end
