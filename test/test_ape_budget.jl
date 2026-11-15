function test_ape_budget()
    @testset "APE budget" begin
        grid = test_grid()
        Nz = size(grid, 3)
        z = reshape(znodes(grid, Center()), 1, 1, Nz)
        V = cell_volumes(CenterField(grid))

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

        @testset "APE budget of a stable, horizontally uniform column" begin
            model = component_model(grid, cabbeling)
            set!(model, θ = (x, y, z) -> 283 + 2z, S = 0.035)
            j_b = computed(buoyancy_flux(model).z)
            @test maximum(abs, computed(ape_dissipation(model, reference_height(model)))) < 1e-10 * maximum(abs, j_b)
        end
    end
end
