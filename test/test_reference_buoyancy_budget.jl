function test_reference_buoyancy_budget()
    #=
    The first moment Σ V z★ f = z_top Σ V f - A Σₖ Δz 𝓘[f]ₖ of a cell field f, with 𝓘[f] the isopycnal integral that a
    diagnostic computes, holds exactly when no two cells share a buoyancy, as in a random state. It checks that each
    diagnostic integrates its own part of Ḃ, with its sign.
    =#
    @testset "Isopycnal integrals [$label]" for (label, build) in (("Flux conditions", flux_bc_model),
                                                                ("Fickian wall conditions", fickian_wall_model))
        model = build(test_grid())
        set_random_state!(model)
        grid = model.grid
        z★ = reference_height(model)
        compute!(z★)
        z★_values, V = interior_values(z★), cell_volumes(z★)
        Δx, Δy, Δz = xspacings(grid, Center())[1], yspacings(grid, Center())[1], zspacings(grid, Center())[1]
        A = grid.Lx * grid.Ly
        z_top = znodes(grid, Face())[end]

        # ∇·j of the face arrays of the buoyancy flux, with the wall faces or the interior faces set to zero
        jx, jy, jz = map(computed, buoyancy_flux(model))
        function flux_divergence_values(zero_faces)
            jx_part, jy_part, jz_part = copy(jx), copy(jy), copy(jz)
            zero_faces(jx_part, 1)
            zero_faces(jy_part, 2)
            zero_faces(jz_part, 3)
            return diff(jx_part, dims = 1) ./ Δx .+ diff(jy_part, dims = 2) ./ Δy .+ diff(jz_part, dims = 3) ./ Δz
        end
        zero_interior_faces(j, dims) = selectdim(j, dims, 2:size(j, dims)-1) .= 0
        zero_wall_faces(j, dims) = selectdim(j, dims, [1, size(j, dims)]) .= 0

        σ_b = computed(buoyancy_production(model))
        ADV_b = computed(advective_buoyancy_tendency(model))
        cell_fields = (isopycnal_buoyancy_production           => σ_b,
                       isopycnal_buoyancy_flux_divergence      => divergence(buoyancy_flux(model), grid),
                       isopycnal_advective_buoyancy_tendency   => ADV_b,
                       isopycnal_wall_buoyancy_flux_divergence => flux_divergence_values(zero_interior_faces),
                       effective_diapycnal_diffusivity         => flux_divergence_values(zero_wall_faces),
                       numerical_diapycnal_diffusivity         => .- ADV_b)

        @testset "$(nameof(diagnostic))" for (diagnostic, f) in cell_fields
            operation = diagnostic(model, z★)
            field(operation)
            # `arguments[1]` is the reference tendency passed to `with_reference_state`
            isopycnal_integral = interior_data(operation.arguments[1].operand.reference_tendency, Nothing, Nothing, Center, grid)
            @test maximum(abs, f) > 0
            first_moment = sum(V .* z★_values .* f)
            @test abs(first_moment - (z_top * sum(V .* f) - A * Δz * sum(isopycnal_integral))) <
                  1e-11 * sum(V .* abs.(z★_values .* f))
        end
    end

    #=
    All cells of a level share a buoyancy, so that the bands are the levels. Then 𝓘[∇·j_κ]ₖ = (jₖ + jₖ₊₁)/2 and
    ∂z b★ₖ = (bₖ₊₁ - bₖ₋₁)/(2Δz), one-sided at the boundary levels, where j = 0 on the walls. Minus their ratio is κ_θ
    at every level for j = - κ_θ ∂z b with a linear equation of state. The other four diagnostics are zero because
    u = 0, ∂b/∂θ is constant and there are no wall fluxes.
    =#
    @testset "Diffusivities of a stratified column at rest [Fickian]" begin
        κ_θ = 1e-3
        # `linear` and `cabbeling` share a type, so this compiles no new model
        model = fickian_model(test_grid(), linear; κ_θ)
        set!(model, θ = (x, y, z) -> 283 + 2z + z^2 / 2, S = 0.035)
        grid = model.grid
        Δz = zspacings(grid, Center())[1]
        z★ = reference_height(model)
        b★ = interior_values(compute!(reference_buoyancy(z★)))
        ∂zb★ = diff(b★, dims = 3) ./ Δz

        @test all(K_eff -> isapprox(K_eff, κ_θ, rtol = 1e-10), computed(effective_diapycnal_diffusivity(model, z★)))
        @test computed(isopycnal_buoyancy_flux_divergence(model, z★))[:, :, 2:end-1] ≈
              - κ_θ .* diff(∂zb★, dims = 3) ./ Δz rtol = 1e-10
        @testset "$(nameof(diagnostic)) is zero" for diagnostic in (numerical_diapycnal_diffusivity,
                                                                    isopycnal_buoyancy_production,
                                                                    isopycnal_advective_buoyancy_tendency,
                                                                    isopycnal_wall_buoyancy_flux_divergence)
            @test all(iszero, computed(diagnostic(model, z★)))
        end
    end
end
