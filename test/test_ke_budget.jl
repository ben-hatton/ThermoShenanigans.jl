# δp / Δ at the faces of the Bounded direction `dims` of the cell-centred p, zero on the walls
function face_gradient(p, Δ, dims)
    wall = zeros(size(selectdim(p, dims, 1:1)))
    return cat(wall, diff(p; dims) ./ Δ, wall; dims)
end

function test_ke_budget()
    @testset "KE budget" begin
        grid = test_grid()
        V = cell_volumes(CenterField(grid))

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
    end
end
