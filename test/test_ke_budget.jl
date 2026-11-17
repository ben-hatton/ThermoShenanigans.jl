using Oceananigans.BoundaryConditions: fill_halo_regions!
using ThermoShenanigans: ke_tendency

# δp / Δ at the faces of the Bounded direction `dims` of the cell-centred p, zero on the walls
function face_gradient(p, Δ, dims)
    wall = zeros(size(selectdim(p, dims, 1:1)))
    return cat(wall, diff(p; dims) ./ Δ, wall; dims)
end

function test_ke_budget()
    @testset "KE budget" begin
        grid = test_grid()
        V = cell_volumes(CenterField(grid))

        @testset "KE budget in every cell [$label]" for (label, build) in (("consistent", component_model), ("Fickian", fickian_model))
            # ℑ(u·(G - ∇p_NHS)) = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε for a divergence-free velocity and any p_NHS, with the
            # model's tendencies G, which hold - ∇p_HY′ in u and v and neither b̃ nor - ∂z p_HY′ in w
            model = build(grid, cabbeling)
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
            expected_ke_tendency = sum(center_average(velocities[d] .* (tendencies[d] .- face_gradient(p, Δ[d], d)), d) for d in 1:3)

            A_𝒦 = computed(ke_advection(model))
            budget = A_𝒦 .- divergence(ke_flux(model, z★), grid) .+
                     center_average(computed(available_buoyancy_work(model, z★)), 3) .- computed(viscous_dissipation(model))
            @test computed(ke_tendency(model)) ≈ expected_ke_tendency rtol = 1e-12
            @test computed(ke_tendency(model)) ≈ budget rtol = 1e-12

            # Centred second-order advection only redistributes 𝒦; 𝒦 integrates to the energy at the velocity points
            @test abs(sum(A_𝒦 .* V)) < 1e-13 * sum(abs.(A_𝒦) .* V)
            @test sum(computed(kinetic_energy(model)) .* V) ≈ sum(sum(velocities[d] .^ 2 ./ 2) for d in 1:3) * V[1]
            @test zero_at_walls(ke_flux(model, z★))
        end

        #=
        After a time step the halos of p_NHS hold p·Δτ, not p. The diagnostics that read p_NHS must give the same values
        as after a halo fill of p_NHS. This is the only periodic grid of the tests, since only a periodic direction reads
        the halos of p_NHS: at a wall the flux j_𝒦 and the pressure work vanish.
        =#
        @testset "p_NHS is read with filled halos" begin
            periodic_grid = RectilinearGrid(size = (6, 6, 6), extent = (1, 1, 1), topology = (Periodic, Bounded, Bounded))
            model = component_model(periodic_grid, cabbeling)
            set_random_state!(model)
            time_step!(model, 1e-3)
            pNHS = model.pressures.pNHS
            Nx, Ny, Nz = size(periodic_grid)

            # The x halo of p_NHS is not the periodic copy of the interior, so that the test is not vacuous
            @test Array(pNHS.data[0:0, 1:Ny, 1:Nz]) != Array(pNHS.data[Nx:Nx, 1:Ny, 1:Nz])

            z★ = reference_height(model)
            ∂ₜ𝒦 = computed(ke_tendency(model))
            ke_flux_x = computed(ke_flux(model, z★).x)

            fill_halo_regions!(pNHS)
            @test computed(ke_tendency(model)) ≈ ∂ₜ𝒦 rtol = 1e-14
            @test computed(ke_flux(model, z★).x) ≈ ke_flux_x rtol = 1e-14
        end

        #=
        ∂ₜ𝒦 against the change of 𝒦 over one time step, after Oceanostics v0.21.7
        (`test/test_kinetic_energy_equation.jl:381-415`). After the warm-up step, p_NHS holds the pressure of its last
        substep.
        =#
        @testset "∂ₜ𝒦 over one time step" begin
            model = component_model(test_grid(), cabbeling)
            set_random_state!(model)
            Δt = 1e-5
            time_step!(model, Δt)

            𝒦ⁿ, ∂ₜ𝒦 = computed(kinetic_energy(model)), computed(ke_tendency(model))
            time_step!(model, Δt)
            𝒦ⁿ⁺¹ = computed(kinetic_energy(model))

            @test maximum(abs, (𝒦ⁿ⁺¹ .- 𝒦ⁿ) ./ Δt .- ∂ₜ𝒦) < 1e-3 * maximum(abs, ∂ₜ𝒦)
        end
    end
end
