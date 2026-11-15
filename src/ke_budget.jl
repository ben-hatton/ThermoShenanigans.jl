#=
Local kinetic energy budget. With 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2 at the cell centres, the model's momentum equation
∂u/∂t = ADV_u - ∇p + b̃ 𝐤 - ∇·F, with p = p_NHS + p_HY′ and F the viscous fluxes of Oceananigans (F = - τ), gives

    ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε,    j_𝒦 = (p - p★) u + u·F,

exactly in every cell for a divergence-free velocity on a regular grid:

- A_𝒦 = ℑ(u·ADV_u) is the advective tendency of 𝒦. For centred second-order advection it is exactly the divergence
  of a discrete flux of 𝒦: in each direction u_{i+½} δ(ū ū) = δ(ū u₋ u₊ / 2) + u² δū / 2, and the last terms sum to
  u² ∇·u / 2 = 0. A_𝒦 then integrates to zero; for other schemes its integral is their numerical dissipation.
- The pressure and viscous terms follow by summation by parts, ℑ(u δF) = δ(u ℑF) - F δu, whose last term is the
  dissipation ε of `viscous_dissipation`, formed in the same way at the native locations of the viscous fluxes.
- The w equation holds the buoyancy work in the hydrostatic pressure anomaly, δz p_HY′ = b̃ Δz, and
  ∇·(p★ u) = ℑz(w b★) for a divergence-free velocity, with p★ at the u- and v-faces taken at their level and
  interpolated to the w-faces. So the pressure work becomes the conversion ℑz[w (b̃ - b★)] of 𝒜 into 𝒦.

The budget is ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε, with the conversion ℑz[w (b̃ - b★)] of
`available_buoyancy_work` and the dissipation ε of `viscous_dissipation`. Forcing of the velocities, Coriolis and
background fields are not included.
=#

# 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2
@inline kinetic_energyᶜᶜᶜ(i, j, k, grid, velocities) =
    ℑxᶜᵃᵃ(i, j, k, grid, half_ϕ², velocities.u) + ℑyᵃᶜᵃ(i, j, k, grid, half_ϕ², velocities.v) +
    ℑzᵃᵃᶜ(i, j, k, grid, half_ϕ², velocities.w)

"""
$(TYPEDSIGNATURES)

Return the kinetic energy 𝒦 = (ℑx u² + ℑy v² + ℑz w²) / 2 of `model` at (Center, Center, Center). On a regular grid,
its integral is that of the kinetic energy of the velocities at their own locations.
"""
kinetic_energy(model) = center_operation(kinetic_energyᶜᶜᶜ, model.grid, field_data(model.velocities))

# u ADV_u, v ADV_v and w ADV_w at the velocity points, with the model's momentum advection scheme
@inline u_advective_workᶠᶜᶜ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.u[i, j, k] * div_𝐯u(i, j, k, grid, advection, velocities, velocities.u)
@inline v_advective_workᶜᶠᶜ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.v[i, j, k] * div_𝐯v(i, j, k, grid, advection, velocities, velocities.v)
@inline w_advective_workᶜᶜᶠ(i, j, k, grid, advection, velocities) =
    @inbounds - velocities.w[i, j, k] * div_𝐯w(i, j, k, grid, advection, velocities, velocities.w)

# A_𝒦 = ℑ(u·ADV_u)
@inline ke_advectionᶜᶜᶜ(i, j, k, grid, advection, velocities) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_advective_workᶠᶜᶜ, advection, velocities) +
    ℑyᵃᶜᵃ(i, j, k, grid, v_advective_workᶜᶠᶜ, advection, velocities) +
    ℑzᵃᵃᶜ(i, j, k, grid, w_advective_workᶜᶜᶠ, advection, velocities)

"""
$(TYPEDSIGNATURES)

Return the advective tendency A_𝒦 = ℑ(u·ADV_u) of the kinetic energy of `model`, with ADV_u the model's momentum
advection, at (Center, Center, Center). For a divergence-free velocity and Oceananigans' default `Centered(order = 2)`
advection, A_𝒦 is exactly the divergence of a discrete flux of 𝒦 in every cell, and integrates to zero; for other
schemes, its integral is their numerical dissipation of kinetic energy.
"""
ke_advection(model) =
    center_operation(ke_advectionᶜᶜᶜ, model.grid, model.advection.momentum, field_data(model.velocities))

# p = p_NHS + p_HY′, where the hydrostatic pressure anomaly may be absent
@inline pressureᶜᶜᶜ(i, j, k, grid, nonhydrostatic_pressure, hydrostatic_pressure) =
    @inbounds nonhydrostatic_pressure[i, j, k] + hydrostatic_pressure[i, j, k]
@inline pressureᶜᶜᶜ(i, j, k, grid, nonhydrostatic_pressure, ::Nothing) = @inbounds nonhydrostatic_pressure[i, j, k]

# p - p★
@inline pressure_anomalyᶜᶜᶜ(i, j, k, grid, pressures, reference_potential) =
    pressureᶜᶜᶜ(i, j, k, grid, pressures...) - reference_potentialᶜᶜᶜ(i, j, k, grid, reference_potential)

# uᵢ Fᵢⱼ at the native location of the viscous flux Fᵢⱼ, with uᵢ interpolated there; the arguments after `grid` are those
# of the viscous fluxes of Oceananigans, as in `viscous_dissipation_rateᶜᶜᶜ`
@inline u_F₁₁ᶠᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    @inbounds fields.u[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, viscous_flux_ux, closure, K, clock, fields, b)
@inline v_F₂₁ᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑxᶠᵃᵃ(i, j, k, grid, fields.v) * viscous_flux_vx(i, j, k, grid, closure, K, clock, fields, b)
@inline w_F₃₁ᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑxᶠᵃᵃ(i, j, k, grid, fields.w) * viscous_flux_wx(i, j, k, grid, closure, K, clock, fields, b)
@inline u_F₁₂ᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑyᵃᶠᵃ(i, j, k, grid, fields.u) * viscous_flux_uy(i, j, k, grid, closure, K, clock, fields, b)
@inline v_F₂₂ᶜᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) =
    @inbounds fields.v[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, viscous_flux_vy, closure, K, clock, fields, b)
@inline w_F₃₂ᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑyᵃᶠᵃ(i, j, k, grid, fields.w) * viscous_flux_wy(i, j, k, grid, closure, K, clock, fields, b)
@inline u_F₁₃ᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑzᵃᵃᶠ(i, j, k, grid, fields.u) * viscous_flux_uz(i, j, k, grid, closure, K, clock, fields, b)
@inline v_F₂₃ᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, b) =
    ℑzᵃᵃᶠ(i, j, k, grid, fields.v) * viscous_flux_vz(i, j, k, grid, closure, K, clock, fields, b)
@inline w_F₃₃ᶜᶜᶠ(i, j, k, grid, closure, K, clock, fields, b) =
    @inbounds fields.w[i, j, k] * ℑzᵃᵃᶠ(i, j, k, grid, viscous_flux_wz, closure, K, clock, fields, b)

# u·F, the viscous flux of 𝒦 (u·F = - u·τ), at the faces
@inline viscous_ke_flux_x(i, j, k, grid, closure, K, clock, fields, b) =
    u_F₁₁ᶠᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) +
    ℑyᵃᶜᵃ(i, j, k, grid, v_F₂₁ᶠᶠᶜ, closure, K, clock, fields, b) +
    ℑzᵃᵃᶜ(i, j, k, grid, w_F₃₁ᶠᶜᶠ, closure, K, clock, fields, b)
@inline viscous_ke_flux_y(i, j, k, grid, closure, K, clock, fields, b) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_F₁₂ᶠᶠᶜ, closure, K, clock, fields, b) +
    v_F₂₂ᶜᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) +
    ℑzᵃᵃᶜ(i, j, k, grid, w_F₃₂ᶜᶠᶠ, closure, K, clock, fields, b)
@inline viscous_ke_flux_z(i, j, k, grid, closure, K, clock, fields, b) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_F₁₃ᶠᶜᶠ, closure, K, clock, fields, b) +
    ℑyᵃᶜᵃ(i, j, k, grid, v_F₂₃ᶜᶠᶠ, closure, K, clock, fields, b) +
    w_F₃₃ᶜᶜᶠ(i, j, k, grid, closure, K, clock, fields, b)

# Generated: unmasked_ke_flux_{x,y,z} and ke_flux_{x,y,z}, the components of j_𝒦 at faces
for (dir, average_to_face, velocity_component, wall_mask) in ((:x, ℑxᶠᵃᵃ, :u, wall_mask_x),
                                                              (:y, ℑyᵃᶠᵃ, :v, wall_mask_y),
                                                              (:z, ℑzᵃᵃᶠ, :w, wall_mask_z))
    viscous_ke_flux_xyz  = Symbol(:viscous_ke_flux_, dir)
    unmasked_ke_flux_xyz = Symbol(:unmasked_ke_flux_, dir)
    ke_flux_xyz          = Symbol(:ke_flux_, dir)

    @eval begin
        # j_𝒦 = (p - p★) u + u·F
        @inline $unmasked_ke_flux_xyz(i, j, k, grid, clock, fields, parameters, pressures, reference_potential) =
            @inbounds fields.$velocity_component[i, j, k] *
                      $average_to_face(i, j, k, grid, pressure_anomalyᶜᶜᶜ, pressures, reference_potential) +
                      $viscous_ke_flux_xyz(i, j, k, grid, parameters.viscous_closure, parameters.closure_fields, clock,
                                           fields, parameters.buoyancy)

        # Zero at walls, where the velocity normal to the wall and the model's viscous fluxes vanish
        @inline $ke_flux_xyz(i, j, k, grid, clock, fields, parameters, pressures, reference_potential) =
            $wall_mask(i, j, k, grid, $unmasked_ke_flux_xyz, clock, fields, parameters, pressures, reference_potential)
    end
end

# The data of the nonhydrostatic pressure and of the hydrostatic pressure anomaly, which may be `nothing`
pressure_data(model) = (model.pressures.pNHS.data, maybe_data(model.pressures.pHY′))
maybe_data(::Nothing) = nothing
maybe_data(field) = field.data

"""
$(TYPEDSIGNATURES)

Return the flux j_𝒦 = (p - p★) u - u·τ of kinetic energy of `model` by the pressure anomaly and the viscous stress,
a `NamedTuple` of its `(x, y, z)` components at faces, zero at walls. Here p = p_NHS + p_HY′, with p_NHS the
nonhydrostatic pressure of the model's last pressure correction: after `set!`, the potential of the projection of the
velocity. For a divergence-free velocity on a regular grid, in every cell

    ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε

exactly, with A_𝒦 the [`ke_advection`](@ref), ℑz[w (b̃ - b★)] the [`available_buoyancy_work`](@ref) and ε the
[`viscous_dissipation`](@ref).
"""
function ke_flux(model, z★::ReferenceHeightField)
    args = (model.clock, velocities_and_tracers(model), production_parameters(model), pressure_data(model),
            z★.operand.reference_potential)
    return (x = KernelFunctionOperation{Face, Center, Center}(with_reference_state, model.grid, z★, ke_flux_x, args...),
            y = KernelFunctionOperation{Center, Face, Center}(with_reference_state, model.grid, z★, ke_flux_y, args...),
            z = KernelFunctionOperation{Center, Center, Face}(with_reference_state, model.grid, z★, ke_flux_z, args...))
end
