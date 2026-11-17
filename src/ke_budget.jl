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

# p_NHS at the cell centres, read from the data of the model's `pNHS`
@inline nonhydrostatic_pressureᶜᶜᶜ(i, j, k, grid, nonhydrostatic_pressure) = @inbounds nonhydrostatic_pressure[i, j, k]

#=
p_NHS as a computed `Field`, whose halos `compute!` fills from the interior, as in Oceanostics v0.21.7
(`src/KineticEnergyEquation.jl:451-459`). The `_field` suffix keeps the function apart from the argument name
`nonhydrostatic_pressure` of `pressureᶜᶜᶜ`.
=#
nonhydrostatic_pressure_field(model) =
    Field(center_operation(nonhydrostatic_pressureᶜᶜᶜ, model.grid, model.pressures.pNHS.data))

#=
The nonhydrostatic pressure, as a computed `Field`, and the data of the hydrostatic pressure anomaly, which may be
`nothing`. p_NHS is read through a computed `Field` because after a time step the halos of `pNHS` hold p·Δτ, not p:
the pressure correction divides p·Δτ by Δτ over the interior only (Oceananigans 0.116.1,
`src/Models/NonhydrostaticModels/pressure_correction.jl:103-106`). Computing a diagnostic computes the `Field`: a
`KernelFunctionOperation` computes its arguments (`src/AbstractOperations/kernel_function_operation.jl:112`), and a
`Tuple` argument falls through the generic `compute_at!` to `compute!`, which computes the `Field`s inside it
(`src/Fields/field.jl:521` and `:480`).
=#
pressure_data(model) = (nonhydrostatic_pressure_field(model), maybe_data(model.pressures.pHY′))
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

#=
u ∂ₜu, v ∂ₜv and w ∂ₜw at the velocity points, with ∂ₜ𝐮 = G_𝐮 - ∇p_NHS, G_𝐮 the model's tendencies, which hold
- ∇p_HY′ in u and v, and p_NHS read through `nonhydrostatic_pressure_field`. After `uᵢGᵢᶜᶜᶜ` of Oceanostics v0.21.7
(`src/KineticEnergyEquation.jl:103-123`), which omits the pressure term.
=#
@inline u_tendency_workᶠᶜᶜ(i, j, k, grid, velocities, tendencies, nonhydrostatic_pressure) =
    @inbounds velocities.u[i, j, k] * (tendencies.u[i, j, k] - ∂xᶠᶜᶜ(i, j, k, grid, nonhydrostatic_pressure))
@inline v_tendency_workᶜᶠᶜ(i, j, k, grid, velocities, tendencies, nonhydrostatic_pressure) =
    @inbounds velocities.v[i, j, k] * (tendencies.v[i, j, k] - ∂yᶜᶠᶜ(i, j, k, grid, nonhydrostatic_pressure))
@inline w_tendency_workᶜᶜᶠ(i, j, k, grid, velocities, tendencies, nonhydrostatic_pressure) =
    @inbounds velocities.w[i, j, k] * (tendencies.w[i, j, k] - ∂zᶜᶜᶠ(i, j, k, grid, nonhydrostatic_pressure))

# ∂ₜ𝒦 = ℑx(u ∂ₜu) + ℑy(v ∂ₜv) + ℑz(w ∂ₜw)
@inline ke_tendencyᶜᶜᶜ(i, j, k, grid, velocities, tendencies, nonhydrostatic_pressure) =
    ℑxᶜᵃᵃ(i, j, k, grid, u_tendency_workᶠᶜᶜ, velocities, tendencies, nonhydrostatic_pressure) +
    ℑyᵃᶜᵃ(i, j, k, grid, v_tendency_workᶜᶠᶜ, velocities, tendencies, nonhydrostatic_pressure) +
    ℑzᵃᵃᶜ(i, j, k, grid, w_tendency_workᶜᶜᶠ, velocities, tendencies, nonhydrostatic_pressure)

"""
$(TYPEDSIGNATURES)

Return the tendency ∂ₜ𝒦 = ℑx(u ∂ₜu) + ℑy(v ∂ₜv) + ℑz(w ∂ₜw) of the [`kinetic_energy`](@ref) of `model` at
(Center, Center, Center), with ∂ₜ𝐮 = G_𝐮 - ∇p_NHS and G_𝐮 the model's tendencies. For a divergence-free velocity on a
regular grid, with no forcing of the velocities, Coriolis or background fields, in every cell

    ∂𝒦/∂t = A_𝒦 - ∇·j_𝒦 + ℑz[w (b̃ - b★)] - ε

exactly, with A_𝒦 the [`ke_advection`](@ref), j_𝒦 the [`ke_flux`](@ref), ℑz[w (b̃ - b★)] the
[`available_buoyancy_work`](@ref) and ε the [`viscous_dissipation`](@ref).

It reads the model's tendencies `Gⁿ`, which `time_step!`, `set!` and `update_state!` compute for the current state; it
is valid after any of them. p_NHS is that of the model's last pressure correction, as in [`ke_flux`](@ref): after
`set!`, the potential of the projection of the velocity, and after `time_step!`, the pressure of its last substep. It
differs from the pressure of `Gⁿ` by a flux divergence, so that ∫ ∂ₜ𝒦 dV is exact and the budget closes in every cell.
The tendencies of the velocities by their `FluxBoundaryCondition`s are not included.
"""
ke_tendency(model) = center_operation(ke_tendencyᶜᶜᶜ, model.grid, field_data(model.velocities), model_tendencies(model),
                                      nonhydrostatic_pressure_field(model))
