# KernelFunctionOperation at (Center, Center, Center) of f(i, j, k, grid, args...)
center_operation(f, grid, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)

# x, y and z components of a face flux, at (Face, Center, Center), (Center, Face, Center) and (Center, Center, Face)
face_operations(f_x, f_y, f_z, grid, args...) = (x = KernelFunctionOperation{Face, Center, Center}(f_x, grid, args...),
                                                 y = KernelFunctionOperation{Center, Face, Center}(f_y, grid, args...),
                                                 z = KernelFunctionOperation{Center, Center, Face}(f_z, grid, args...))

"""
    thermodynamic_diagnostics(model)

Return a `NamedTuple` of `KernelFunctionOperation`s for a model built with
[`thermodynamic_model_components`](@ref) (`consistent = true`):

* at (Center, Center, Center): static energy `Σ = h⁰ - b z`, temperature `T`, Exner function `π`,
  chemical potential analogue `Σ_S`, chemical potential `μ`, dissipation `ε`, θ production `σ_θ`,
  buoyancy production `σ_b`, its nonlinear part `σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S`, entropy production `σ_η`,
  and the static energy `conversion = j_θ·∇π + j_S·∇Σ_S`;
* at faces: the fluxes `j_θ`, `j_S` and `j_b = b_θ j_θ + b_S j_S`, each a `NamedTuple` of `(x, y, z)` components,
  and the buoyancy work `wb = w b̃` at (Center, Center, Face), with b̃ the buoyancy force used by the model.

Wrap them in `Field` to compute them, e.g. `Field(thermodynamic_diagnostics(model).σ_θ)`.
"""
function thermodynamic_diagnostics(model)
    grid, C, clock = model.grid, merge(model.velocities, model.tracers), model.clock
    buoyancy_formulation = model.buoyancy.formulation
    diffusivity = only(filter(Base.Fix2(isa, ThermodynamicDiffusivity), model.closure))
    parameters = (; viscous_closure = model.closure, closure_fields = model.closure_fields, diffusivity, buoyancy = buoyancy_formulation)

    return (Σ          = center_operation(thermodynamic_functionᶜᶜᶜ, grid, static_energy, buoyancy_formulation, C),
            T          = center_operation(thermodynamic_functionᶜᶜᶜ, grid, temperature, buoyancy_formulation, C),
            π          = center_operation(thermodynamic_functionᶜᶜᶜ, grid, exner, buoyancy_formulation, C),
            Σ_S        = center_operation(thermodynamic_functionᶜᶜᶜ, grid, chemical_potential_analogue, buoyancy_formulation, C),
            μ          = center_operation(thermodynamic_functionᶜᶜᶜ, grid, chemical_potential, buoyancy_formulation, C),
            ε          = center_operation(dissipationᶜᶜᶜ, grid, clock, C, parameters),
            σ_θ        = center_operation(θ_productionᶜᶜᶜ, grid, clock, C, parameters),
            σ_b        = center_operation(buoyancy_productionᶜᶜᶜ, grid, clock, C, parameters),
            σ_b_nonlin = center_operation(nonlinear_buoyancy_productionᶜᶜᶜ, grid, clock, C, parameters),
            σ_η        = center_operation(entropy_productionᶜᶜᶜ, grid, clock, C, parameters),
            conversion = center_operation(static_energy_conversionᶜᶜᶜ, grid, clock, C, parameters),
            j_θ        = face_operations(masked_heat_flux_x, masked_heat_flux_y, masked_heat_flux_z, grid, diffusivity, buoyancy_formulation, C),
            j_S        = face_operations(masked_salt_flux_x, masked_salt_flux_y, masked_salt_flux_z, grid, diffusivity, buoyancy_formulation, C),
            j_b        = face_operations(buoyancy_flux_x, buoyancy_flux_y, buoyancy_flux_z, grid, diffusivity, buoyancy_formulation, C),
            wb         = KernelFunctionOperation{Center, Center, Face}(buoyancy_workᶜᶜᶠ, grid, buoyancy_formulation, C))
end

# w b̃ at (Center, Center, Face)
@inline buoyancy_workᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, C) =
    @inbounds C.w[i, j, k] * consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, C)
