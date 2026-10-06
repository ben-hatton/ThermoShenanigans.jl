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
    tb = model.buoyancy.formulation
    cl = only(filter(c -> c isa ThermodynamicDiffusivity, model.closure))
    p = (; viscous_closure = model.closure, closure_fields = model.closure_fields, diffusivity = cl, buoyancy = tb)

    centre(f, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)
    faces(fx, fy, fz) = (x = KernelFunctionOperation{Face, Center, Center}(fx, grid, cl, tb, C),
                         y = KernelFunctionOperation{Center, Face, Center}(fy, grid, cl, tb, C),
                         z = KernelFunctionOperation{Center, Center, Face}(fz, grid, cl, tb, C))

    return (Σ          = centre(thermodynamic_functionᶜᶜᶜ, static_energy, tb, C),
            T          = centre(thermodynamic_functionᶜᶜᶜ, temperature, tb, C),
            π          = centre(thermodynamic_functionᶜᶜᶜ, exner, tb, C),
            Σ_S        = centre(thermodynamic_functionᶜᶜᶜ, chemical_potential_analogue, tb, C),
            μ          = centre(thermodynamic_functionᶜᶜᶜ, chemical_potential, tb, C),
            ε          = centre(dissipationᶜᶜᶜ, clock, C, p),
            σ_θ        = centre(θ_productionᶜᶜᶜ, clock, C, p),
            σ_b        = centre(buoyancy_productionᶜᶜᶜ, clock, C, p),
            σ_b_nonlin = centre(nonlinear_buoyancy_productionᶜᶜᶜ, clock, C, p),
            σ_η        = centre(entropy_productionᶜᶜᶜ, clock, C, p),
            conversion = centre(static_energy_conversionᶜᶜᶜ, clock, C, p),
            j_θ        = faces(masked_heat_flux_x, masked_heat_flux_y, masked_heat_flux_z),
            j_S        = faces(masked_salt_flux_x, masked_salt_flux_y, masked_salt_flux_z),
            j_b        = faces(buoyancy_flux_x, buoyancy_flux_y, buoyancy_flux_z),
            wb         = KernelFunctionOperation{Center, Center, Face}(buoyancy_workᶜᶜᶠ, grid, tb, C))
end

@inline buoyancy_workᶜᶜᶠ(i, j, k, grid, tb, C) = @inbounds C.w[i, j, k] * consistent_buoyancyᶜᶜᶠ(i, j, k, grid, tb, C)
