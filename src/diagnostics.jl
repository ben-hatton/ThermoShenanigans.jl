# KernelFunctionOperation at (Center, Center, Center) of f(i, j, k, grid, args...)
center_operation(f, grid, args...) = KernelFunctionOperation{Center, Center, Center}(f, grid, args...)

# x, y and z components of a face flux, at (Face, Center, Center), (Center, Face, Center) and (Center, Center, Face)
face_operations(f_x, f_y, f_z, grid, args...) = (x = KernelFunctionOperation{Face, Center, Center}(f_x, grid, args...),
                                                 y = KernelFunctionOperation{Center, Face, Center}(f_y, grid, args...),
                                                 z = KernelFunctionOperation{Center, Center, Face}(f_z, grid, args...))

#=
The diagnostics read the data of the model's fields, rather than the Fields themselves: a Field's type carries its
grid and boundary conditions, about 2000 characters, and the type of every diagnostic, and of every output writer or
reduction built on it, carries the types of its arguments. Compile times grow with the size of these types; within
a kernel, a Field is reduced to its data anyway.
=#
field_data(fields) = map(field -> field.data, fields)

# The data of the model fields read by the diagnostics
velocities_and_tracers(model) = field_data(merge(model.velocities, model.tracers))

# The parameters of the production terms in `production.jl` for the closure and buoyancy of a model, built as for its
# θ forcing σ_θ
production_parameters(model) = production_parameters(model.closure, model.buoyancy.formulation)

#=
One function per diagnostic, after Oceanostics.jl: each returns a `KernelFunctionOperation` (or a `NamedTuple` of
the x, y and z components of a face flux), so that only the diagnostics asked for are compiled. Wrap them in `Field`
to compute them. All take a consistent model, built with `thermodynamic_model_components` (`consistent = true`) or by
hand with the θ forcing of `θ_production_forcing`, and all but those that depend on h⁰ and z₀ or on the consistent fluxes
also take a Fickian model (`consistent = false`), whose closure is a single `ScalarDiffusivity`. The refused ones
forward on the host to a method typed on the model's diffusivity, which exists only for a `ThermodynamicDiffusivity`.
=#

"""
$(TYPEDSIGNATURES)

Return the `KernelFunctionOperation` at (Center, Center, Center) of the thermodynamic function `f(θ, S, z,
thermodynamics)` of `model`, such as `potential_energy` (𝒫 = h⁰ - b (z - z₀) + g₀ z₀), `temperature` (T), `exner` (π),
`chemical_potential_analogue` (m), `chemical_potential` (μ) or `potential_enthalpy`, the internal energy ℐ = h⁰.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = 283, S = 0.035);

julia> T = Field(thermodynamic_operation(model, temperature));

julia> round(T[1, 1, 4], digits = 6) # T = θ at z₀ = 0, slightly warmer at the top cell centre, z = -0.125
283.000017
```

`potential_energy` needs a consistent model (`consistent = true`), since 𝒫 depends on h⁰ and z₀, which the dynamics
of a Fickian (`consistent = false`) model never use: for a Fickian model, `thermodynamic_operation(model,
potential_energy)` throws a `MethodError`. The other functions, such as π, m, h⁰ and μ, remain thermodynamic properties
of the state of a Fickian model, but have no energetic meaning for it.
"""
thermodynamic_operation(model, f) = thermodynamic_function_operation(model, f)

thermodynamic_operation(model, f::typeof(potential_energy)) =
    thermodynamic_operation(production_parameters(model).diffusivity, model, f)

thermodynamic_operation(::ThermodynamicDiffusivity, model, f::typeof(potential_energy)) =
    thermodynamic_function_operation(model, f)

# Operation of the thermodynamic function f(θ, S, z, thermodynamics) of the model's tracers, at (Center, Center, Center)
thermodynamic_function_operation(model, f) =
    center_operation(thermodynamic_functionᶜᶜᶜ, model.grid, f, model.buoyancy.formulation, field_data(model.tracers))

# Operation of a production term in `production.jl`, at (Center, Center, Center)
production_operation(kernel, model) =
    center_operation(kernel, model.grid, model.clock, velocities_and_tracers(model), production_parameters(model))

"""
$(TYPEDSIGNATURES)

Return the kinetic energy dissipation ε = ∂ⱼuᵢ τᵢⱼ of `model` at (Center, Center, Center), formed from the viscous
fluxes at their native locations so that ∫ ε dV is exactly the discrete kinetic energy sink.
"""
viscous_dissipation(model) = production_operation(dissipationᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the θ production σ_θ = [ε - j_θ·∇π - j_S·∇m] / π of `model` at (Center, Center, Center), the θ source that
makes the potential energy gain exactly the kinetic energy lost to viscosity. It is zero for a Fickian
(`consistent = false`) model, which has no θ source.
"""
θ_production(model) = production_operation(θ_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy production σ_b = σ_b_heating + σ_b_nonlin of `model` at (Center, Center, Center), so that
Db/Dt = σ_b - ∇·j_b. See [`heating_buoyancy_production`](@ref) and [`nonlinear_buoyancy_production`](@ref).
"""
buoyancy_production(model) = production_operation(buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy production σ_b_heating = b_θ σ_θ of `model` by the θ source, at (Center, Center, Center). It is
zero for a Fickian (`consistent = false`) model, which has no θ source.
"""
heating_buoyancy_production(model) = production_operation(heating_buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the nonlinear (cabbeling) buoyancy production σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S of `model` at
(Center, Center, Center), which vanishes for a linear equation of state.
"""
nonlinear_buoyancy_production(model) = production_operation(nonlinear_buoyancy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the entropy production σ_η = ε / T + κ_θ cₚ |∇T|² / T² + κ_S (∂μ/∂S)_T |∇S + Γ_S 𝐤|² / T ≥ 0 of `model`
at (Center, Center, Center). It needs a consistent model (`consistent = true`), since it is the entropy production of
the consistent fluxes: for a Fickian (`consistent = false`) model it throws a `MethodError`.
"""
entropy_production(model) = entropy_production(production_parameters(model).diffusivity, model)

entropy_production(::ThermodynamicDiffusivity, model) = production_operation(entropy_productionᶜᶜᶜ, model)

"""
$(TYPEDSIGNATURES)

Return the conversion j_θ·∇π + j_S·∇m of potential energy by the diffusive fluxes of `model`, at
(Center, Center, Center). It needs a consistent model (`consistent = true`), since it is a term of the budget of 𝒫,
which depends on h⁰ and z₀: for a Fickian (`consistent = false`) model it throws a `MethodError`.
"""
potential_energy_conversion(model) = potential_energy_conversion(production_parameters(model).diffusivity, model)

potential_energy_conversion(::ThermodynamicDiffusivity, model) = production_operation(potential_energy_conversionᶜᶜᶜ, model)

# x, y and z components of a diffusive flux of the diffusivity of a model: its `ThermodynamicDiffusivity`, or its
# Fickian `ScalarDiffusivity`
diffusive_flux_operations(f_x, f_y, f_z, model) =
    face_operations(f_x, f_y, f_z, model.grid, production_parameters(model).diffusivity, model.buoyancy.formulation,
                    field_data(model.tracers))

"""
$(TYPEDSIGNATURES)

Return the diffusive θ flux j_θ of `model`, a `NamedTuple` of its `(x, y, z)` components at faces, zero at walls.
"""
θ_flux(model) = diffusive_flux_operations(masked_θ_flux_x, masked_θ_flux_y, masked_θ_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the diffusive salt flux j_S of `model`, a `NamedTuple` of its `(x, y, z)` components at faces, zero at walls.
"""
salt_flux(model) = diffusive_flux_operations(masked_salt_flux_x, masked_salt_flux_y, masked_salt_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the diffusive buoyancy flux j_b = b_θ j_θ + b_S j_S of `model`, a `NamedTuple` of its `(x, y, z)` components
at faces, zero at walls.
"""
buoyancy_flux(model) = diffusive_flux_operations(buoyancy_flux_x, buoyancy_flux_y, buoyancy_flux_z, model)

"""
$(TYPEDSIGNATURES)

Return the buoyancy work w b̃ of `model` at (Center, Center, Face), with b̃ the buoyancy force used by the model:
the conversion of potential into kinetic energy.
"""
buoyancy_work(model) = KernelFunctionOperation{Center, Center, Face}(buoyancy_workᶜᶜᶠ, model.grid, model.buoyancy.formulation,
                                                                     velocities_and_tracers(model))

# w b̃ at (Center, Center, Face)
@inline buoyancy_workᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields) =
    @inbounds fields.w[i, j, k] * consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields)
