"""
    thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                   consistent = true, constant_salinity = nothing)

Return `(; buoyancy, closure, forcing)` for `NonhydrostaticModel(grid; tracers, thermodynamic_model_components(...)...)`.
With `consistent = true` the closure is `(ScalarDiffusivity(ν = viscosity), ThermodynamicDiffusivity(...))` with the
θ forcing `σ_θ`, so that kinetic plus potential energy is conserved. With `consistent = false` it is the standard
`ScalarDiffusivity(ν = viscosity, κ = (θ = thermal_diffusivity, S = salt_diffusivity))` with no `σ_θ`.
`σ_θ` evaluates ε without closure fields, so `viscosity` must not need any. Energy is conserved exactly with
Oceananigans' default `Centered(order = 2)` advection. Wall fluxes of θ and S are set only by `FluxBoundaryCondition`s.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9);

julia> components.closure
(ScalarDiffusivity{ExplicitTimeDiscretization}(ν=1.0e-6, κ=nothing), ThermodynamicDiffusivity(thermal_diffusivity=1.0e-7, salt_diffusivity=1.0e-9))
```
"""
function thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                        consistent = true, buoyancy_kwargs...)
    FT = eltype(grid)
    buoyancy = ThermodynamicBuoyancy(grid, thermodynamics; buoyancy_kwargs...)

    if consistent
        viscous_closure = ScalarDiffusivity(FT; ν = viscosity)
        diffusivity = ThermodynamicDiffusivity(FT; thermal_diffusivity, salt_diffusivity)
        parameters = (; viscous_closure, closure_fields = nothing, diffusivity, buoyancy)
        forcing = (; θ = Forcing(θ_productionᶜᶜᶜ; discrete_form = true, parameters))
        return (; buoyancy, closure = (viscous_closure, diffusivity), forcing)
    else
        κ = buoyancy isa ConstantSalinityThermodynamicBuoyancy ? (; θ = thermal_diffusivity) :
                                                                (; θ = thermal_diffusivity, S = salt_diffusivity)
        return (; buoyancy, closure = ScalarDiffusivity(FT; ν = viscosity, κ), forcing = NamedTuple())
    end
end
