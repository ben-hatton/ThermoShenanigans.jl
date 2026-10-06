"""
    thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                   consistent = true, surface_height = top of grid, constant_salinity = nothing)

Return `(; buoyancy, closure, forcing)` for `NonhydrostaticModel(grid; tracers, thermodynamic_model_components(...)...)`.
With `consistent = true` the closure is `(ScalarDiffusivity(ν = viscosity), ThermodynamicDiffusivity(...))` with the
θ forcing σ_θ, so that kinetic plus static energy is conserved. With `consistent = false` it is the standard
`ScalarDiffusivity(ν = viscosity, κ = (θ = thermal_diffusivity, S = salt_diffusivity))` with no σ_θ.
σ_θ evaluates ε without closure fields, so `viscosity` must not need any. Energy is conserved exactly with
Oceananigans' default `Centered(order = 2)` advection. Wall fluxes of θ and S are set only by `FluxBoundaryCondition`s.
"""
function thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                        consistent = true, kw...)
    FT = eltype(grid)
    buoyancy = ThermodynamicBuoyancy(grid, thermodynamics; kw...)

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
