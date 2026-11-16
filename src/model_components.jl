"""
$(TYPEDSIGNATURES)

Return the θ forcing σ_θ = [ε - j_θ·∇π - j_S·∇m] / π of a model with the closure `closure`, a pair of a viscous closure
and a [`ThermodynamicDiffusivity`](@ref) in either order, and the buoyancy formulation `buoyancy_formulation`, a
[`ThermodynamicBuoyancy`](@ref). σ_θ makes the potential energy gain exactly the kinetic energy lost to viscosity
when the model is built with the same `closure` and `buoyancy_formulation`. It evaluates ε without closure fields, so
that the viscous closure must not need any. Other θ forcings are passed with σ_θ in a tuple, `forcing = (; θ = (σ_θ, F_θ))`,
which Oceananigans adds to the θ tendency as `MultipleForcings`.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> closure = (ScalarDiffusivity(ν = 1e-6), ThermodynamicDiffusivity(thermal_diffusivity = 1e-7, salt_diffusivity = 1e-9));

julia> buoyancy = ThermodynamicBuoyancy(grid, thermodynamics);

julia> σ_θ = θ_production_forcing(closure, buoyancy);

julia> heating(x, z, t) = 1e-6 * cos(2π * x);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), buoyancy, closure, forcing = (; θ = (σ_θ, heating)));

julia> summary(model.forcing.θ)
"MultipleForcings with 2 forcings"
```
"""
θ_production_forcing(closure::Tuple, buoyancy_formulation) =
    Forcing(θ_productionᶜᶜᶜ; discrete_form = true, parameters = production_parameters(closure, buoyancy_formulation))

"""
    thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                   forcing = NamedTuple(), consistent = true, constant_salinity = nothing)

Return `(; buoyancy, closure, forcing)` for `NonhydrostaticModel(grid; tracers, thermodynamic_model_components(...)...)`.
With `consistent = true` the closure is `(ScalarDiffusivity(ν = viscosity), ThermodynamicDiffusivity(...))` and
`forcing` holds the θ forcing σ_θ of [`θ_production_forcing`](@ref), so that kinetic plus potential energy is
conserved. With `consistent = false` it is the standard
`ScalarDiffusivity(ν = viscosity, κ = (θ = thermal_diffusivity, S = salt_diffusivity))` with no σ_θ.
σ_θ evaluates ε without closure fields, so `viscosity` must not need any. Energy is conserved exactly with
Oceananigans' default `Centered(order = 2)` advection. With `consistent = true`, wall fluxes of θ and S are set only by
`FluxBoundaryCondition`s.

The dynamics of a Fickian (`consistent = false`) model never use the internal energy h⁰ or the reference pressure
height z₀, so that of the potential-type energies only the available potential energy 𝒜 and the background
gravitational energy 𝒢★ are defined for it. Its diagnostics are:

* supported: `thermodynamic_operation` (of every function but `potential_energy`), `viscous_dissipation`,
  `buoyancy_production`, `nonlinear_buoyancy_production`, `θ_flux`, `salt_flux`, `buoyancy_flux`, `buoyancy_work`,
  `buoyancy_tendency`, `advective_buoyancy_tendency`, `reference_height`, `reference_displacement`,
  `reference_potential`, `reference_buoyancy`, `reference_work`, `available_potential_energy`,
  `background_gravitational_energy`, `ape_production`, `nonlinear_ape_production`, `ape_dissipation`, `ape_flux`,
  `ape_advection`, `available_buoyancy_work`, `spurious_ape_production`, `reference_tendency_work`, `kinetic_energy`,
  `ke_advection` and `ke_flux`;
* supported and identically zero, since the model has no θ source: `θ_production`, `heating_buoyancy_production` and
  `heating_ape_production`;
* refused, with a `MethodError` when the diagnostic is constructed, since they depend on h⁰ and z₀ or on the consistent
  fluxes: `entropy_production`, `potential_energy_conversion`, `thermodynamic_operation(model, potential_energy)`,
  `background_potential_energy`, `bpe_flux`, `bpe_advection` and `spurious_bpe_production`.

Only a Fickian `ScalarDiffusivity` with constant κ is supported. For both closures, the diagnostic fluxes carry the wall
fluxes of every boundary condition the model accepts: `FluxBoundaryCondition`s, and for a Fickian model also
`ValueBoundaryCondition`s and `GradientBoundaryCondition`s, whose wall fluxes its closure carries.

A model is built in either of two ways:

* with these components, passing any other forcing here, as the `NamedTuple` `forcing`: the returned `forcing` is
  `forcing` with its θ entry combined with σ_θ, which Oceananigans adds to the θ tendency as `MultipleForcings`.
  Julia keeps only the last of two `forcing` keywords, so that a `forcing` given to the model constructor next to
  these components drops either σ_θ or the user forcing;
* by hand, passing a closure, a [`ThermodynamicBuoyancy`](@ref) and the θ forcing
  `θ_production_forcing(closure, buoyancy_formulation)`, with any other θ forcing next to it, to
  `NonhydrostaticModel`, as in the example of [`θ_production_forcing`](@ref).

Either way, energy is conserved only if σ_θ is built from the same closure and buoyancy as the model. Any other θ
forcing F_θ changes the energy by ∫ π F_θ dV.

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

julia> heating(x, z, t) = 1e-6 * cos(2π * x);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-9, forcing = (; θ = heating));

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> summary(model.forcing.θ)
"MultipleForcings with 2 forcings"

julia> fickian_components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6,
                                                           thermal_diffusivity = 1e-7, salt_diffusivity = 1e-9,
                                                           consistent = false);

julia> fickian = NonhydrostaticModel(grid; tracers = (:θ, :S), fickian_components...);

julia> try entropy_production(fickian) catch exception; (typeof(exception), exception.f) end
(MethodError, entropy_production)
```
"""
function thermodynamic_model_components(grid, thermodynamics; viscosity, thermal_diffusivity, salt_diffusivity,
                                        forcing = NamedTuple(), consistent = true, buoyancy_kwargs...)
    FT = eltype(grid)
    buoyancy_formulation = ThermodynamicBuoyancy(grid, thermodynamics; buoyancy_kwargs...)

    if consistent
        closure = (ScalarDiffusivity(FT; ν = viscosity),
                   ThermodynamicDiffusivity(FT; thermal_diffusivity, salt_diffusivity))
        σ_θ = θ_production_forcing(closure, buoyancy_formulation)
        θ_forcing = haskey(forcing, :θ) ? combine_forcings(forcing.θ, σ_θ) : σ_θ
        return (; buoyancy = buoyancy_formulation, closure, forcing = merge(forcing, (; θ = θ_forcing)))
    else
        κ = buoyancy_formulation isa ConstantSalinityThermodynamicBuoyancy ? (; θ = thermal_diffusivity) :
                                                                (; θ = thermal_diffusivity, S = salt_diffusivity)
        return (; buoyancy = buoyancy_formulation, closure = ScalarDiffusivity(FT; ν = viscosity, κ), forcing)
    end
end

# A user forcing, or a tuple or `MultipleForcings` of user forcings, followed by σ_θ, for `MultipleForcings`
combine_forcings(user_forcing, σ_θ) = (user_forcing, σ_θ)
combine_forcings(user_forcings::Tuple, σ_θ) = (user_forcings..., σ_θ)
combine_forcings(user_forcings::MultipleForcings, σ_θ) = (user_forcings.forcings..., σ_θ)
