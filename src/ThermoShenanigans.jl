module ThermoShenanigans

export
    # Thermodynamics
    AbstractBoussinesqThermodynamics, CabbelingBoussinesqThermodynamics,
    buoyancy,           ∂b∂θ,  ∂b∂S,  ∂²b∂θ²,  ∂²b∂θ∂S,  ∂²b∂S²,
    potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
    entropy,            ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²,
    static_energy, exner, chemical_potential, in_situ_temperature,
    ∂T∂θ, ∂T∂S, heat_capacity, isothermal_∂θ∂S, isothermal_∂b∂S, isothermal_∂m∂S,

    # Buoyancy and closure
    ThermodynamicBuoyancy, ThermodynamicDiffusivity

using Oceananigans: Oceananigans
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ
using Oceananigans.Utils: prettysummary
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization
using Oceananigans.BuoyancyFormulations: θ_and_sᴬ
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, biharmonic_mask_x, biharmonic_mask_y, biharmonic_mask_z

import Oceananigans.Utils: with_tracers
import Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
import Oceananigans.BuoyancyFormulations: AbstractBuoyancyFormulation, required_tracers, get_temperature_and_salinity,
                                          buoyancy_perturbationᶜᶜᶜ, ∂x_b, ∂y_b, ∂z_b

include("thermodynamics.jl")
include("thermodynamic_buoyancy.jl")
include("thermodynamic_diffusivity.jl")
include("show_methods.jl")

end # module
