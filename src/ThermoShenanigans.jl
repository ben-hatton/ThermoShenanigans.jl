module ThermoShenanigans

export
    # Thermodynamics
    AbstractBoussinesqThermodynamics, CabbelingBoussinesqThermodynamics,
    buoyancy,           ∂b∂θ,  ∂b∂S,  ∂²b∂θ²,  ∂²b∂θ∂S,  ∂²b∂S²,
    potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
    entropy,            ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²,
    potential_energy, exner, chemical_potential_analogue, temperature, chemical_potential,
    ∂T∂θ, ∂T∂S, heat_capacity, isothermal_∂b∂S, isothermal_∂μ∂S,

    # Buoyancy, closure and model components
    ThermodynamicBuoyancy, ThermodynamicDiffusivity, θ_production_forcing, thermodynamic_model_components,

    # Thermodynamic diagnostics
    thermodynamic_operation, internal_energy, potential_energy_tendency, viscous_dissipation, θ_production, buoyancy_production,
    heating_buoyancy_production, nonlinear_buoyancy_production, entropy_production, potential_energy_conversion,
    θ_flux, salt_flux, buoyancy_flux, buoyancy_work, buoyancy_tendency, advective_buoyancy_tendency,

    # Reference state and local energetics
    Sorting, reference_height, reference_displacement, reference_potential, reference_buoyancy, reference_work,
    available_potential_energy, background_potential_energy, gravitational_energy, background_gravitational_energy,

    # Local APE budget
    ape_production, heating_ape_production, nonlinear_ape_production, ape_dissipation, ape_flux, ape_advection,
    available_buoyancy_work, spurious_ape_production, reference_tendency_work, ape_tendency,

    # Local KE and BPE budgets
    kinetic_energy, ke_advection, ke_flux, bpe_flux, bpe_advection, spurious_bpe_production, ke_tendency,
    bpe_tendency,

    # Isopycnal budget of the reference buoyancy
    isopycnal_buoyancy_production, isopycnal_buoyancy_flux_divergence, isopycnal_advective_buoyancy_tendency,
    isopycnal_wall_buoyancy_flux_divergence, effective_diapycnal_diffusivity, numerical_diapycnal_diffusivity

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel
using Oceananigans: Oceananigans
using Oceananigans.Advection: div_Uc, div_𝐯u, div_𝐯v, div_𝐯w, half_ϕ²
using Oceananigans.Architectures: CPU, architecture, on_architecture
using Oceananigans.BoundaryConditions: BoundaryCondition, Bottom, Flux, Top, explicit_flux, fill_halo_regions!, getbc
using Oceananigans.Fields: Field, FieldStatus, compute_at!, interior, set_status!
using Oceananigans.Grids: Center, Face, NegativeZDirection, XYZRegularRG, inactive_cell, znode
using Oceananigans.Operators: ∂xᶠᶜᶜ, ∂yᶜᶠᶜ, ∂zᶜᶜᶠ, ∂zᶜᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ,
                              ℑxyᶜᶜᵃ, ℑxzᶜᵃᶜ, ℑyzᵃᶜᶜ, δxᶜᵃᵃ, δxᶠᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
                              Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ, Axᶜᶜᶜ, Axᶠᶜᶜ, Axᶠᶠᶜ, Axᶠᶜᶠ, Ayᶜᶜᶜ, Ayᶜᶠᶜ, Ayᶠᶠᶜ, Ayᶜᶠᶠ, Azᶜᶜᶜ, Azᶜᶜᶠ, Azᶠᶜᶠ, Azᶜᶠᶠ,
                              Vᶜᶜᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ, Vᶜᶜᶠ, Δzᶜᶜᶜ, Δzᶜᶜᶠ
using Oceananigans.AbstractOperations: KernelFunctionOperation, _compute!
using Oceananigans.Forcings: Forcing, MultipleForcings
using Oceananigans.Utils: launch!, prettysummary
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization
using Oceananigans.BuoyancyFormulations: BuoyancyForce, θ_and_sᴬ
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, ScalarDiffusivity, biharmonic_mask_x, biharmonic_mask_y, biharmonic_mask_z,
                                       viscous_flux_ux, viscous_flux_uy, viscous_flux_uz, viscous_flux_vx, viscous_flux_vy,
                                       viscous_flux_vz, viscous_flux_wx, viscous_flux_wy, viscous_flux_wz,
                                       min_Δxyz, ThreeDimensionalFormulation, x_peripheral_node, y_peripheral_node,
                                       z_peripheral_node

import Oceananigans.Fields: compute!
import Oceananigans.Utils: with_tracers
import Oceananigans.Diagnostics: cell_diffusion_timescale
import Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
import Oceananigans.BuoyancyFormulations: AbstractBuoyancyFormulation, required_tracers, get_temperature_and_salinity,
                                          buoyancy_perturbationᶜᶜᶜ, z_dot_g_bᶜᶜᶠ, ∂x_b, ∂y_b, ∂z_b

include("abstract_boussinesq_thermodynamics.jl")
include("cabbeling_boussinesq_thermodynamics.jl")
include("thermodynamic_buoyancy.jl")
include("thermodynamic_diffusivity.jl")
include("production.jl")
include("model_components.jl")
include("diagnostics.jl")
include("reference_state.jl")
include("sorting.jl")
include("ape_budget.jl")
include("ke_budget.jl")
include("bpe_budget.jl")
include("reference_buoyancy_budget.jl")

end # module
