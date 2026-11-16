"""
    ThermodynamicDiffusivity

Closure for the thermodynamically consistent θ and salt fluxes. Fields:

* `thermal_diffusivity`: κ_θ
* `salt_diffusivity`: κ_S

The type parameter `N` holds the tracer names, set by `with_tracers`.
"""
struct ThermodynamicDiffusivity{FT, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    thermal_diffusivity :: FT
    salt_diffusivity :: FT
end

"""
    ThermodynamicDiffusivity(FT = Oceananigans.defaults.FloatType; thermal_diffusivity, salt_diffusivity)

Return a closure for the thermodynamically consistent θ and salt fluxes

    j_θ = - κ_θ (cₚ / π) ∇T - (T_S / T_θ) j_S
    j_S = - κ_S (∇S + Γ_S 𝐤),   Γ_S = - (∂b/∂S)_T / (∂μ/∂S)_T

where cₚ is the local heat capacity and `Γ_S` is minus the equilibrium salt gradient (barodiffusion). Since
`cₚ / π = 1 / T_θ`, the ∇θ part of `j_θ` has diffusivity exactly `κ_θ`: `j_θ = - κ_θ (∇θ + (T_z / T_θ) 𝐤)` when
`T_S = 0`, as for `CabbelingBoussinesqThermodynamics`. The heat flux `π j_θ = - cₚ κ_θ ∇T` then has the conductivity
`ρ₀ cₚ κ_θ`, which varies with depth like cₚ.

Use it with a [`ThermodynamicBuoyancy`](@ref), and a separate closure for viscosity.

```jldoctest
julia> using ThermoShenanigans

julia> ThermodynamicDiffusivity(thermal_diffusivity = 1.4e-7, salt_diffusivity = 1e-9)
ThermodynamicDiffusivity(thermal_diffusivity=1.4e-7, salt_diffusivity=1.0e-9)
```
"""
ThermodynamicDiffusivity(FT = Oceananigans.defaults.FloatType; thermal_diffusivity, salt_diffusivity) =
    ThermodynamicDiffusivity{FT, nothing}(convert(FT, thermal_diffusivity), convert(FT, salt_diffusivity))

Base.summary(closure::ThermodynamicDiffusivity) =
    string("ThermodynamicDiffusivity(thermal_diffusivity=", prettysummary(closure.thermal_diffusivity),
                                  ", salt_diffusivity=", prettysummary(closure.salt_diffusivity), ")")

Base.show(io::IO, closure::ThermodynamicDiffusivity) = print(io, summary(closure))

# Record the tracer names, so that the tracer index `id` identifies θ and S at compile time
with_tracers(tracers, closure::ThermodynamicDiffusivity{FT}) where FT =
    ThermodynamicDiffusivity{FT, tracers}(closure.thermal_diffusivity, closure.salt_diffusivity)

@inline tracer_name(::ThermodynamicDiffusivity{FT, N}, ::Val{id}) where {FT, N, id} = Val(N[id])

#=
Δ² / max(κ_θ, κ_S) for `TimeStepWizard(; diffusive_cfl)` and `DiffusiveCFL`.
=#
cell_diffusion_timescale(closure::ThermodynamicDiffusivity, closure_fields, grid, clock, fields) =
    min_Δxyz(grid, ThreeDimensionalFormulation())^2 / max(closure.thermal_diffusivity, closure.salt_diffusivity)

#####
##### Fluxes at faces, from differences and interpolations of cell-centre thermodynamic variables
#####

# Γ_S
@inline barodiffusion_coefficient(θ, S, z, thermodynamics) =
    - isothermal_∂b∂S(θ, S, z, thermodynamics) / isothermal_∂μ∂S(θ, S, z, thermodynamics)

# cₚ / π
@inline θ_flux_prefactor(θ, S, z, thermodynamics) =
    heat_capacity(θ, S, z, thermodynamics) / exner(θ, S, z, thermodynamics)

# - T_S / T_θ
@inline θ_flux_salt_coefficient(θ, S, z, thermodynamics) =
    - ∂T∂S(θ, S, z, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)

# ∇S + Γ_S 𝐤, the (scaled) isothermal salt force
@inline salt_gradient_x(i, j, k, grid, buoyancy_formulation, fields) = ∂xᶠᶜᶜ(i, j, k, grid, fields.S)
@inline salt_gradient_y(i, j, k, grid, buoyancy_formulation, fields) = ∂yᶜᶠᶜ(i, j, k, grid, fields.S)
@inline salt_gradient_z(i, j, k, grid, buoyancy_formulation, fields) =
    ∂zᶜᶜᶠ(i, j, k, grid, fields.S) +
    ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, barodiffusion_coefficient, buoyancy_formulation, fields)

# Oceananigans' `biharmonic_mask_{x,y,z}(i, j, k, grid, f, args...)` return zero where the face is a wall of a bounded
# direction, and `f(i, j, k, grid, args...)` elsewhere. Here they set the θ and salt fluxes of both closures to zero
# at walls.
const wall_mask_x = biharmonic_mask_x
const wall_mask_y = biharmonic_mask_y
const wall_mask_z = biharmonic_mask_z

# Generated for each direction, with `dir` and the operators of that direction:
#   salt_gradient_{x,y,z}, salt_flux_{x,y,z}, θ_flux_{x,y,z}, masked_salt_flux_{x,y,z}, masked_θ_flux_{x,y,z},
#   tracer_flux_{x,y,z}, and the `ThermodynamicDiffusivity` methods of Oceananigans' diffusive_flux_{x,y,z}
for (dir, ∂, ℑ, wall_mask) in ((:x, ∂xᶠᶜᶜ, ℑxᶠᵃᵃ, wall_mask_x),
                               (:y, ∂yᶜᶠᶜ, ℑyᵃᶠᵃ, wall_mask_y),
                               (:z, ∂zᶜᶜᶠ, ℑzᵃᵃᶠ, wall_mask_z))
    salt_gradient    = Symbol(:salt_gradient_, dir)
    salt_flux_kernel = Symbol(:salt_flux_, dir)
    θ_flux_kernel    = Symbol(:θ_flux_, dir)
    masked_salt_flux = Symbol(:masked_salt_flux_, dir)
    masked_θ_flux    = Symbol(:masked_θ_flux_, dir)
    tracer_flux      = Symbol(:tracer_flux_, dir)
    diffusive_flux   = Symbol(:diffusive_flux_, dir)

    @eval begin
        @inline $salt_gradient(i, j, k, grid, buoyancy_formulation::ConstantSalinityThermodynamicBuoyancy, fields) = zero(grid)

        # j_S = - κ_S (∇S + Γ_S 𝐤)
        @inline $salt_flux_kernel(i, j, k, grid, closure::ThermodynamicDiffusivity, buoyancy_formulation, fields) =
            - closure.salt_diffusivity * $salt_gradient(i, j, k, grid, buoyancy_formulation, fields)

        # j_θ = - κ_θ ℑ(cₚ / π) ∂T + ℑ(- T_S / T_θ) j_S
        @inline $θ_flux_kernel(i, j, k, grid, closure::ThermodynamicDiffusivity, buoyancy_formulation, fields) =
            - closure.thermal_diffusivity *
              $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, θ_flux_prefactor, buoyancy_formulation, fields) *
              $∂(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, temperature, buoyancy_formulation, fields) +
            $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, θ_flux_salt_coefficient, buoyancy_formulation, fields) *
            $salt_flux_kernel(i, j, k, grid, closure, buoyancy_formulation, fields)

        #=
        Fickian fluxes of the constant-κ `ScalarDiffusivity` of `consistent = false`, evaluated by Oceananigans' own
        `diffusive_flux_{x,y,z}(i, j, k, grid, closure, K, id, c, clock, fields, b)` (Oceananigans 0.116.1,
        `src/TurbulenceClosures/abstract_scalar_diffusivity_closure.jl:260-262`), so that they are exactly the model's.
        * The tracer name is passed as the id. `diffusivity(closure, K, ::Val{id}) = closure.κ[id]`
          (`turbulence_closure_implementations/scalar_diffusivity.jl:195`), and `with_tracers` (same file, line 189)
          keys κ by tracer name, in the model's tracer order, through `tracer_diffusivities(tracer_names,
          user_κ::NamedTuple)` (`turbulence_closure_utils.jl:8`), so that `Val(:θ)` selects the same κ as the model's
          integer tracer index.
        * The form without a time-discretisation argument is the full flux for any time discretisation and formulation.
          The model reaches it through `∇_dot_qᶜ` (`closure_kernel_operators.jl:41-46`), whose time-discretisation
          argument is stripped for explicit closures (`implicit_explicit_time_discretization.jl:15-17`), and directions
          that a formulation does not diffuse fall back to zero (`TurbulenceClosures.jl:162`).
        * `nothing` for the closure fields K and the clock is valid because `production_parameters` admits only a
          `NamedTuple` of `Number` κ, which is read without either (`abstract_scalar_diffusivity_closure.jl:337-342`).
        * The constant-salinity model has neither `fields.S` nor `κ.S`: its salt flux is zero.
        =#
        @inline $θ_flux_kernel(i, j, k, grid, closure::ScalarDiffusivity, buoyancy_formulation, fields) =
            $diffusive_flux(i, j, k, grid, closure, nothing, Val(:θ), fields.θ, nothing, fields, nothing)
        @inline $salt_flux_kernel(i, j, k, grid, closure::ScalarDiffusivity, buoyancy_formulation, fields) =
            $diffusive_flux(i, j, k, grid, closure, nothing, Val(:S), fields.S, nothing, fields, nothing)
        @inline $salt_flux_kernel(i, j, k, grid, ::ScalarDiffusivity, ::ConstantSalinityThermodynamicBuoyancy, fields) = zero(grid)

        #=
        Zero at walls, for both closures. The `ThermodynamicDiffusivity`'s fluxes in the model are these masked fluxes,
        and Oceananigans adds the flux of a `FluxBoundaryCondition` separately, as for biharmonic closures. The Fickian
        closure's own flux at a wall is zero for Flux conditions, including the default no flux, whose halo mirrors the
        interior (`src/BoundaryConditions/fill_halo_regions_flux.jl`), so that the mask leaves its fluxes unchanged.
        Value and Gradient conditions, which only Fickian models accept, extrapolate the halo
        (`src/BoundaryConditions/fill_halo_regions_value_gradient.jl`): the model's closure carries their wall flux,
        and the mask removes it from these fluxes.
        =#
        @inline $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
            $wall_mask(i, j, k, grid, $salt_flux_kernel, closure, buoyancy_formulation, fields)
        @inline $masked_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
            $wall_mask(i, j, k, grid, $θ_flux_kernel, closure, buoyancy_formulation, fields)

        # Flux of the tracer with the given name; zero for tracers other than θ and S
        @inline $tracer_flux(i, j, k, grid, ::Val{:θ}, closure, buoyancy_formulation, fields) =
            $masked_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields)
        @inline $tracer_flux(i, j, k, grid, ::Val{:S}, closure, buoyancy_formulation, fields) =
            $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields)
        @inline $tracer_flux(i, j, k, grid, ::Val, closure, buoyancy_formulation, fields) = zero(grid)

        # The Oceananigans interface: the tracer index `id` selects θ or S through the closure's tracer names
        @inline $diffusive_flux(i, j, k, grid, closure::ThermodynamicDiffusivity, K, id, c, clock, fields, b) =
            $tracer_flux(i, j, k, grid, tracer_name(closure, id), closure, b.formulation, fields)
    end
end
