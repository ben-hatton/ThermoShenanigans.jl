"""
    ThermodynamicDiffusivity

Closure for the thermodynamically consistent heat and salt fluxes. Fields:

* `thermal_diffusivity`: κ_θ
* `salt_diffusivity`: κ_S

The type parameter `N` holds the tracer names, set by `with_tracers`.
"""
struct ThermodynamicDiffusivity{FT, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    thermal_diffusivity :: FT
    salt_diffusivity :: FT
end

"""
    ThermodynamicDiffusivity([FT;] thermal_diffusivity, salt_diffusivity)

Return a closure for the thermodynamically consistent heat and salt fluxes

    j_θ = - κ_θ (cₚ / π) ∇T - (T_S / T_θ) j_S
    j_S = - κ_S (∇S + Γ_S 𝐤),   Γ_S = - (∂b/∂S)_T / (∂μ/∂S)_T

where cₚ is the local heat capacity and Γ_S is minus the equilibrium salt gradient (barodiffusion). Since
cₚ / π = 1 / T_θ, the ∇θ part of j_θ has diffusivity exactly κ_θ: j_θ = - κ_θ (∇θ + (T_z / T_θ) 𝐤) when T_S = 0, as for
`CabbelingBoussinesqThermodynamics`. The heat flux π j_θ = - cₚ κ_θ ∇T then has the conductivity ρ₀ cₚ κ_θ, which
varies with depth like cₚ.

Use it with a [`ThermodynamicBuoyancy`](@ref), and a separate closure for viscosity.

```jldoctest
julia> using ThermoShenanigans

julia> ThermodynamicDiffusivity(thermal_diffusivity = 1.4e-7, salt_diffusivity = 1e-9)
ThermodynamicDiffusivity(κ_θ=1.4e-7, κ_S=1.0e-9)
```
"""
ThermodynamicDiffusivity(FT = Oceananigans.defaults.FloatType; thermal_diffusivity, salt_diffusivity) =
    ThermodynamicDiffusivity{FT, nothing}(convert(FT, thermal_diffusivity), convert(FT, salt_diffusivity))

# Record the tracer names, so that the tracer index `id` identifies θ and S at compile time
with_tracers(tracers, closure::ThermodynamicDiffusivity{FT}) where FT =
    ThermodynamicDiffusivity{FT, tracers}(closure.thermal_diffusivity, closure.salt_diffusivity)

@inline tracer_name(::ThermodynamicDiffusivity{FT, N}, ::Val{id}) where {FT, N, id} = Val(N[id])

#=
Δ² / max(κ_θ, κ_S) for `TimeStepWizard(; diffusive_cfl)` and `DiffusiveCFL`. Since cₚ / π = 1 / T_θ, j_θ diffuses θ with
exactly κ_θ.
=#
cell_diffusion_timescale(closure::ThermodynamicDiffusivity, closure_fields, grid, clock, fields) =
    min_Δxyz(grid, ThreeDimensionalFormulation())^2 / max(closure.thermal_diffusivity, closure.salt_diffusivity)

# Fluxes at faces, from differences and interpolations of cell-centre thermodynamic variables

@inline barodiffusion_coefficient(θ, S, z, thermodynamics)  = - isothermal_∂b∂S(θ, S, z, thermodynamics) / isothermal_∂μ∂S(θ, S, z, thermodynamics) # Γ_S
@inline heat_flux_prefactor(θ, S, z, thermodynamics)        = heat_capacity(θ, S, z, thermodynamics) / exner(θ, S, z, thermodynamics)            # cₚ / π = 1 / T_θ
@inline heat_flux_salt_coefficient(θ, S, z, thermodynamics) = - ∂T∂S(θ, S, z, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)                    # - T_S / T_θ

# ∇S + Γ_S 𝐤, the (scaled) isothermal salt force: ∇μ|_T = (∂μ/∂S)_T (∇S + Γ_S 𝐤), using the Maxwell relation
# (∂μ/∂z)_T,S = - (∂b/∂S)_T,z
@inline salt_gradient_x(i, j, k, grid, buoyancy_formulation, C) = ∂xᶠᶜᶜ(i, j, k, grid, C.S)
@inline salt_gradient_y(i, j, k, grid, buoyancy_formulation, C) = ∂yᶜᶠᶜ(i, j, k, grid, C.S)
@inline salt_gradient_z(i, j, k, grid, buoyancy_formulation, C) =
    ∂zᶜᶜᶠ(i, j, k, grid, C.S) +
    ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, barodiffusion_coefficient, buoyancy_formulation, C)

for (dir, ∂, ℑ, wall_mask) in ((:x, ∂xᶠᶜᶜ, ℑxᶠᵃᵃ, biharmonic_mask_x),
                               (:y, ∂yᶜᶠᶜ, ℑyᵃᶠᵃ, biharmonic_mask_y),
                               (:z, ∂zᶜᶜᶠ, ℑzᵃᵃᶠ, biharmonic_mask_z))
    salt_gradient    = Symbol(:salt_gradient_, dir)
    salt_flux        = Symbol(:salt_flux_, dir)
    heat_flux        = Symbol(:heat_flux_, dir)
    masked_salt_flux = Symbol(:masked_salt_flux_, dir)
    masked_heat_flux = Symbol(:masked_heat_flux_, dir)
    tracer_flux      = Symbol(:tracer_flux_, dir)
    diffusive_flux   = Symbol(:diffusive_flux_, dir)

    @eval begin
        @inline $salt_gradient(i, j, k, grid, buoyancy_formulation::ConstantSalinityThermodynamicBuoyancy, C) = zero(grid)

        # j_S = - κ_S (∇S + Γ_S 𝐤)
        @inline $salt_flux(i, j, k, grid, closure, buoyancy_formulation, C) =
            - closure.salt_diffusivity * $salt_gradient(i, j, k, grid, buoyancy_formulation, C)

        # j_θ = - κ_θ ℑ(cₚ / π) ∂T + ℑ(- T_S / T_θ) j_S
        @inline $heat_flux(i, j, k, grid, closure, buoyancy_formulation, C) =
            - closure.thermal_diffusivity *
              $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, heat_flux_prefactor, buoyancy_formulation, C) *
              $∂(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, temperature, buoyancy_formulation, C) +
            $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, heat_flux_salt_coefficient, buoyancy_formulation, C) *
            $salt_flux(i, j, k, grid, closure, buoyancy_formulation, C)

        # Zero at walls, where Oceananigans adds the boundary-condition flux instead (as for biharmonic closures)
        @inline $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, C) =
            $wall_mask(i, j, k, grid, $salt_flux, closure, buoyancy_formulation, C)
        @inline $masked_heat_flux(i, j, k, grid, closure, buoyancy_formulation, C) =
            $wall_mask(i, j, k, grid, $heat_flux, closure, buoyancy_formulation, C)

        # Flux of the tracer with the given name; zero for tracers other than θ and S
        @inline $tracer_flux(i, j, k, grid, ::Val{:θ}, closure, buoyancy_formulation, C) = $masked_heat_flux(i, j, k, grid, closure, buoyancy_formulation, C)
        @inline $tracer_flux(i, j, k, grid, ::Val{:S}, closure, buoyancy_formulation, C) = $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, C)
        @inline $tracer_flux(i, j, k, grid, ::Val,     closure, buoyancy_formulation, C) = zero(grid)

        # The Oceananigans interface: the tracer index `id` selects θ or S through the closure's tracer names
        @inline $diffusive_flux(i, j, k, grid, closure::ThermodynamicDiffusivity, K, id, c, clock, fields, b) =
            $tracer_flux(i, j, k, grid, tracer_name(closure, id), closure, b.formulation, fields)
    end
end
