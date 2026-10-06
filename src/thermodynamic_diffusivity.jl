struct ThermodynamicDiffusivity{FT, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    thermal_diffusivity :: FT # κ_T
    salt_diffusivity :: FT    # κ_S
end

"""
    ThermodynamicDiffusivity([FT;] thermal_diffusivity, salt_diffusivity)

Return a closure for the thermodynamically consistent heat and salt fluxes

    j_θ = - κ_T (cₚ / π) ∇T - (T_S / T_θ) j_S
    j_S = - κ_S (∇S - Γ 𝐤),   Γ = (∂b/∂S)_T / (∂μ/∂S)_T

Use it with a [`ThermodynamicBuoyancy`](@ref), and a separate closure for viscosity.

```jldoctest
julia> using ThermoShenanigans

julia> ThermodynamicDiffusivity(thermal_diffusivity = 1.4e-7, salt_diffusivity = 1e-9)
ThermodynamicDiffusivity(κ_T=1.4e-7, κ_S=1.0e-9)
```
"""
ThermodynamicDiffusivity(FT = Oceananigans.defaults.FloatType; thermal_diffusivity, salt_diffusivity) =
    ThermodynamicDiffusivity{FT, nothing}(convert(FT, thermal_diffusivity), convert(FT, salt_diffusivity))

# Record the tracer names, so that the tracer index `id` identifies θ and S at compile time
with_tracers(tracers, cl::ThermodynamicDiffusivity{FT}) where FT = ThermodynamicDiffusivity{FT, tracers}(cl.thermal_diffusivity, cl.salt_diffusivity)

@inline tracer_name(::ThermodynamicDiffusivity{FT, N}, ::Val{id}) where {FT, N, id} = Val(N[id])

# Δ² / max(κ_T, κ_S) for `TimeStepWizard(; diffusive_cfl)` and `DiffusiveCFL`; since T = π / η⁰_θ, j_θ diffuses θ with exactly κ_T
cell_diffusion_timescale(cl::ThermodynamicDiffusivity, closure_fields, grid, clock, fields) =
    min_Δxyz(grid, ThreeDimensionalFormulation())^2 / max(cl.thermal_diffusivity, cl.salt_diffusivity)

#####
##### Fluxes at faces, from differences and interpolations of cell-centre thermodynamic variables
#####

@inline barodiffusion_coefficient(θ, S, z, t)  = isothermal_∂b∂S(θ, S, z, t) / isothermal_∂μ∂S(θ, S, z, t) # Γ
@inline heat_flux_prefactor(θ, S, z, t)        = heat_capacity(θ, S, z, t) / exner(θ, S, z, t)            # cₚ / π
@inline heat_flux_salt_coefficient(θ, S, z, t) = - ∂T∂S(θ, S, z, t) / ∂T∂θ(θ, S, z, t)                    # - T_S / T_θ

# ∇S - Γ 𝐤, the (scaled) isothermal salt force: ∇μ|_T = (∂μ/∂S)_T (∇S - Γ 𝐤)
@inline salt_gradient_x(i, j, k, grid, tb, C) = ∂xᶠᶜᶜ(i, j, k, grid, C.S)
@inline salt_gradient_y(i, j, k, grid, tb, C) = ∂yᶜᶠᶜ(i, j, k, grid, C.S)
@inline salt_gradient_z(i, j, k, grid, tb, C) = ∂zᶜᶜᶠ(i, j, k, grid, C.S) -
                                                ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, barodiffusion_coefficient, tb, C)

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
        @inline $salt_gradient(i, j, k, grid, tb::ConstantSalinityThermodynamicBuoyancy, C) = zero(grid)

        # j_S = - κ_S (∇S - Γ 𝐤)
        @inline $salt_flux(i, j, k, grid, cl, tb, C) = - cl.salt_diffusivity * $salt_gradient(i, j, k, grid, tb, C)

        # j_θ = - κ_T ℑ(cₚ / π) ∂T + ℑ(- T_S / T_θ) j_S
        @inline $heat_flux(i, j, k, grid, cl, tb, C) =
            - cl.thermal_diffusivity * $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, heat_flux_prefactor, tb, C) *
                                       $∂(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, temperature, tb, C) +
            $ℑ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, heat_flux_salt_coefficient, tb, C) * $salt_flux(i, j, k, grid, cl, tb, C)

        # Zero at walls, where Oceananigans adds the boundary-condition flux instead (as for biharmonic closures)
        @inline $masked_salt_flux(i, j, k, grid, cl, tb, C) = $wall_mask(i, j, k, grid, $salt_flux, cl, tb, C)
        @inline $masked_heat_flux(i, j, k, grid, cl, tb, C) = $wall_mask(i, j, k, grid, $heat_flux, cl, tb, C)

        @inline $tracer_flux(i, j, k, grid, ::Val{:θ}, cl, tb, C) = $masked_heat_flux(i, j, k, grid, cl, tb, C)
        @inline $tracer_flux(i, j, k, grid, ::Val{:S}, cl, tb, C) = $masked_salt_flux(i, j, k, grid, cl, tb, C)
        @inline $tracer_flux(i, j, k, grid, ::Val,     cl, tb, C) = zero(grid)

        @inline $diffusive_flux(i, j, k, grid, cl::ThermodynamicDiffusivity, K, id, c, clock, fields, b) =
            $tracer_flux(i, j, k, grid, tracer_name(cl, id), cl, b.formulation, fields)
    end
end
