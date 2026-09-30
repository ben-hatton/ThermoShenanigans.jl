struct ThermodynamicDiffusivity{FT, N} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization, 1}
    thermal_diffusivity :: FT # κ_T
    salt_diffusivity :: FT    # κ_S
end

"""
    ThermodynamicDiffusivity([FT;] thermal_diffusivity, salt_diffusivity)

Return a closure for the thermodynamically consistent heat and salt fluxes

    j_θ = - κ_T (cₚ / π) ∇T + (∂θ/∂S)_T j_S
    j_S = - κ_S (∇S - Γ ẑ),   Γ = (∂b/∂S)_T / (∂m/∂S)_T

Use it with a [`ThermodynamicBuoyancy`](@ref), and a separate `ScalarDiffusivity(ν = ν)` for viscosity.

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

#####
##### Fluxes at faces, from differences and interpolations of cell-centre thermodynamic variables
#####

@inline barodiffusion_coefficient(θ, S, z, t) = isothermal_∂b∂S(θ, S, z, t) / isothermal_∂m∂S(θ, S, z, t) # Γ
@inline heat_flux_prefactor(θ, S, z, t)       = heat_capacity(θ, S, z, t) / exner(θ, S, z, t)            # cₚ / π

# j_S = - κ_S (∇S - Γ ẑ)
@inline salt_flux_x(i, j, k, grid, cl, tb, C) = - cl.salt_diffusivity * ∂xᶠᶜᶜ(i, j, k, grid, C.S)
@inline salt_flux_y(i, j, k, grid, cl, tb, C) = - cl.salt_diffusivity * ∂yᶜᶠᶜ(i, j, k, grid, C.S)
@inline salt_flux_z(i, j, k, grid, cl, tb, C) = - cl.salt_diffusivity * (∂zᶜᶜᶠ(i, j, k, grid, C.S) -
                                                                         ℑzᵃᵃᶠ(i, j, k, grid, height_dependent_functionᶜᶜᶜ, barodiffusion_coefficient, tb, C))

for (dir, ∂, ℑ) in ((:x, :∂xᶠᶜᶜ, :ℑxᶠᵃᵃ), (:y, :∂yᶜᶠᶜ, :ℑyᵃᶠᵃ), (:z, :∂zᶜᶜᶠ, :ℑzᵃᵃᶠ))
    salt_flux      = Symbol(:salt_flux_, dir)
    heat_flux      = Symbol(:heat_flux_, dir)
    tracer_flux    = Symbol(:tracer_flux_, dir)
    diffusive_flux = Symbol(:diffusive_flux_, dir)
    wall_mask      = Symbol(:biharmonic_mask_, dir)

    @eval begin
        # j_θ = - κ_T ℑ(cₚ / π) ∂T + ℑ(∂θ/∂S)_T j_S
        @inline $heat_flux(i, j, k, grid, cl, tb, C) =
            - cl.thermal_diffusivity * $ℑ(i, j, k, grid, height_dependent_functionᶜᶜᶜ, heat_flux_prefactor, tb, C) *
                                       $∂(i, j, k, grid, height_dependent_functionᶜᶜᶜ, in_situ_temperature, tb, C) +
            $ℑ(i, j, k, grid, height_dependent_functionᶜᶜᶜ, isothermal_∂θ∂S, tb, C) * $salt_flux(i, j, k, grid, cl, tb, C)

        @inline $salt_flux(i, j, k, grid, cl, tb::ConstantSalinityThermodynamicBuoyancy, C) = zero(grid)

        @inline $tracer_flux(i, j, k, grid, ::Val{:θ}, cl, tb, C) = $heat_flux(i, j, k, grid, cl, tb, C)
        @inline $tracer_flux(i, j, k, grid, ::Val{:S}, cl, tb, C) = $salt_flux(i, j, k, grid, cl, tb, C)
        @inline $tracer_flux(i, j, k, grid, ::Val,     cl, tb, C) = zero(grid)

        # Zero at walls, where Oceananigans adds the boundary-condition flux instead (as for biharmonic closures)
        @inline $diffusive_flux(i, j, k, grid, cl::ThermodynamicDiffusivity, K, id, c, clock, fields, b) =
            $wall_mask(i, j, k, grid, $tracer_flux, tracer_name(cl, id), cl, b.formulation, fields)
    end
end
