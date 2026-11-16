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

#####
##### Wall fluxes: the fluxes that the model applies at the walls
#####

#=
The flux of a `FluxBoundaryCondition` at its wall, evaluated as `compute_flux_bc_tendencies!` adds it to the tendency
of the boundary cell (Oceananigans 0.116.1, `src/BoundaryConditions/compute_flux_bcs.jl:113-165`), with the arguments
`(clock, model_fields)` that the model passes (`compute_nonhydrostatic_tendencies.jl:192`): `getbc` on the x and y walls,
at the boundary indices (a, b) of the face, and `explicit_flux` on the bottom and top, at the boundary cell (i, j, k) of
the field data c. Only the explicit part of an implicit-explicit flux condition enters the model's tendency Gⁿ; its
implicit part, applied by the vertically implicit solver, is not included. The value is the face flux, positive in
+x, +y or +z. Every other condition, the set `NotFluxBC` that `compute_flux_bcs.jl` skips (line 23), adds no flux: the
wall flux of a Value or Gradient condition is carried by a Fickian closure (`closure_wall_flux_{x,y,z}`). The flux is
converted to the float type of the grid, as Oceananigans converts a `Number` condition.
=#
@inline lateral_boundary_flux(bc::BoundaryCondition{<:Flux}, a, b, grid, clock, model_fields) =
    convert(eltype(grid), getbc(bc, a, b, grid, clock, model_fields))
@inline lateral_boundary_flux(bc, a, b, grid, clock, model_fields) = zero(grid)

@inline vertical_boundary_flux(bc::BoundaryCondition{<:Flux}, side, i, j, k, grid, c, clock, model_fields) =
    convert(eltype(grid), explicit_flux(bc, side, i, j, k, grid, c, clock, model_fields))
@inline vertical_boundary_flux(bc, side, i, j, k, grid, c, clock, model_fields) = zero(grid)

#=
The flux of the boundary conditions `bcs` = (; west, east, south, north, bottom, top) of a tracer with data c at the face
(i, j, k): at a wall, face 1 of a direction whose cell 0 is outside the domain or face N + 1 whose cell N + 1 is, the
flux of the condition on that wall, and zero elsewhere. `inactive_cell` is the test of the wall masks
(`peripheral_node`, `src/Grids/inactive_node.jl:151-153`).
=#
@inline boundary_condition_flux_x(i, j, k, grid, bcs, c, clock, model_fields) =
    ifelse(inactive_cell(i - 1, j, k, grid), lateral_boundary_flux(bcs.west, j, k, grid, clock, model_fields),
           ifelse(inactive_cell(i, j, k, grid), lateral_boundary_flux(bcs.east, j, k, grid, clock, model_fields),
                  zero(grid)))

@inline boundary_condition_flux_y(i, j, k, grid, bcs, c, clock, model_fields) =
    ifelse(inactive_cell(i, j - 1, k, grid), lateral_boundary_flux(bcs.south, i, k, grid, clock, model_fields),
           ifelse(inactive_cell(i, j, k, grid), lateral_boundary_flux(bcs.north, i, k, grid, clock, model_fields),
                  zero(grid)))

@inline boundary_condition_flux_z(i, j, k, grid, bcs, c, clock, model_fields) =
    ifelse(inactive_cell(i, j, k - 1, grid),
           vertical_boundary_flux(bcs.bottom, Bottom(), i, j, 1, grid, c, clock, model_fields),
           ifelse(inactive_cell(i, j, k, grid),
                  vertical_boundary_flux(bcs.top, Top(), i, j, grid.Nz, grid, c, clock, model_fields),
                  zero(grid)))

#=
f in the boundary cell of the face (i, j, k): cell i where cell i - 1 lies outside the domain (face 1 of a wall), and
cell i - 1 elsewhere, which at face N + 1 of a wall is the boundary cell. The model applies a wall flux to the tendency
of the boundary cell, so that its contribution to b, 𝒜 and 𝒫 takes the coefficients b_θ, b_S, Υ, π and m of that cell.
Used only in products with a wall flux, which is zero away from walls.
=#
@inline boundary_cell_value_x(i, j, k, grid, f::F, args...) where F =
    f(ifelse(inactive_cell(i - 1, j, k, grid), i, i - 1), j, k, grid, args...)
@inline boundary_cell_value_y(i, j, k, grid, f::F, args...) where F =
    f(i, ifelse(inactive_cell(i, j - 1, k, grid), j, j - 1), k, grid, args...)
@inline boundary_cell_value_z(i, j, k, grid, f::F, args...) where F =
    f(i, j, ifelse(inactive_cell(i, j, k - 1, grid), k, k - 1), grid, args...)

# Generated for each direction, with `dir` and the operators of that direction:
#   salt_gradient_{x,y,z}, salt_flux_{x,y,z}, θ_flux_{x,y,z}, masked_salt_flux_{x,y,z}, masked_θ_flux_{x,y,z},
#   tracer_flux_{x,y,z}, the `ThermodynamicDiffusivity` methods of Oceananigans' diffusive_flux_{x,y,z},
#   closure_wall_flux_{x,y,z}, wall_θ_flux_{x,y,z}, wall_salt_flux_{x,y,z}, total_θ_flux_{x,y,z} and
#   total_salt_flux_{x,y,z}
for (dir, ∂, ℑ, wall_mask, wall_face) in ((:x, ∂xᶠᶜᶜ, ℑxᶠᵃᵃ, wall_mask_x, x_peripheral_node),
                                          (:y, ∂yᶜᶠᶜ, ℑyᵃᶠᵃ, wall_mask_y, y_peripheral_node),
                                          (:z, ∂zᶜᶜᶠ, ℑzᵃᵃᶠ, wall_mask_z, z_peripheral_node))
    salt_gradient           = Symbol(:salt_gradient_, dir)
    salt_flux_kernel        = Symbol(:salt_flux_, dir)
    θ_flux_kernel           = Symbol(:θ_flux_, dir)
    masked_salt_flux        = Symbol(:masked_salt_flux_, dir)
    masked_θ_flux           = Symbol(:masked_θ_flux_, dir)
    tracer_flux             = Symbol(:tracer_flux_, dir)
    diffusive_flux          = Symbol(:diffusive_flux_, dir)
    boundary_condition_flux = Symbol(:boundary_condition_flux_, dir)
    closure_wall_flux       = Symbol(:closure_wall_flux_, dir)
    wall_θ_flux             = Symbol(:wall_θ_flux_, dir)
    wall_salt_flux          = Symbol(:wall_salt_flux_, dir)
    total_θ_flux            = Symbol(:total_θ_flux_, dir)
    total_salt_flux         = Symbol(:total_salt_flux_, dir)

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
        Two fluxes, for both closures.

        The closure fluxes, masked to zero at walls: the `ThermodynamicDiffusivity`'s fluxes in the model, and the
        fluxes of σ_θ and of every face contraction j δX. Oceananigans adds the flux of a `FluxBoundaryCondition` to the
        boundary cell separately, as for biharmonic closures, so that carrying it in the closure as well would count it
        twice. The Fickian closure's own flux at a wall is zero for Flux conditions, including the default no flux, whose
        halo mirrors the interior (`src/BoundaryConditions/fill_halo_regions_flux.jl`). Value and Gradient conditions,
        which only Fickian models accept, extrapolate the halo (`src/BoundaryConditions/fill_halo_regions_value_gradient.jl`):
        the model's closure carries their wall flux, and the mask removes it from these fluxes.

        The total fluxes of the diagnostics: the closure fluxes at interior faces, and at wall faces the wall fluxes
        Q_θ and Q_S that the model applies, the flux of the `FluxBoundaryCondition` plus, for a Fickian closure, the
        closure's own flux at the wall. Then G_θ - σ_θ = - ∇·j_θ and G_S = - ∇·j_S with the tendencies of
        `compute_tendencies!` and `compute_flux_bc_tendencies!`, in every cell.
        =#
        @inline $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
            $wall_mask(i, j, k, grid, $salt_flux_kernel, closure, buoyancy_formulation, fields)
        @inline $masked_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields) =
            $wall_mask(i, j, k, grid, $θ_flux_kernel, closure, buoyancy_formulation, fields)

        # The closure's own flux of `flux` at the walls, zero elsewhere: the model applies the Fickian closure's flux
        # at a wall in `compute_tendencies!`, and masks the `ThermodynamicDiffusivity`'s
        @inline $closure_wall_flux(i, j, k, grid, flux::F, closure::ScalarDiffusivity, buoyancy_formulation, fields) where F =
            ifelse($wall_face(i, j, k, grid), flux(i, j, k, grid, closure, buoyancy_formulation, fields), zero(grid))
        @inline $closure_wall_flux(i, j, k, grid, flux, ::ThermodynamicDiffusivity, buoyancy_formulation, fields) =
            zero(grid)

        #=
        Q_θ and Q_S, the wall fluxes that the model applies, zero away from walls. `walls` holds the boundary
        conditions of θ and S, the clock and the data of the model fields (`wall_conditions(model)`). The
        constant-salinity model has no S, and no salt wall flux.
        =#
        @inline $wall_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
            $closure_wall_flux(i, j, k, grid, $θ_flux_kernel, closure, buoyancy_formulation, fields) +
            $boundary_condition_flux(i, j, k, grid, walls.θ, walls.fields.θ, walls.clock, walls.fields)
        @inline $wall_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
            $closure_wall_flux(i, j, k, grid, $salt_flux_kernel, closure, buoyancy_formulation, fields) +
            $boundary_condition_flux(i, j, k, grid, walls.S, walls.fields.S, walls.clock, walls.fields)
        @inline $wall_salt_flux(i, j, k, grid, closure, ::ConstantSalinityThermodynamicBuoyancy, fields, walls) =
            zero(grid)

        # j_θ and j_S of the diagnostics: the masked closure flux plus the wall flux
        @inline $total_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
            $masked_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields) +
            $wall_θ_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls)
        @inline $total_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls) =
            $masked_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields) +
            $wall_salt_flux(i, j, k, grid, closure, buoyancy_formulation, fields, walls)

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
