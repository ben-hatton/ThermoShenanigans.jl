#=
Lorenz reference state: the adiabatic, isohaline, volume-conserving rearrangement of the fluid that minimises
potential energy. The diagnostics built on it read only four model-grid fields, which a method of computing the
reference state must fill (see `AbstractReferenceStateMethod`). Apart from z★, they are held as data arrays with the
halos of a Field at their location, which keeps the types of the diagnostics small (see `diagnostics.jl`):

    z★          the reference height of each cell, at (Center, Center, Center),
    p★(z★)      the reference potential at each cell's reference height, at (Center, Center, Center),
    p★ᶜᶜᶜ       the reference potential at each level, at (Nothing, Nothing, Center),
    b★ᶜᶜᶠ       δz p★ᶜᶜᶜ / Δz at the w-faces, at (Nothing, Nothing, Face),

so that 𝒲★ = p★ᶜᶜᶜ - p★(z★) needs no knowledge of the method.

The reference term R★ = Ṗ★(z) - Ṗ★(z★) of the APE budget is formed in the same way from the tendency Ṗ★ = ∂p★/∂t of
the reference potential, due to the buoyancy tendency Ḃ = ADV_b + σ_b - ∇·j_b of every cell. In the continuum,
advection only rearranges the fluid and leaves the reference state unchanged, but the model's discrete advection
ADV_b = - b_θ ∇·(u θ) - b_S ∇·(u S) does not, so that Ṗ★ includes it:

    Ṗ★(z★)      at each cell's reference height, at (Center, Center, Center),
    Ṗ★ᶜᶜᶜ       at each level, at (Nothing, Nothing, Center).
=#

"""
    AbstractReferenceStateMethod

Abstract supertype of the methods that compute the Lorenz reference state. A method `M` implements

* `validate_reference_state_grid(::M, grid)`, which throws an `ArgumentError` for a grid it does not support;
* `reference_state_workspace(::M, grid)`, which returns the method's own storage (or `nothing`);
* `compute_reference_state!(z★, state, ::M)`, which, given the computed `state.buoyancy`, fills the interior points of
  the `Field` z★ and of the three data arrays that the diagnostics read:
  - `z★`, the reference height of each cell;
  - `state.parcel_reference_potential`, p★(z★) in each cell;
  - `state.reference_potential`, p★ at each level, at (Nothing, Nothing, Center);
  - `state.reference_buoyancy`, b★ = δz p★ / Δz at the interior w-faces (w = 0 on the boundary faces, so the
    boundary values only need to be finite).

Here p★ = ∫ b★ dz is the reference potential of the method's reference column, and the per-level and per-cell
values of p★ must be consistent with each other, so that 𝒲★ = p★ᶜᶜᶜ - p★(z★) integrates to zero and 𝒜 ≥ 0.
A method that bins the buoyancy, rather than sorting it, satisfies 𝒜 ≥ 0 only to within the bin width.

For the reference term R★ of the APE budget, a method also implements

* `reference_tendency_workspace(::M, grid)`, which returns the storage of the tendency step (or `nothing`);
* `compute_reference_tendency!(parcel_tendency, tendency, ::M)`, which, given the computed reference state and the
  computed buoyancy tendency `tendency.buoyancy_tendency`, fills the interior points of
  - `parcel_tendency`, Ṗ★(z★) in each cell;
  - `tendency.reference_tendency`, Ṗ★ at each level, at (Nothing, Nothing, Center),

  consistently with each other, so that R★ = Ṗ★ᶜᶜᶜ - Ṗ★(z★) integrates to zero.
"""
abstract type AbstractReferenceStateMethod end

"""
    ReferenceState

Operand of the reference height `Field` z★ returned by [`reference_height`](@ref). Its `compute!` computes the reference
state with `method`, so that one computation serves every diagnostic built on z★. Apart from `method`,
`buoyancy_formulation` and `workspace`, its fields are data arrays, with the halos of a `Field` at their location:

* `method`: the [`AbstractReferenceStateMethod`](@ref)
* `buoyancy_formulation`: the `ThermodynamicBuoyancy` of the model
* `tracers`: the data of the tracers of the model, from which b is computed
* `buoyancy`: the buoyancy b at (Center, Center, Center)
* `parcel_reference_potential`: p★(z★), the reference potential at each cell's reference height
* `reference_potential`: p★ at each level, at (Nothing, Nothing, Center)
* `reference_buoyancy`: b★ = δz p★ / Δz at w-faces, at (Nothing, Nothing, Face)
* `workspace`: storage specific to `method`
"""
struct ReferenceState{M, BF, T, B, H, R, F, W}
    method :: M
    buoyancy_formulation :: BF
    tracers :: T
    buoyancy :: B
    parcel_reference_potential :: H
    reference_potential :: R
    reference_buoyancy :: F
    workspace :: W
end

Base.summary(state::ReferenceState) = string("ReferenceState with ", summary(state.method))

const ReferenceHeightField = Field{<:Any, <:Any, <:Any, <:ReferenceState}

"""
    ReferenceProfile

Operand of the `Field`s, such as p★ and b★, that share their data arrays with the `ReferenceState` of z★.
Computing one computes z★, so that a profile handed to an output writer on its own is never stale.
"""
struct ReferenceProfile{Z}
    reference_height :: Z
end

Base.summary(::ReferenceProfile) = "ReferenceProfile"

const ReferenceProfileField = Field{<:Any, <:Any, <:Any, <:ReferenceProfile}

#####
##### Computing the reference state
#####

#=
The reference state and its tendency hold the model's own objects, and compute b and Ḃ into data arrays, rather
than holding operations over the model fields: like the diagnostics of Oceanostics.jl, every operation is then only
one level deep, which keeps the types, and so the compile times, of the diagnostics built on z★ small.
=#

@kernel function _compute_buoyancy!(buoyancy, grid, buoyancy_formulation, tracers)
    i, j, k = @index(Global, NTuple)
    @inbounds buoyancy[i, j, k] = buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy_formulation, tracers)
end

@kernel function _compute_buoyancy_tendency!(buoyancy_tendency, grid, advection, clock, fields, parameters)
    i, j, k = @index(Global, NTuple)
    @inbounds buoyancy_tendency[i, j, k] = total_buoyancy_tendencyᶜᶜᶜ(i, j, k, grid, advection, clock, fields, parameters)
end

function compute!(profile::ReferenceProfileField, time = nothing)
    compute_at!(profile.operand.reference_height, time)
    set_status!(profile.status, time)
    return profile
end

# Fill the halos of the data of a field at (Center, Center, Center) with the boundary conditions of z★
fill_center_halo_regions!(data, z★) =
    fill_halo_regions!(data, z★.boundary_conditions, z★.indices, (Center(), Center(), Center()), z★.grid)

function compute!(z★::ReferenceHeightField, time = nothing)
    state = z★.operand
    grid = z★.grid
    launch!(architecture(grid), grid, :xyz, _compute_buoyancy!, state.buoyancy, grid, state.buoyancy_formulation,
            state.tracers)
    fill_center_halo_regions!(state.buoyancy, z★)
    compute_reference_state!(z★, state, state.method)
    fill_halo_regions!(z★)
    fill_center_halo_regions!(state.parcel_reference_potential, z★)
    set_status!(z★.status, time)
    return z★
end

"""
    reference_height(model; method = Sorting())

Return the reference height z★ of each cell of a model built with [`thermodynamic_model_components`](@ref), as a
`Field` at (Center, Center, Center). Its operand is a [`ReferenceState`](@ref), and its `compute!` computes the
Lorenz reference state with `method`, an [`AbstractReferenceStateMethod`](@ref). The diagnostics of the reference
state and the local APE budget take z★: pass the same z★ to each, so that they share one computation of the
reference state per time. With [`Sorting`](@ref), the grid must be a `RectilinearGrid` regular in x, y and z.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 8), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> components = thermodynamic_model_components(grid, thermodynamics; viscosity = 1e-6, thermal_diffusivity = 1e-7,
                                                   salt_diffusivity = 1e-7);

julia> model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...);

julia> set!(model, θ = (x, z) -> 283 - z, S = 0.035);

julia> z★ = reference_height(model);

julia> 𝒜 = Field(available_potential_energy(z★));

julia> round(maximum(𝒜), sigdigits = 3) # an unstable column has available potential energy
0.000751
```
"""
function reference_height(model; method = Sorting())
    grid = model.grid
    validate_reference_state_grid(method, grid)

    state = ReferenceState(method, model.buoyancy.formulation, field_data(model.tracers), field_data_at(Center, Center, Center, grid),
                           field_data_at(Center, Center, Center, grid), field_data_at(Nothing, Nothing, Center, grid),
                           field_data_at(Nothing, Nothing, Face, grid), reference_state_workspace(method, grid))

    return Field{Center, Center, Center}(grid; operand = state, status = FieldStatus())
end

# The data array, with halos, of a Field at (LX, LY, LZ)
field_data_at(LX, LY, LZ, grid) = Field{LX, LY, LZ}(grid).data

# A profile Field at (LX, LY, LZ) that shares `data` with the reference state, and recomputes z★ when computed
reference_profile(z★, data, LX, LY, LZ) =
    Field{LX, LY, LZ}(z★.grid; data, operand = ReferenceProfile(z★), status = FieldStatus())

#####
##### The tendency of the reference state
#####

"""
    ReferenceTendency

Operand of the `Field` Ṗ★(z★), the tendency of the reference potential at each cell's reference height, used by
the reference term R★ of [`reference_tendency_work`](@ref). Its `compute!` computes the reference state, the
buoyancy tendency Ḃ = ADV_b + σ_b - ∇·j_b of the model, including its discrete advection, and then the tendency of
the reference state with the method of the reference state. Fields:

* `reference_height`: the reference height `Field` z★, whose operand is a [`ReferenceState`](@ref)
* `advection`, `clock`, `fields` and `parameters`: the model's advection schemes, clock, the data of its velocities
  and tracers, and the parameters of the production terms, from which Ḃ is computed
* `buoyancy_tendency`: the data of Ḃ at (Center, Center, Center)
* `reference_tendency`: the data of Ṗ★ at each level, at (Nothing, Nothing, Center)
* `workspace`: storage specific to the method
"""
struct ReferenceTendency{Z, A, C, F, P, B, R, W}
    reference_height :: Z
    advection :: A
    clock :: C
    fields :: F
    parameters :: P
    buoyancy_tendency :: B
    reference_tendency :: R
    workspace :: W
end

Base.summary(tendency::ReferenceTendency) =
    string("ReferenceTendency with ", summary(tendency.reference_height.operand.method))

const ReferenceTendencyField = Field{<:Any, <:Any, <:Any, <:ReferenceTendency}

reference_tendency_workspace(method, grid) = nothing

function compute!(parcel_tendency::ReferenceTendencyField, time = nothing)
    tendency = parcel_tendency.operand
    z★ = tendency.reference_height
    compute_at!(z★, time)
    grid = z★.grid
    launch!(architecture(grid), grid, :xyz, _compute_buoyancy_tendency!, tendency.buoyancy_tendency, grid,
            tendency.advection, tendency.clock, tendency.fields, tendency.parameters)
    compute_reference_tendency!(parcel_tendency, tendency, z★.operand.method)
    fill_halo_regions!(parcel_tendency)
    set_status!(parcel_tendency.status, time)
    return parcel_tendency
end

"""
    reference_tendency(model, z★)

Return the tendency Ṗ★(z★) of the reference potential at each cell's reference height, as a `Field` at
(Center, Center, Center), due to the buoyancy tendency Ḃ = ADV_b + σ_b - ∇·j_b of `model`, for the reference
height `Field` z★ returned by [`reference_height`](@ref). Its operand is a [`ReferenceTendency`](@ref), whose
`reference_tendency` holds Ṗ★ at each level.
"""
function reference_tendency(model, z★)
    grid = z★.grid
    method = z★.operand.method
    tendency = ReferenceTendency(z★, model.advection, model.clock, velocities_and_tracers(model), production_parameters(model),
                                 field_data_at(Center, Center, Center, grid), field_data_at(Nothing, Nothing, Center, grid),
                                 reference_tendency_workspace(method, grid))
    return Field{Center, Center, Center}(grid; operand = tendency, status = FieldStatus())
end
