#=
Lorenz reference state: the adiabatic, isohaline, volume-conserving rearrangement of the fluid that minimises
potential energy. Since b = b(θ, S) does not depend on z, it is found by sorting the cells by buoyancy and stacking
them from the bottom of the domain, densest first (Winters et al. 1995, J. Fluid Mech. 289).

On a regular grid every cell has the same volume, so the sorted column is a stack of N = Nx Ny Nz slots of
thickness h = Δz / (Nx Ny), and b★ is piecewise constant on the slots. Then

    z★ = Z★(b)     the mid-height of the band of slots whose buoyancy equals b (Winters et al. 1995, eq. 11,
                   with a half-weighted Heaviside function), so that z★ depends on b alone,
    p★(z) = ∫ b★ dz′ from the bottom, piecewise linear and exact on the slots.

On the model grid, p★ᶜᶜᶜ at level k is the mean of p★ over the centres of the Nx Ny slots that the level spans,
and 𝒲★ = p★ᶜᶜᶜ - p★(z★). Since the cells' z★ fill the same slots (or the mid-heights of tied runs, where p★ is
linear), Σ 𝒲★ V = 0 exactly; since p★ is convex (b★ is non-decreasing), p★ᶜᶜᶜ ≥ p★(z) by Jensen's inequality,
so that 𝒜 ≥ 0; and a stable, horizontally uniform level has z★ = z and 𝒜 = 0.

At the model's w-faces, the reference buoyancy is b★ᶜᶜᶠ = δz p★ᶜᶜᶜ / Δz, the discrete gradient of p★ᶜᶜᶜ.
With discrete continuity this makes ∇·(p★ u) = ℑz(w b★ᶜᶜᶠ) hold exactly in every cell.
=#

"""
    ReferenceState

Operand of the reference height `Field` z★ returned by [`energetics`](@ref). Its `compute!` sorts the buoyancy
and fills z★, the reference potential p★ at cell centres and the reference buoyancy b★ at w-faces, so that one
sort serves every diagnostic built on z★. Fields:

* `buoyancy`: the buoyancy b at (Center, Center, Center), as a computed `Field`
* `permutation`: the linear cell index of each slot of the sorted column, densest first
* `sorted_buoyancy`: b★ on the slots
* `slot_potential`: p★ at the bottom face of each slot
* `sorted_height`: z★ on the slots
* `reference_potential`: p★ averaged over each level, at (Nothing, Nothing, Center)
* `reference_buoyancy`: b★ at w-faces, at (Nothing, Nothing, Face)
* `bottom`: height of the bottom of the domain
* `slot_thickness`: thickness h = Δz / (Nx Ny) of each slot
"""
struct ReferenceState{B, P, V, R, F, FT}
    buoyancy :: B
    permutation :: P
    sorted_buoyancy :: V
    slot_potential :: V
    sorted_height :: V
    reference_potential :: R
    reference_buoyancy :: F
    bottom :: FT
    slot_thickness :: FT
end

Base.summary(::ReferenceState) = "ReferenceState"

const ReferenceHeightField = Field{<:Any, <:Any, <:Any, <:ReferenceState}

"""
    ReferenceProfile

Operand of the p★ and b★ profile `Field`s, which share their data with the `ReferenceState` of z★.
Computing a profile computes z★, so that a profile handed to an output writer on its own is never stale.
"""
struct ReferenceProfile{Z}
    reference_height :: Z
end

Base.summary(::ReferenceProfile) = "ReferenceProfile"

const ReferenceProfileField = Field{<:Any, <:Any, <:Any, <:ReferenceProfile}

function compute!(profile::ReferenceProfileField, time = nothing)
    compute_at!(profile.operand.reference_height, time)
    set_status!(profile.status, time)
    return profile
end

# p★(z) on the sorted column: P★ at the bottom face of the slot containing z, plus b★ across the rest of the slot.
# p★ is continuous, so a height on a slot face gives the same value from either neighbouring slot.
@inline function reference_potential_at(z, sorted_buoyancy, slot_potential, bottom, slot_thickness)
    N = length(sorted_buoyancy)
    n = clamp(unsafe_trunc(Int, (z - bottom) / slot_thickness) + 1, 1, N)
    return @inbounds slot_potential[n] + sorted_buoyancy[n] * (z - bottom - (n - 1) * slot_thickness)
end

# b of each cell, flattened in linear order
@kernel function _gather_buoyancy!(flat_buoyancy, buoyancy, cells)
    n = @index(Global)
    @inbounds flat_buoyancy[n] = buoyancy[cells[n]]
end

# b★ on the slots, b★[n] = b[perm[n]]
@kernel function _permute_buoyancy!(sorted_buoyancy, flat_buoyancy, permutation)
    n = @index(Global)
    @inbounds sorted_buoyancy[n] = flat_buoyancy[permutation[n]]
end

# z★ of the cell that fills slot n
@kernel function _scatter_reference_height!(reference_height, sorted_height, permutation, cells)
    n = @index(Global)
    @inbounds reference_height[cells[permutation[n]]] = sorted_height[n]
end

# b★ = δz p★ / Δz at w-faces; on the boundary faces, where w = 0, b★ is the boundary slot's
@kernel function _compute_reference_buoyancy!(reference_buoyancy, grid, reference_potential, sorted_buoyancy)
    k = @index(Global)
    Nz = size(grid, 3)
    N = length(sorted_buoyancy)
    @inbounds reference_buoyancy[1, 1, k] = ifelse(k == 1, sorted_buoyancy[1],
                                            ifelse(k == Nz + 1, sorted_buoyancy[N],
                                                   (reference_potential[1, 1, min(k, Nz)] - reference_potential[1, 1, max(k - 1, 1)]) /
                                                   Δzᶜᶜᶠ(1, 1, k, grid)))
end

function compute!(z★::ReferenceHeightField, time = nothing)
    state = z★.operand
    grid = z★.grid
    arch = architecture(grid)
    N = length(state.permutation)
    h = state.slot_thickness

    compute_at!(state.buoyancy, time)
    cells = CartesianIndices(size(grid))

    # Sort the cells by buoyancy, densest (lowest b) first; sorted_height holds the flattened b until it is overwritten
    flat_buoyancy = state.sorted_height
    launch!(arch, grid, (N,), _gather_buoyancy!, flat_buoyancy, state.buoyancy.data, cells)
    sortperm!(state.permutation, flat_buoyancy)
    launch!(arch, grid, (N,), _permute_buoyancy!, state.sorted_buoyancy, flat_buoyancy, state.permutation)

    # P★ at the bottom face of each slot, h Σ_{m<n} b★[m]
    b★ = state.sorted_buoyancy
    P★ = state.slot_potential
    cumsum!(P★, b★)
    P★ .= h .* (P★ .- b★)

    # A tied run of slots n₁…n₂ fills [(n₁ - 1) h, n₂ h] above the bottom, and all of its cells take the mid-height.
    # The run's bottom is the last run start at or below n (a running maximum) and its top the first run end at or
    # above n (a running minimum from the top).
    slots = 1:N
    starts = (slots .== 1) .| (b★ .!= circshift(b★, 1))
    ends = (slots .== N) .| (b★ .!= circshift(b★, -1))
    run_bottom = accumulate(max, ifelse.(starts, (slots .- 1) .* h, -h))
    run_top = reverse(accumulate(min, reverse(ifelse.(ends, slots .* h, (N + 1) * h))))
    state.sorted_height .= state.bottom .+ (run_bottom .+ run_top) ./ 2

    launch!(arch, grid, (N,), _scatter_reference_height!, z★.data, state.sorted_height, state.permutation, cells)

    # p★ᶜᶜᶜ: the mean over each level's slots of p★ at the slot centres, P★ + b★ h / 2
    Nx, Ny, Nz = size(grid)
    slot_centre_potential = reshape(P★ .+ b★ .* (h / 2), Nx * Ny, Nz)
    interior(state.reference_potential) .= reshape(sum(slot_centre_potential; dims = 1) ./ (Nx * Ny), 1, 1, Nz)

    launch!(arch, grid, (Nz + 1,), _compute_reference_buoyancy!, state.reference_buoyancy.data, grid, state.reference_potential.data, b★)

    fill_halo_regions!(z★)
    set_status!(z★.status, time)

    return z★
end

validate_reference_state_grid(grid::XYZRegularRG) = nothing
validate_reference_state_grid(grid) =
    throw(ArgumentError("The reference state is only implemented for a RectilinearGrid that is regular in x, y and z, " *
                        "without an immersed boundary, but got a $(summary(grid))."))

"""
    reference_height(buoyancy)

Return the reference height z★ of each cell, as a `Field` at (Center, Center, Center) on the grid of `buoyancy`,
a buoyancy `Field` or operation at (Center, Center, Center). Its operand is a [`ReferenceState`](@ref), and
`compute!` re-sorts the buoyancy.
"""
function reference_height(buoyancy)
    grid = buoyancy.grid
    validate_reference_state_grid(grid)

    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    N = Nx * Ny * Nz

    cpu_grid = on_architecture(CPU(), grid)
    bottom = convert(FT, znode(1, 1, 1, cpu_grid, Center(), Center(), Face()))
    slot_thickness = convert(FT, Δzᶜᶜᶜ(1, 1, 1, cpu_grid) / (Nx * Ny))

    sorted_vector() = on_architecture(arch, zeros(FT, N))
    reference_potential = Field{Nothing, Nothing, Center}(grid)
    reference_buoyancy = Field{Nothing, Nothing, Face}(grid)

    state = ReferenceState(Field(buoyancy), on_architecture(arch, zeros(Int, N)),
                           sorted_vector(), sorted_vector(), sorted_vector(),
                           reference_potential, reference_buoyancy, bottom, slot_thickness)

    return Field{Center, Center, Center}(grid; operand = state, status = FieldStatus())
end

# Profile Fields that share the data of the reference state and recompute z★ when computed
function reference_profile(z★, profile::Field{LX, LY, LZ}) where {LX, LY, LZ}
    return Field{LX, LY, LZ}(profile.grid; data = profile.data, operand = ReferenceProfile(z★), status = FieldStatus())
end
