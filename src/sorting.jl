#=
Sorting: since b = b(θ, S) does not depend on z, the reference state is found by sorting the cells by buoyancy and
stacking them from the bottom of the domain, densest first (Winters et al. 1995, J. Fluid Mech. 289).

On a regular grid every cell has the same volume, so the sorted column is a stack of N = Nx Ny Nz slots of
thickness h = Δz / (Nx Ny), and b★ is piecewise constant on the slots. Then

    z★ = Z★(b)     the mid-height of the band of slots whose buoyancy equals b (Winters et al. 1995, eq. 11,
                   with a half-weighted Heaviside function), so that z★ depends on b alone,
    p★(z) = ∫ b★ dz′ from the bottom, piecewise linear and exact on the slots.

The band of a cell with buoyancy b spans the slots n₁…n₂, where n₁ - 1 slots have b★ < b and n₂ slots have b★ ≤ b,
so that z★ = bottom + h (n₁ - 1 + n₂) / 2. Both counts are found by bisection of the sorted b★, cell by cell.

On the model grid, p★ᶜᶜᶜ at level k is the mean of p★ over the centres of the Nx Ny slots that the level spans,
and 𝒲★ = p★ᶜᶜᶜ - p★(z★). Since the cells' z★ fill the same slots (or the mid-heights of tied runs, where p★ is
linear), Σ 𝒲★ V = 0 exactly; since p★ is convex (b★ is non-decreasing), p★ᶜᶜᶜ ≥ p★(z) by Jensen's inequality,
so that 𝒜 ≥ 0; and a stable, horizontally uniform level has z★ = z and 𝒜 = 0.

At the model's w-faces, the reference buoyancy is b★ᶜᶜᶠ = δz p★ᶜᶜᶜ / Δz, the discrete gradient of p★ᶜᶜᶜ.
With discrete continuity this makes ∇·(p★ u) = ℑz(w b★ᶜᶜᶠ) hold exactly in every cell.

The tendency of the reference state follows from the buoyancy tendencies Ḃ of the cells, sorted with their
buoyancies into Ḃ★ (a key-value sort). Since p★ at the bottom face of slot n is h Σ_{m<n} b★[m], the sum of the
n - 1 lowest buoyancies, its tendency is h Ċ[n - 1], with Ċ[n] = Σ_{m≤n} Ḃ★[m]. Ċ at the ends of a band of tied
slots does not depend on the order of the tied cells, and within a band Ṗ★ is interpolated linearly between its
ends, which averages over the tied cells. Then

    Ṗ★(z★) = h (Ċ[n₁ - 1] + Ċ[n₂]) / 2,

and Ṗ★ᶜᶜᶜ at level k is the mean over the level's slots of Ṗ★ at the slot centres, so that R★ = Ṗ★ᶜᶜᶜ - Ṗ★(z★)
integrates to zero exactly, as 𝒲★ does. Without ties, Ṗ★ᶜᶜᶜ and Ṗ★(z★) are the exact time derivatives of p★ᶜᶜᶜ
and p★(z★).
=#

"""
    Sorting()

Compute the Lorenz reference state by sorting the cells by buoyancy and stacking them from the bottom of the domain
(Winters et al. 1995, J. Fluid Mech. 289). Cells of equal buoyancy share the mid-height of the band they fill, so that
z★ depends on b alone. Only a `RectilinearGrid` regular in x, y and z, without an immersed boundary, is supported.
This is the default method of [`reference_height`](@ref).

```jldoctest
julia> using ThermoShenanigans

julia> Sorting()
Sorting()
```
"""
struct Sorting <: AbstractReferenceStateMethod end

Base.summary(::Sorting) = "Sorting()"

"""
    SortingWorkspace

Storage of the [`Sorting`](@ref) method, on a sorted column of N = Nx Ny Nz slots of thickness h = Δz / (Nx Ny):

* `sorted_buoyancy`: b★ on the slots, densest first
* `slot_potential`: p★ at the bottom face of each slot
* `bottom`: height of the bottom of the domain
* `slot_thickness`: h
"""
struct SortingWorkspace{V, FT}
    sorted_buoyancy :: V
    slot_potential :: V
    bottom :: FT
    slot_thickness :: FT
end

validate_reference_state_grid(::Sorting, grid::XYZRegularRG) = nothing
validate_reference_state_grid(::Sorting, grid) =
    throw(ArgumentError("Sorting is only implemented for a RectilinearGrid that is regular in x, y and z, " *
                        "without an immersed boundary, but got a $(summary(grid))."))

function reference_state_workspace(::Sorting, grid)
    arch = architecture(grid)
    FT = eltype(grid)
    Nx, Ny, Nz = size(grid)
    N = Nx * Ny * Nz

    cpu_grid = on_architecture(CPU(), grid)
    bottom = convert(FT, znode(1, 1, 1, cpu_grid, Center(), Center(), Face()))
    slot_thickness = convert(FT, Δzᶜᶜᶜ(1, 1, 1, cpu_grid) / (Nx * Ny))

    return SortingWorkspace(on_architecture(arch, zeros(FT, N)), on_architecture(arch, zeros(FT, N)),
                            bottom, slot_thickness)
end

"""
    SortingTendencyWorkspace

Storage of the tendency step of the [`Sorting`](@ref) method, on the sorted column of N slots:

* `sorted_pairs`: the pairs (b, Ḃ) of the cells, sorted by b
* `cumulative_tendency`: Ċ[n] = Σ_{m≤n} Ḃ★[m]
* `sorted_tendency`: Ḃ★, the buoyancy tendencies in the order of b★
"""
struct SortingTendencyWorkspace{P, V}
    sorted_pairs :: P
    cumulative_tendency :: V
    sorted_tendency :: V
end

function reference_tendency_workspace(::Sorting, grid)
    arch = architecture(grid)
    FT = eltype(grid)
    N = prod(size(grid))
    return SortingTendencyWorkspace(on_architecture(arch, fill((zero(FT), zero(FT)), N)),
                                    on_architecture(arch, zeros(FT, N)), on_architecture(arch, zeros(FT, N)))
end

#####
##### Functions of the sorted column
#####

# p★(z) on the sorted column: P★ at the bottom face of the slot containing z, plus b★ across the rest of the slot.
# p★ is continuous, so a height on a slot face gives the same value from either neighbouring slot.
@inline function reference_potential_at(z, sorted_buoyancy, slot_potential, bottom, slot_thickness)
    N = length(sorted_buoyancy)
    n = clamp(unsafe_trunc(Int, (z - bottom) / slot_thickness) + 1, 1, N)
    return @inbounds slot_potential[n] + sorted_buoyancy[n] * (z - bottom - (n - 1) * slot_thickness)
end

# The number of slots n for which comparison(b★[n], b) holds, for comparison = < or ≤, by bisection: since b★ is
# non-decreasing, these are the leading slots 1…n_holds
@inline function count_sorted_slots(comparison::C, b, sorted_buoyancy) where C
    n_holds = 0                            # comparison holds on slots 1…n_holds
    n_fails = length(sorted_buoyancy) + 1  # and fails on slots n_fails…N
    while n_fails - n_holds > 1
        n = (n_holds + n_fails) ÷ 2
        holds = @inbounds comparison(sorted_buoyancy[n], b)
        n_holds = ifelse(holds, n, n_holds)
        n_fails = ifelse(holds, n_fails, n)
    end
    return n_holds
end

# Ṗ★ = h Ċ[n] at the top face of slot n, with Ċ[0] = 0
@inline face_reference_tendency(n, cumulative_tendency, slot_thickness) =
    @inbounds slot_thickness * ifelse(n == 0, zero(slot_thickness), cumulative_tendency[max(n, 1)])

# Ṗ★ at a fraction f of the way up the band of slots n₁…n₂, where n_below = n₁ - 1 and n_at_or_below = n₂
@inline function band_reference_tendency(f, n_below, n_at_or_below, cumulative_tendency, slot_thickness)
    Ṗ_bottom = face_reference_tendency(n_below, cumulative_tendency, slot_thickness)
    Ṗ_top = face_reference_tendency(n_at_or_below, cumulative_tendency, slot_thickness)
    return Ṗ_bottom + f * (Ṗ_top - Ṗ_bottom)
end

#####
##### Kernels
#####

# b of each cell, flattened in linear order
@kernel function _gather_buoyancy!(flat_buoyancy, buoyancy, cells)
    n = @index(Global)
    @inbounds flat_buoyancy[n] = buoyancy[cells[n]]
end

# z★ = bottom + h (n₁ - 1 + n₂) / 2, the mid-height of the band of slots with the cell's buoyancy, and p★(z★)
@kernel function _compute_reference_height!(z★, parcel_reference_potential, buoyancy, sorted_buoyancy, slot_potential,
                                            bottom, slot_thickness)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        b = buoyancy[i, j, k]
        n_below = count_sorted_slots(<, b, sorted_buoyancy)
        n_at_or_below = count_sorted_slots(≤, b, sorted_buoyancy)
        z = bottom + slot_thickness * (n_below + n_at_or_below) / 2
        z★[i, j, k] = z
        parcel_reference_potential[i, j, k] = reference_potential_at(z, sorted_buoyancy, slot_potential, bottom, slot_thickness)
    end
end

# p★ᶜᶜᶜ at level k: the mean over the level's slots of p★ at the slot centres, P★ + b★ h / 2
@kernel function _compute_level_reference_potential!(reference_potential, sorted_buoyancy, slot_potential, slot_thickness,
                                                     slots_per_level)
    k = @index(Global)
    level_potential = zero(slot_thickness)
    for m in 1:slots_per_level
        n = (k - 1) * slots_per_level + m
        level_potential += @inbounds slot_potential[n] + sorted_buoyancy[n] * slot_thickness / 2
    end
    @inbounds reference_potential[1, 1, k] = level_potential / slots_per_level
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

function compute_reference_state!(z★, state, ::Sorting)
    grid = z★.grid
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    workspace = state.workspace
    b★ = workspace.sorted_buoyancy
    P★ = workspace.slot_potential
    h = workspace.slot_thickness

    # b★: the buoyancy of every cell, densest (lowest b) first
    launch!(arch, grid, (Nx * Ny * Nz,), _gather_buoyancy!, b★, state.buoyancy, CartesianIndices(size(grid)))
    sort!(b★)

    # P★ at the bottom face of each slot, h Σ_{m<n} b★[m]
    cumsum!(P★, b★)
    P★ .= h .* (P★ .- b★)

    launch!(arch, grid, :xyz, _compute_reference_height!, z★.data, state.parcel_reference_potential, state.buoyancy, b★,
            P★, workspace.bottom, h)

    launch!(arch, grid, (Nz,), _compute_level_reference_potential!, state.reference_potential, b★, P★, h, Nx * Ny)

    launch!(arch, grid, (Nz + 1,), _compute_reference_buoyancy!, state.reference_buoyancy, grid, state.reference_potential,
            b★)

    return nothing
end

#####
##### The tendency of the reference state
#####

# (b, Ḃ) of each cell, flattened in linear order
@kernel function _gather_buoyancy_and_tendency!(pairs, buoyancy, buoyancy_tendency, cells)
    n = @index(Global)
    @inbounds pairs[n] = (buoyancy[cells[n]], buoyancy_tendency[cells[n]])
end

# Ṗ★(z★) = h (Ċ[n₁ - 1] + Ċ[n₂]) / 2, at the mid-height of the band of slots with the cell's buoyancy
@kernel function _compute_parcel_reference_tendency!(parcel_tendency, buoyancy, sorted_buoyancy, cumulative_tendency,
                                                     slot_thickness)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        b = buoyancy[i, j, k]
        n_below = count_sorted_slots(<, b, sorted_buoyancy)
        n_at_or_below = count_sorted_slots(≤, b, sorted_buoyancy)
        parcel_tendency[i, j, k] = band_reference_tendency(one(slot_thickness) / 2, n_below, n_at_or_below,
                                                           cumulative_tendency, slot_thickness)
    end
end

# Ṗ★ᶜᶜᶜ at level k: the mean over the level's slots of Ṗ★ at the slot centres, interpolated within each band of ties
@kernel function _compute_level_reference_tendency!(reference_tendency, sorted_buoyancy, cumulative_tendency,
                                                    slot_thickness, slots_per_level)
    k = @index(Global)
    FT = typeof(slot_thickness)
    level_tendency = zero(slot_thickness)
    for m in 1:slots_per_level
        n = (k - 1) * slots_per_level + m
        b = @inbounds sorted_buoyancy[n]
        n_below = count_sorted_slots(<, b, sorted_buoyancy)
        n_at_or_below = count_sorted_slots(≤, b, sorted_buoyancy)
        f = convert(FT, 2 * (n - n_below) - 1) / convert(FT, 2 * (n_at_or_below - n_below))
        level_tendency += band_reference_tendency(f, n_below, n_at_or_below, cumulative_tendency, slot_thickness)
    end
    @inbounds reference_tendency[1, 1, k] = level_tendency / slots_per_level
end

function compute_reference_tendency!(parcel_tendency, tendency, ::Sorting)
    grid = parcel_tendency.grid
    arch = architecture(grid)
    Nx, Ny, Nz = size(grid)
    state = tendency.reference_height.operand
    b★ = state.workspace.sorted_buoyancy
    h = state.workspace.slot_thickness
    pairs = tendency.workspace.sorted_pairs
    Ḃ★ = tendency.workspace.sorted_tendency
    Ċ = tendency.workspace.cumulative_tendency

    # Ḃ★: the buoyancy tendencies in the order of b★; the order within a run of tied buoyancies does not matter
    launch!(arch, grid, (Nx * Ny * Nz,), _gather_buoyancy_and_tendency!, pairs, state.buoyancy, tendency.buoyancy_tendency,
            CartesianIndices(size(grid)))
    sort!(pairs; by = first)
    Ḃ★ .= last.(pairs)
    cumsum!(Ċ, Ḃ★)

    launch!(arch, grid, :xyz, _compute_parcel_reference_tendency!, parcel_tendency.data, state.buoyancy, b★, Ċ, h)
    launch!(arch, grid, (Nz,), _compute_level_reference_tendency!, tendency.reference_tendency, b★, Ċ, h, Nx * Ny)

    return nothing
end
