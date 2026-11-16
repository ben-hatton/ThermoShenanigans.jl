#=
Sorting: since b = b(θ, S) does not depend on z, the reference state is found by sorting the cells by buoyancy and
stacking them from the bottom of the domain, densest first (Winters et al. 1995, J. Fluid Mech. 289).

On a regular grid every cell has the same volume, so the sorted column is a stack of N = Nx Ny Nz slots of
thickness h = Δz / (Nx Ny), and b★ is piecewise constant on the slots. The cells are ranked once per computation of
the reference state: `sortperm!` gives the permutation that orders the flattened buoyancy of the cells, and b★ is
gathered through it. Then

    z★ = Z★(b)     the mid-height of the band of slots whose buoyancy equals b (Winters et al. 1995, eq. 11,
                   with a half-weighted Heaviside function), so that z★ depends on b alone,
    Ψ(z) = ∫ b★ dz′ from the bottom, the reference potential p★, piecewise linear and exact on the slots.

The band of a cell spans the slots n₁…n₂ of its run of tied b★. With P★[n] = h Σ_{m<n} b★[m], Ψ at the bottom face of
slot n, and since Ψ is linear across the band,

    z★ = bottom + h (n₁ - 1 + n₂) / 2,
    p★(z★) = P★[n₁] + b h (n₂ - n₁ + 1) / 2.

The bands are found by scans over the ends of the runs of tied b★, after Oceanostics.jl v0.21.7 (`HeavisideIntegral`,
src/BackgroundPotentialEnergyEquation.jl): n₁ of a slot is the running maximum of the run starts below and at it, and
n₂ the running minimum, taken from the top, of the run ends above and at it. Each cell keeps the band of its slot,
scattered through the permutation.

On the model grid, p★ᶜᶜᶜ at level k is the mean of Ψ over the centres of the Nx Ny slots that the level spans, which,
since Ψ is linear on each slot, is the exact cell average of Ψ. The means of all levels are one `sum!` over the first
dimension of the values at the slot centres, arranged as an (Nx Ny) × Nz array. Then 𝒲★ = p★ᶜᶜᶜ - p★(z★). Since the
cells' z★ fill the same slots (or the mid-heights of tied runs, where Ψ is linear), Σ 𝒲★ V = 0 exactly; since Ψ is
convex (b★ is non-decreasing), p★ᶜᶜᶜ ≥ p★(z) by Jensen's inequality, so that 𝒜 ≥ 0; and a stable, horizontally
uniform level has z★ = z and 𝒜 = 0.

At the model's w-faces, the reference buoyancy is b★ᶜᶜᶠ = δz p★ᶜᶜᶜ / Δz, the discrete gradient of p★ᶜᶜᶜ.
With discrete continuity this makes ∇·(p★ u) = ℑz(w b★ᶜᶜᶠ) hold exactly in every cell.

The tendency of the reference state follows from the buoyancy tendencies Ḃ of the cells, wall fluxes of θ and S
included, gathered through the permutation of the reference state into Ḃ★, in the order of b★. Since P★[n] is h times
the sum of the n - 1 lowest buoyancies, its tendency is h Ċ[n - 1], with Ċ[n] = Σ_{m≤n} Ḃ★[m]. Ċ at the ends of a band
of tied slots does not depend on the order of the tied cells, and within a band Ṗ★ is interpolated linearly between
its ends, which averages over the tied cells. Then

    Ṗ★(z★) = h (Ċ[n₁ - 1] + Ċ[n₂]) / 2,

and Ṗ★ᶜᶜᶜ at level k is the mean over the level's slots of Ṗ★ at the slot centres, taken with the same `sum!`, so that
R★ = Ṗ★ᶜᶜᶜ - Ṗ★(z★) integrates to zero exactly, as 𝒲★ does. Without ties, Ṗ★ᶜᶜᶜ and Ṗ★(z★) are the exact time
derivatives of p★ᶜᶜᶜ and p★(z★).
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

* `flat_values`: N values in linear order: the flattened b or Ḃ of the cells, from which b★ or Ḃ★ is gathered, and
  then Ψ or Ṗ★ at the slot centres, which are averaged over each level
* `permutation`: the linear indices of the flattened cells, in the order of b★
* `slot_indices`: N slot indices in the order of b★: the scratch of the sort on the CPU, and then the first or the
  last slot of the band of each slot
* `sorted_buoyancy`: b★ on the slots, densest first
* `slot_potential`: P★, Ψ at the bottom face of each slot
* `band_first`, `band_last`: the first and the last slot, n₁ and n₂, of the band of each cell, of the size of the grid
* `level_sum`: the sums over the Nx Ny slots of each level, of size (1, Nz)
* `bottom`: height of the bottom of the domain
* `slot_thickness`: h
"""
struct SortingWorkspace{V, P, B, L, FT}
    flat_values :: V
    permutation :: P
    slot_indices :: P
    sorted_buoyancy :: V
    slot_potential :: V
    band_first :: B
    band_last :: B
    level_sum :: L
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

    return SortingWorkspace(on_architecture(arch, zeros(FT, N)), on_architecture(arch, zeros(Int, N)),
                            on_architecture(arch, zeros(Int, N)), on_architecture(arch, zeros(FT, N)),
                            on_architecture(arch, zeros(FT, N)), on_architecture(arch, zeros(Int32, Nx, Ny, Nz)),
                            on_architecture(arch, zeros(Int32, Nx, Ny, Nz)), on_architecture(arch, zeros(FT, 1, Nz)),
                            bottom, slot_thickness)
end

"""
    SortingTendencyWorkspace

Storage of the tendency step of the [`Sorting`](@ref) method, on the sorted column of N slots:

* `sorted_tendency`: Ḃ★, the buoyancy tendencies in the order of b★
* `cumulative_tendency`: Ċ[n] = Σ_{m≤n} Ḃ★[m]
"""
struct SortingTendencyWorkspace{V}
    sorted_tendency :: V
    cumulative_tendency :: V
end

function reference_tendency_workspace(::Sorting, grid)
    arch = architecture(grid)
    FT = eltype(grid)
    N = prod(size(grid))
    return SortingTendencyWorkspace(on_architecture(arch, zeros(FT, N)), on_architecture(arch, zeros(FT, N)))
end

#####
##### Functions of the sorted column
#####

# The interior of `data`, the OffsetArray, with halos, of a Field at (Center, Center, Center), flattened in linear order
# into `flat`
flatten_interior!(flat, data, grid) = reshape(flat, size(grid)) .= interior(data, (Center, Center, Center), grid, (:, :, :))

# The permutation that orders `values`, into `permutation`. On the CPU, Base's `sortperm!` sorts through `scratch`, so that it allocates no
# array of the size of the grid
rank_cells!(permutation::Array, values::Array, scratch::Array) = sortperm!(permutation, values; scratch)
rank_cells!(permutation, values, scratch) = sortperm!(permutation, values)

# The band n₁…n₂ of the run of tied b★ of each slot, scattered through the permutation to the cells. Slot 1 and every
# slot n with b★[n] ≠ b★[n - 1] start a run, and n₁ is the running maximum of the run starts; slot N and every slot n
# with b★[n] ≠ b★[n + 1] end a run, and n₂ is the running minimum of the run ends, taken from the top
function compute_bands!(band_first, band_last, slot_indices, sorted_buoyancy, permutation)
    N = length(sorted_buoyancy)
    b★ = sorted_buoyancy

    fill!(view(slot_indices, 1:1), 1)
    view(slot_indices, 2:N) .= ifelse.(view(b★, 2:N) .!= view(b★, 1:N-1), 2:N, 1)
    accumulate!(max, slot_indices, slot_indices)
    vec(band_first)[permutation] = slot_indices

    fill!(view(slot_indices, N:N), N)
    view(slot_indices, 1:N-1) .= ifelse.(view(b★, 1:N-1) .!= view(b★, 2:N), 1:N-1, N)
    reverse!(slot_indices)
    accumulate!(min, slot_indices, slot_indices)
    reverse!(slot_indices)
    vec(band_last)[permutation] = slot_indices

    return nothing
end

# The means over the Nx Ny slots of each level of `slot_values`, into `level_data` at (Nothing, Nothing, Center)
function level_mean!(level_data, slot_values, level_sum, grid)
    Nx, Ny, Nz = size(grid)
    sum!(level_sum, reshape(slot_values, Nx * Ny, Nz))
    interior(level_data, (Nothing, Nothing, Center), grid, (:, :, :)) .= reshape(level_sum, 1, 1, Nz) ./ (Nx * Ny)
    return nothing
end

# Ṗ★ = h Ċ[n] at the top face of slot n, with Ċ[0] = 0
@inline face_reference_tendency(n, cumulative_tendency, slot_thickness) =
    @inbounds slot_thickness * ifelse(n == 0, zero(slot_thickness), cumulative_tendency[max(n, 1)])

# Ṗ★ at a fraction f of the way up the band of slots n₁…n₂
@inline function band_reference_tendency(f, n₁, n₂, cumulative_tendency, slot_thickness)
    Ṗ_bottom = face_reference_tendency(n₁ - 1, cumulative_tendency, slot_thickness)
    Ṗ_top = face_reference_tendency(n₂, cumulative_tendency, slot_thickness)
    return Ṗ_bottom + f * (Ṗ_top - Ṗ_bottom)
end

# Ṗ★ at the centre of slot n, in the band of slots n₁…n₂
@inline function slot_reference_tendency(n, n₁, n₂, cumulative_tendency, slot_thickness)
    FT = typeof(slot_thickness)
    f = convert(FT, 2 * (n - n₁) + 1) / convert(FT, 2 * (n₂ - n₁ + 1))
    return band_reference_tendency(f, n₁, n₂, cumulative_tendency, slot_thickness)
end

#####
##### Kernels
#####

# z★ = bottom + h (n₁ - 1 + n₂) / 2, the mid-height of the band of slots n₁…n₂ of the cell, and
# p★(z★) = P★[n₁] + b h (n₂ - n₁ + 1) / 2, since b★ = b across the band
@kernel function _compute_reference_height!(z★, parcel_reference_potential, buoyancy, band_first, band_last,
                                            slot_potential, bottom, slot_thickness)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        n₁ = band_first[i, j, k]
        n₂ = band_last[i, j, k]
        z★[i, j, k] = bottom + slot_thickness * (n₁ - 1 + n₂) / 2
        parcel_reference_potential[i, j, k] = slot_potential[n₁] + buoyancy[i, j, k] * slot_thickness * (n₂ - n₁ + 1) / 2
    end
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
    Nz = size(grid, 3)
    workspace = state.workspace
    b★ = workspace.sorted_buoyancy
    P★ = workspace.slot_potential
    h = workspace.slot_thickness

    # b★: the buoyancy of every cell, densest (lowest b) first, gathered through the permutation that ranks the cells
    flatten_interior!(workspace.flat_values, state.buoyancy, grid)
    rank_cells!(workspace.permutation, workspace.flat_values, workspace.slot_indices)
    b★ .= view(workspace.flat_values, workspace.permutation)

    # P★ at the bottom face of each slot, h Σ_{m<n} b★[m]
    cumsum!(P★, b★)
    P★ .= h .* (P★ .- b★)

    compute_bands!(workspace.band_first, workspace.band_last, workspace.slot_indices, b★, workspace.permutation)

    launch!(arch, grid, :xyz, _compute_reference_height!, z★.data, state.parcel_reference_potential, state.buoyancy,
            workspace.band_first, workspace.band_last, P★, workspace.bottom, h)

    # p★ᶜᶜᶜ: the mean over each level's slots of Ψ at the slot centres, P★ + b★ h / 2
    workspace.flat_values .= P★ .+ b★ .* h ./ 2
    level_mean!(state.reference_potential, workspace.flat_values, workspace.level_sum, grid)

    launch!(arch, grid, (Nz + 1,), _compute_reference_buoyancy!, state.reference_buoyancy, grid, state.reference_potential,
            b★)

    return nothing
end

#####
##### The tendency of the reference state
#####

# Ṗ★(z★) = h (Ċ[n₁ - 1] + Ċ[n₂]) / 2, at the mid-height of the band of slots n₁…n₂ of the cell
@kernel function _compute_parcel_reference_tendency!(parcel_tendency, band_first, band_last, cumulative_tendency,
                                                     slot_thickness)
    i, j, k = @index(Global, NTuple)
    @inbounds parcel_tendency[i, j, k] = band_reference_tendency(one(slot_thickness) / 2, band_first[i, j, k],
                                                                 band_last[i, j, k], cumulative_tendency, slot_thickness)
end

function compute_reference_tendency!(parcel_tendency, tendency, ::Sorting)
    grid = parcel_tendency.grid
    arch = architecture(grid)
    N = prod(size(grid))
    workspace = tendency.reference_height.operand.workspace
    h = workspace.slot_thickness
    Ḃ★ = tendency.workspace.sorted_tendency
    Ċ = tendency.workspace.cumulative_tendency

    # Ḃ★: the buoyancy tendencies in the order of b★, gathered through the permutation of the reference state; the
    # order within a run of tied buoyancies does not matter
    flatten_interior!(workspace.flat_values, tendency.buoyancy_tendency, grid)
    Ḃ★ .= view(workspace.flat_values, workspace.permutation)
    cumsum!(Ċ, Ḃ★)

    launch!(arch, grid, :xyz, _compute_parcel_reference_tendency!, parcel_tendency.data, workspace.band_first,
            workspace.band_last, Ċ, h)

    # Ṗ★ᶜᶜᶜ: the mean over each level's slots of Ṗ★ at the slot centres, with the bands of the slots gathered through
    # the permutation
    slot_band_first = view(vec(workspace.band_first), workspace.permutation)
    slot_band_last = view(vec(workspace.band_last), workspace.permutation)
    workspace.flat_values .= slot_reference_tendency.(1:N, slot_band_first, slot_band_last, Ref(Ċ), h)
    level_mean!(tendency.reference_tendency, workspace.flat_values, workspace.level_sum, grid)

    return nothing
end
