struct ThermodynamicBuoyancy{TH, FT, S} <: AbstractBuoyancyFormulation{TH}
    thermodynamics :: TH
    surface_height :: FT
    constant_salinity :: S
end

"""
    ThermodynamicBuoyancy(grid, thermodynamics; surface_height = top of grid, constant_salinity = nothing)

Return a buoyancy formulation with b(θ, S) from `thermodynamics`, reading tracers `θ` and `S`.
z-dependent variables (π, T, μ) use the height above `surface_height`, where the pressure is P₀.
If `constant_salinity` is a number, only `θ` is required and S is held at that value.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size=(4, 4), x=(0, 1), z=(-1, 0), topology=(Periodic, Flat, Bounded));

julia> thermo = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
                                                  reference_salinity = 0.035, reference_heat_capacity = 4000);

julia> ThermodynamicBuoyancy(grid, thermo)
ThermodynamicBuoyancy with CabbelingBoussinesqThermodynamics{Float64} and surface_height=0.0
```
"""
function ThermodynamicBuoyancy(grid, thermodynamics::AbstractBoussinesqThermodynamics;
                               surface_height = znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face()),
                               constant_salinity = nothing)

    FT = eltype(grid)
    constant_salinity = isnothing(constant_salinity) ? nothing : convert(FT, constant_salinity)

    return ThermodynamicBuoyancy(thermodynamics, convert(FT, surface_height), constant_salinity)
end

const ConstantSalinityThermodynamicBuoyancy = ThermodynamicBuoyancy{<:Any, <:Any, <:Number}

required_tracers(::ThermodynamicBuoyancy) = (:θ, :S)
required_tracers(::ConstantSalinityThermodynamicBuoyancy) = (:θ,)

@inline get_temperature_and_salinity(::ThermodynamicBuoyancy, C) = C.θ, C.S
@inline get_temperature_and_salinity(tb::ConstantSalinityThermodynamicBuoyancy, C) = C.θ, tb.constant_salinity

#####
##### Thermodynamic functions at (Center, Center, Center), from the centred θ, S and z
#####

@inline height_above_surfaceᶜᶜᶜ(i, j, k, grid, tb) = znode(i, j, k, grid, Center(), Center(), Center()) - tb.surface_height

# f(θ, S, z, thermo), e.g. π, μ or T
@inline thermodynamic_functionᶜᶜᶜ(i, j, k, grid, f::F, tb, C) where F =
    f(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(tb, C)...)..., height_above_surfaceᶜᶜᶜ(i, j, k, grid, tb), tb.thermodynamics)

# z-independent functions in the same form
@inline θ_value(θ, S, z, t)    = θ
@inline S_value(θ, S, z, t)    = S
@inline buoyancy_θ(θ, S, z, t) = ∂b∂θ(θ, S, t)
@inline buoyancy_S(θ, S, z, t) = ∂b∂S(θ, S, t)

# b at (Center, Center, Center); the w equation uses the energy-consistent b̃ below rather than ℑz b
@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, tb::ThermodynamicBuoyancy, C) =
    buoyancy(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(tb, C)...)..., tb.thermodynamics)

@inline ∂x_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂xᶠᶜᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)
@inline ∂y_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂yᶜᶠᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)
@inline ∂z_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)

#####
##### Energy-consistent buoyancy force at w-points
#####

# Discrete-gradient buoyancy b̃ = [δz(b z) - ℑz(b_θ z) δzθ - ℑz(b_S z) δzS] / Δz at (Center, Center, Face).
# With centred second-order tracer advection, the buoyancy work Σ w b̃ V then exactly balances the advective change of
# Σ Σ(θ, S, z) V whenever h⁰ and b are at most quadratic in (θ, S). For a linear b, b̃ = ℑz b.
@inline buoyancy_height(θ, S, z, t)   = buoyancy(θ, S, t) * z
@inline buoyancy_θ_height(θ, S, z, t) = ∂b∂θ(θ, S, t) * z
@inline buoyancy_S_height(θ, S, z, t) = ∂b∂S(θ, S, t) * z

@inline consistent_buoyancyᶜᶜᶠ(i, j, k, grid, tb, C) =
    (δzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_height, tb, C) -
     ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_θ_height, tb, C) * δzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, θ_value, tb, C) -
     ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_S_height, tb, C) * δzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, S_value, tb, C)) /
    Δzᶜᶜᶠ(i, j, k, grid)

# Oceananigans uses z_dot_g_bᶜᶜᶠ for both the w equation and the hydrostatic pressure anomaly (gravity along -z only)
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, bf::BuoyancyForce{<:ThermodynamicBuoyancy, NegativeZDirection}, C) =
    consistent_buoyancyᶜᶜᶠ(i, j, k, grid, bf.formulation, C)
