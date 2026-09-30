struct ThermodynamicBuoyancy{TH, FT, S} <: AbstractBuoyancyFormulation{TH}
    thermodynamics :: TH
    surface_height :: FT
    constant_salinity :: S
end

"""
    ThermodynamicBuoyancy(grid, thermodynamics; surface_height = top of grid, constant_salinity = nothing)

Return a buoyancy formulation with b(θ, S) from `thermodynamics`, reading tracers `θ` and `S`.
z-dependent variables (π, μ, T) use the height above `surface_height`, where the pressure is P₀.
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

# f(θ, S, thermo), e.g. b or η⁰_S
@inline thermodynamic_functionᶜᶜᶜ(i, j, k, grid, f::F, tb, C) where F =
    f(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(tb, C)...)..., tb.thermodynamics)

# f(θ, S, z, thermo), e.g. π, μ or T
@inline height_dependent_functionᶜᶜᶜ(i, j, k, grid, f::F, tb, C) where F =
    f(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(tb, C)...)..., height_above_surfaceᶜᶜᶜ(i, j, k, grid, tb), tb.thermodynamics)

# b at (Center, Center, Center); Oceananigans interpolates it to w-points in `z_dot_g_bᶜᶜᶠ`
@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = thermodynamic_functionᶜᶜᶜ(i, j, k, grid, buoyancy, tb, C)

@inline ∂x_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂xᶠᶜᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)
@inline ∂y_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂yᶜᶠᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)
@inline ∂z_b(i, j, k, grid, tb::ThermodynamicBuoyancy, C) = ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, tb, C)
