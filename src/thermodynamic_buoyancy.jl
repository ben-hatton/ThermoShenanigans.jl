"""
    ThermodynamicBuoyancy

Buoyancy formulation with b(θ, S) from `thermodynamics`. Fields:

* `thermodynamics`: an `AbstractBoussinesqThermodynamics`
* `constant_salinity`: the value at which S is held, or `nothing` if S is a tracer
"""
struct ThermodynamicBuoyancy{TH, S} <: AbstractBuoyancyFormulation{TH}
    thermodynamics :: TH
    constant_salinity :: S
end

"""
    ThermodynamicBuoyancy(grid, thermodynamics; constant_salinity = nothing)

Return a buoyancy formulation with b(θ, S) from `thermodynamics`, reading tracers `θ` and `S`.
z-dependent variables (π, m, T, μ) are evaluated at the height z of each cell, relative to the
`reference_pressure_height` z₀ of `thermodynamics`, where the pressure is P₀.
If `constant_salinity` is a number, only `θ` is required and S is held at that value.

```jldoctest
julia> using Oceananigans, ThermoShenanigans

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), z = (-1, 0), topology = (Periodic, Flat, Bounded));

julia> thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                          reference_temperature = 283, reference_salinity = 0.035,
                                                          reference_heat_capacity = 4000);

julia> ThermodynamicBuoyancy(grid, thermodynamics)
ThermodynamicBuoyancy:
└── thermodynamics: CabbelingBoussinesqThermodynamics{Float64}
```
"""
function ThermodynamicBuoyancy(grid, thermodynamics::AbstractBoussinesqThermodynamics; constant_salinity = nothing)
    FT = eltype(grid)
    constant_salinity = isnothing(constant_salinity) ? nothing : convert(FT, constant_salinity)
    return ThermodynamicBuoyancy(thermodynamics, constant_salinity)
end

# Buoyancy with S held constant, for which only θ is a tracer
const ConstantSalinityThermodynamicBuoyancy = ThermodynamicBuoyancy{<:Any, <:Number}

Base.summary(buoyancy_formulation::ThermodynamicBuoyancy) =
    string("ThermodynamicBuoyancy with ", summary(buoyancy_formulation.thermodynamics))

function Base.show(io::IO, buoyancy_formulation::ThermodynamicBuoyancy)
    print(io, "ThermodynamicBuoyancy:", "\n")

    if !isnothing(buoyancy_formulation.constant_salinity)
        print(io, "├── constant_salinity: ", prettysummary(buoyancy_formulation.constant_salinity), "\n")
    end

    print(io, "└── thermodynamics: ", summary(buoyancy_formulation.thermodynamics))
end

required_tracers(::ThermodynamicBuoyancy) = (:θ, :S)
required_tracers(::ConstantSalinityThermodynamicBuoyancy) = (:θ,)

# Return (θ, S) from the model fields, with S the constant value if there is one
@inline get_temperature_and_salinity(::ThermodynamicBuoyancy, fields) = fields.θ, fields.S
@inline get_temperature_and_salinity(buoyancy_formulation::ConstantSalinityThermodynamicBuoyancy, fields) =
    fields.θ, buoyancy_formulation.constant_salinity

#####
##### Thermodynamic functions at (Center, Center, Center), from the centred θ, S and z
#####

# Evaluate f(θ, S, z, thermodynamics) at the height z of the cell centre, e.g. f = exner, chemical_potential or temperature
@inline thermodynamic_functionᶜᶜᶜ(i, j, k, grid, f::F, buoyancy_formulation, fields) where F =
    f(θ_and_sᴬ(i, j, k, get_temperature_and_salinity(buoyancy_formulation, fields)...)...,
      znode(i, j, k, grid, Center(), Center(), Center()), buoyancy_formulation.thermodynamics)

#####
##### Buoyancy force
#####

# b at (Center, Center, Center); the w equation uses the energy-consistent b̃ below rather than ℑz b
@inline buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy_formulation::ThermodynamicBuoyancy, fields) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, buoyancy, buoyancy_formulation, fields)

@inline ∂x_b(i, j, k, grid, buoyancy_formulation::ThermodynamicBuoyancy, fields) =
    ∂xᶠᶜᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy_formulation, fields)

@inline ∂y_b(i, j, k, grid, buoyancy_formulation::ThermodynamicBuoyancy, fields) =
    ∂yᶜᶠᶜ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy_formulation, fields)

@inline ∂z_b(i, j, k, grid, buoyancy_formulation::ThermodynamicBuoyancy, fields) =
    ∂zᶜᶜᶠ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy_formulation, fields)

#=
Energy-consistent buoyancy force at w-points: the discrete-gradient buoyancy
    b̃ = [δz(b z') - ℑz(b_θ z') δzθ - ℑz(b_S z') δzS] / Δz = δz(b z') / Δz - ℑz(b_θ z') ∂zθ - ℑz(b_S z') ∂zS
at (Center, Center, Face), with z' = z - z₀. ∂zS is zero for constant salinity, by Oceananigans' `∂zᶜᶜᶠ` of a
`Number` (Operators/derivative_operators.jl). With centred second-order tracer advection, the buoyancy work Σ w b̃ V
then exactly balances the advective change of Σ 𝒫(θ, S, z) V whenever h⁰ and b are at most quadratic in (θ, S), when
z₀ drops out of b̃. For a linear b, b̃ = ℑz b.
=#
@inline buoyancy_height(θ, S, z, thermodynamics)   = buoyancy(θ, S, z, thermodynamics) * height_above_z₀(z, thermodynamics)
@inline buoyancy_θ_height(θ, S, z, thermodynamics) = ∂b∂θ(θ, S, z, thermodynamics) * height_above_z₀(z, thermodynamics)
@inline buoyancy_S_height(θ, S, z, thermodynamics) = ∂b∂S(θ, S, z, thermodynamics) * height_above_z₀(z, thermodynamics)

@inline function consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_formulation, fields)
    θ, S = get_temperature_and_salinity(buoyancy_formulation, fields)
    return δzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_height, buoyancy_formulation, fields) / Δzᶜᶜᶠ(i, j, k, grid) -
           ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_θ_height, buoyancy_formulation, fields) * ∂zᶜᶜᶠ(i, j, k, grid, θ) -
           ℑzᵃᵃᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_S_height, buoyancy_formulation, fields) * ∂zᶜᶜᶠ(i, j, k, grid, S)
end

# Oceananigans uses z_dot_g_bᶜᶜᶠ for both the w equation and the hydrostatic pressure anomaly (gravity along -z only)
@inline z_dot_g_bᶜᶜᶠ(i, j, k, grid, buoyancy_force::BuoyancyForce{<:ThermodynamicBuoyancy, NegativeZDirection}, fields) =
    consistent_buoyancyᶜᶜᶠ(i, j, k, grid, buoyancy_force.formulation, fields)
