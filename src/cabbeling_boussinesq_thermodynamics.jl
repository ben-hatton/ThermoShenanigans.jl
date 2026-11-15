#####
##### Linear (γ = 0) and cabbeling (γ ≠ 0) thermodynamics
#####

const universal_gas_constant = 8.314462618  # R, J mol⁻¹ K⁻¹
const sea_salt_molar_mass    = 31.4038218e-3 # M_S of standard sea salt, kg mol⁻¹ (Millero et al. 2008, eq. 5.3)

"""
    CabbelingBoussinesqThermodynamics

Thermodynamics with buoyancy quadratic in θ. Fields:

* `gravitational_acceleration`: g
* `reference_pressure_height`: z₀, the height at which the pressure is the reference pressure P₀
* `thermal_expansion`: α
* `haline_contraction`: β
* `cabbeling_coefficient`: γ
* `reference_temperature`: T₀
* `reference_salinity`: S₀
* `reference_heat_capacity`: cₚ₀
* `saline_entropy_constant`: R_S
"""
struct CabbelingBoussinesqThermodynamics{FT} <: AbstractBoussinesqThermodynamics
    gravitational_acceleration :: FT
    reference_pressure_height :: FT
    thermal_expansion :: FT
    haline_contraction :: FT
    cabbeling_coefficient :: FT
    reference_temperature :: FT
    reference_salinity :: FT
    reference_heat_capacity :: FT
    saline_entropy_constant :: FT
end

"""
    CabbelingBoussinesqThermodynamics(FT = Oceananigans.defaults.FloatType;
                                      thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                                      reference_temperature, reference_salinity, reference_heat_capacity,
                                      saline_entropy_constant = R / M_S,
                                      gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                                      reference_pressure_height = 0)

Return thermodynamics with θ the absolute potential temperature and

    b  = g [α (θ - T₀) + ½ γ (θ - T₀)² - β (S - S₀)]
    h⁰ = cₚ₀ (θ - T₀)
    η⁰ = cₚ₀ ln(θ / T₀) - R_S [S ln(S / S₀) - (S - S₀)]

following Tailleux, Dubos & Hatton (arXiv:2512.18461), with the gauge h⁰ = η⁰ = 0 at (T₀, S₀).
The static energy is Σ = h⁰ - b (z - z₀) + p + g z₀, where z₀ = `reference_pressure_height` is the height at which the
pressure is the reference pressure P₀.

γ = 0 gives a linear equation of state.
The default `R_S = R / M_S` ≈ 264.76 J kg⁻¹ K⁻¹ is the ideal entropy of mixing of standard seawater,
with `M_S` = 31.4038218 g mol⁻¹ (Millero et al. 2008, eq. 5.3) and S in kg/kg.
Assumes a specific heat capacity independent of temperature and salinity, and that the fluid is an ideal solution.

```jldoctest
julia> using ThermoShenanigans

julia> CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
                                         reference_salinity = 0.035, reference_heat_capacity = 4000)
CabbelingBoussinesqThermodynamics{Float64}:
├── gravitational_acceleration: 9.80665
├── reference_pressure_height: 0.0
├── thermal_expansion: 0.0002
├── haline_contraction: 0.8
├── cabbeling_coefficient: 0.0
├── reference_temperature: 283.0
├── reference_salinity: 0.035
├── reference_heat_capacity: 4000.0
└── saline_entropy_constant: 264.76
```
"""
function CabbelingBoussinesqThermodynamics(FT = Oceananigans.defaults.FloatType;
                                           thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                                           reference_temperature, reference_salinity, reference_heat_capacity,
                                           saline_entropy_constant = universal_gas_constant / sea_salt_molar_mass,
                                           gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
                                           reference_pressure_height = 0)

    return CabbelingBoussinesqThermodynamics{FT}(convert(FT, gravitational_acceleration),
                                                 convert(FT, reference_pressure_height),
                                                 convert(FT, thermal_expansion),
                                                 convert(FT, haline_contraction),
                                                 convert(FT, cabbeling_coefficient),
                                                 convert(FT, reference_temperature),
                                                 convert(FT, reference_salinity),
                                                 convert(FT, reference_heat_capacity),
                                                 convert(FT, saline_entropy_constant))
end

Base.summary(::CabbelingBoussinesqThermodynamics{FT}) where FT = "CabbelingBoussinesqThermodynamics{$FT}"

function Base.show(io::IO, thermodynamics::CabbelingBoussinesqThermodynamics)
    print(io, summary(thermodynamics), ":", "\n",
              "├── gravitational_acceleration: ", prettysummary(thermodynamics.gravitational_acceleration), "\n",
              "├── reference_pressure_height: ", prettysummary(thermodynamics.reference_pressure_height), "\n",
              "├── thermal_expansion: ", prettysummary(thermodynamics.thermal_expansion), "\n",
              "├── haline_contraction: ", prettysummary(thermodynamics.haline_contraction), "\n",
              "├── cabbeling_coefficient: ", prettysummary(thermodynamics.cabbeling_coefficient), "\n",
              "├── reference_temperature: ", prettysummary(thermodynamics.reference_temperature), "\n",
              "├── reference_salinity: ", prettysummary(thermodynamics.reference_salinity), "\n",
              "├── reference_heat_capacity: ", prettysummary(thermodynamics.reference_heat_capacity), "\n",
              "└── saline_entropy_constant: ", prettysummary(thermodynamics.saline_entropy_constant))
end

@inline gravitational_acceleration(thermodynamics::CabbelingBoussinesqThermodynamics) = thermodynamics.gravitational_acceleration
@inline reference_pressure_height(thermodynamics::CabbelingBoussinesqThermodynamics)  = thermodynamics.reference_pressure_height

#####
##### Buoyancy
#####

@inline buoyancy(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * (thermodynamics.thermal_expansion * (θ - thermodynamics.reference_temperature) +
                                                 thermodynamics.cabbeling_coefficient * (θ - thermodynamics.reference_temperature)^2 / 2 -
                                                 thermodynamics.haline_contraction * (S - thermodynamics.reference_salinity))

@inline ∂b∂θ(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * (thermodynamics.thermal_expansion +
                                                 thermodynamics.cabbeling_coefficient * (θ - thermodynamics.reference_temperature))

@inline ∂b∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.gravitational_acceleration * thermodynamics.haline_contraction

@inline ∂²b∂θ²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * thermodynamics.cabbeling_coefficient

@inline ∂²b∂θ∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)
@inline ∂²b∂S²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)  = zero(θ)

#####
##### Potential enthalpy
#####

@inline potential_enthalpy(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.reference_heat_capacity * (θ - thermodynamics.reference_temperature)

@inline ∂h⁰∂θ(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)    = thermodynamics.reference_heat_capacity
@inline ∂h⁰∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)    = zero(θ)
@inline ∂²h⁰∂θ²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)  = zero(θ)
@inline ∂²h⁰∂θ∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)
@inline ∂²h⁰∂S²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)  = zero(θ)

#####
##### Entropy
#####

@inline entropy(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.reference_heat_capacity * log(θ / thermodynamics.reference_temperature) -
    thermodynamics.saline_entropy_constant * (S * log(S / thermodynamics.reference_salinity) - (S - thermodynamics.reference_salinity))

@inline ∂η⁰∂θ(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) = thermodynamics.reference_heat_capacity / θ

@inline ∂η⁰∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.saline_entropy_constant * log(S / thermodynamics.reference_salinity)

@inline ∂²η⁰∂θ²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)  = - thermodynamics.reference_heat_capacity / θ^2
@inline ∂²η⁰∂θ∂S(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)
@inline ∂²η⁰∂S²(θ, S, z, thermodynamics::CabbelingBoussinesqThermodynamics)  = - thermodynamics.saline_entropy_constant / S
