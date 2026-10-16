"""
    AbstractBoussinesqThermodynamics

Supertype for Boussinesq thermodynamics, defined by b(θ, S) (including g), h⁰(θ, S) and η⁰(θ, S),
where θ is a generic entropic variable. Unlike depth-dependent equations of state (e.g. TEOS-10 in Oceananigans),
b is independent of z: z enters only through Σ = h⁰ - b z.

Subtypes implement, each as `f(θ, S, thermodynamics)`,

* the buoyancy `buoyancy` and its derivatives `∂b∂θ`, `∂b∂S`, `∂²b∂θ²`, `∂²b∂θ∂S`, `∂²b∂S²`,
* the potential enthalpy `potential_enthalpy` and its derivatives `∂h⁰∂θ`, `∂h⁰∂S`, `∂²h⁰∂θ²`, `∂²h⁰∂θ∂S`, `∂²h⁰∂S²`,
* the entropy `entropy` and its derivatives `∂η⁰∂θ`, `∂η⁰∂S`, `∂²η⁰∂θ²`, `∂²η⁰∂θ∂S`, `∂²η⁰∂S²`.
"""
abstract type AbstractBoussinesqThermodynamics end

# Linear (γ = 0) and cabbeling (γ ≠ 0) thermodynamics

"""
    CabbelingBoussinesqThermodynamics

Thermodynamics with buoyancy quadratic in θ. Fields:

* `gravitational_acceleration`: g
* `thermal_expansion`: α
* `haline_contraction`: β
* `cabbeling_coefficient`: γ
* `reference_temperature`: θᵣ
* `reference_salinity`: Sᵣ
* `reference_heat_capacity`: cₚ⁰
* `saline_entropy_constant`: R_S
"""
struct CabbelingBoussinesqThermodynamics{FT} <: AbstractBoussinesqThermodynamics
    gravitational_acceleration :: FT
    thermal_expansion :: FT
    haline_contraction :: FT
    cabbeling_coefficient :: FT
    reference_temperature :: FT
    reference_salinity :: FT
    reference_heat_capacity :: FT
    saline_entropy_constant :: FT
end

"""
    CabbelingBoussinesqThermodynamics([FT;] thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                                      reference_temperature, reference_salinity, reference_heat_capacity,
                                      saline_entropy_constant = R / M_S, gravitational_acceleration = g)

Return thermodynamics with θ the absolute potential temperature and

    b  = g [α (θ - θᵣ) + ½ γ (θ - θᵣ)² - β (S - Sᵣ)]
    h⁰ = cₚ⁰ θ
    η⁰ = cₚ⁰ ln(θ / θᵣ) + η_sal(S),   η_sal = - R_S S (ln S - 1)

γ = 0 gives a linear equation of state.
The default R_S = R / M_S ≈ 264.76 J kg⁻¹ K⁻¹ is the ideal entropy of mixing of standard seawater,
with M_S = 31.4038218 g mol⁻¹ (Millero et al. 2008, eq. 5.3) and S in kg/kg.
Assumes a specific heat capacity independent of temperature and salinity, and that the fluid is an ideal solution.

```jldoctest
julia> using ThermoShenanigans

julia> CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
                                         reference_salinity = 0.035, reference_heat_capacity = 4000)
CabbelingBoussinesqThermodynamics{Float64}(g=9.80665, α=0.0002, β=0.8, γ=0.0, θᵣ=283.0, Sᵣ=0.035, cₚ⁰=4000.0, R_S=264.76)
```
"""
function CabbelingBoussinesqThermodynamics(FT = Oceananigans.defaults.FloatType;
                                           thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                                           reference_temperature, reference_salinity, reference_heat_capacity,
                                           saline_entropy_constant = 8.314462618 / 31.4038218e-3,
                                           gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration)

    return CabbelingBoussinesqThermodynamics{FT}(convert(FT, gravitational_acceleration),
                                                 convert(FT, thermal_expansion),
                                                 convert(FT, haline_contraction),
                                                 convert(FT, cabbeling_coefficient),
                                                 convert(FT, reference_temperature),
                                                 convert(FT, reference_salinity),
                                                 convert(FT, reference_heat_capacity),
                                                 convert(FT, saline_entropy_constant))
end

# Interface: b, h⁰ and η⁰ with their first and second derivatives

"""
    buoyancy(θ, S, thermodynamics)

Buoyancy b(θ, S), including the gravitational acceleration.
"""
@inline buoyancy(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * (thermodynamics.thermal_expansion * (θ - thermodynamics.reference_temperature) +
                                                 thermodynamics.cabbeling_coefficient * (θ - thermodynamics.reference_temperature)^2 / 2 -
                                                 thermodynamics.haline_contraction * (S - thermodynamics.reference_salinity))

"""
    ∂b∂θ(θ, S, thermodynamics)

Derivative b_θ of the buoyancy with respect to θ.
"""
@inline ∂b∂θ(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * (thermodynamics.thermal_expansion +
                                                 thermodynamics.cabbeling_coefficient * (θ - thermodynamics.reference_temperature))

"""
    ∂b∂S(θ, S, thermodynamics)

Derivative b_S of the buoyancy with respect to S.
"""
@inline ∂b∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.gravitational_acceleration * thermodynamics.haline_contraction

"""
    ∂²b∂θ²(θ, S, thermodynamics)

Second derivative b_θθ of the buoyancy.
"""
@inline ∂²b∂θ²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.gravitational_acceleration * thermodynamics.cabbeling_coefficient

"""
    ∂²b∂θ∂S(θ, S, thermodynamics)

Mixed second derivative b_θS of the buoyancy.
"""
@inline ∂²b∂θ∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    ∂²b∂S²(θ, S, thermodynamics)

Second derivative b_SS of the buoyancy.
"""
@inline ∂²b∂S²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    potential_enthalpy(θ, S, thermodynamics)

Potential enthalpy h⁰(θ, S), the enthalpy at the reference pressure.
"""
@inline potential_enthalpy(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.reference_heat_capacity * θ

"""
    ∂h⁰∂θ(θ, S, thermodynamics)

Derivative h⁰_θ of the potential enthalpy with respect to θ.
"""
@inline ∂h⁰∂θ(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = thermodynamics.reference_heat_capacity

"""
    ∂h⁰∂S(θ, S, thermodynamics)

Derivative h⁰_S of the potential enthalpy with respect to S.
"""
@inline ∂h⁰∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    ∂²h⁰∂θ²(θ, S, thermodynamics)

Second derivative h⁰_θθ of the potential enthalpy.
"""
@inline ∂²h⁰∂θ²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    ∂²h⁰∂θ∂S(θ, S, thermodynamics)

Mixed second derivative h⁰_θS of the potential enthalpy.
"""
@inline ∂²h⁰∂θ∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    ∂²h⁰∂S²(θ, S, thermodynamics)

Second derivative h⁰_SS of the potential enthalpy.
"""
@inline ∂²h⁰∂S²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    entropy(θ, S, thermodynamics)

Entropy η⁰(θ, S) at the reference pressure.
"""
@inline entropy(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    thermodynamics.reference_heat_capacity * log(θ / thermodynamics.reference_temperature) -
    thermodynamics.saline_entropy_constant * S * (log(S) - 1)

"""
    ∂η⁰∂θ(θ, S, thermodynamics)

Derivative η⁰_θ of the entropy with respect to θ.
"""
@inline ∂η⁰∂θ(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = thermodynamics.reference_heat_capacity / θ

"""
    ∂η⁰∂S(θ, S, thermodynamics)

Derivative η⁰_S of the entropy with respect to S.
"""
@inline ∂η⁰∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.saline_entropy_constant * log(S)

"""
    ∂²η⁰∂θ²(θ, S, thermodynamics)

Second derivative η⁰_θθ of the entropy.
"""
@inline ∂²η⁰∂θ²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.reference_heat_capacity / θ^2

"""
    ∂²η⁰∂θ∂S(θ, S, thermodynamics)

Mixed second derivative η⁰_θS of the entropy.
"""
@inline ∂²η⁰∂θ∂S(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) = zero(θ)

"""
    ∂²η⁰∂S²(θ, S, thermodynamics)

Second derivative η⁰_SS of the entropy.
"""
@inline ∂²η⁰∂S²(θ, S, thermodynamics::CabbelingBoussinesqThermodynamics) =
    - thermodynamics.saline_entropy_constant / S

#=
Derived from Σ = h⁰ - b z and η⁰, with z the height above the reference level. Since
dΣ = π dθ + Σ_S dS = T dη + μ dS at fixed z, T and μ are the temperature and chemical potential, and
Σ_S = (∂Σ/∂S)_θ is the chemical potential analogue of de Szoeke (2000, J. Phys. Oceanogr.).
=#

"""
    static_energy(θ, S, z, thermodynamics)

Static energy Σ = h⁰ - b z, with z the height above the reference level.
"""
@inline static_energy(θ, S, z, thermodynamics) =
    potential_enthalpy(θ, S, thermodynamics) - buoyancy(θ, S, thermodynamics) * z

"""
    exner(θ, S, z, thermodynamics)

Exner function π = Σ_θ = h⁰_θ - b_θ z.
"""
@inline exner(θ, S, z, thermodynamics) = ∂h⁰∂θ(θ, S, thermodynamics) - ∂b∂θ(θ, S, thermodynamics) * z

"""
    chemical_potential_analogue(θ, S, z, thermodynamics)

Chemical potential analogue Σ_S = (∂Σ/∂S)_θ = h⁰_S - b_S z.
"""
@inline chemical_potential_analogue(θ, S, z, thermodynamics) =
    ∂h⁰∂S(θ, S, thermodynamics) - ∂b∂S(θ, S, thermodynamics) * z

"""
    temperature(θ, S, z, thermodynamics)

Temperature T = π / η⁰_θ.
"""
@inline temperature(θ, S, z, thermodynamics) = exner(θ, S, z, thermodynamics) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
    chemical_potential(θ, S, z, thermodynamics)

Chemical potential μ = Σ_S - T η⁰_S.
"""
@inline chemical_potential(θ, S, z, thermodynamics) =
    chemical_potential_analogue(θ, S, z, thermodynamics) - temperature(θ, S, z, thermodynamics) * ∂η⁰∂S(θ, S, thermodynamics)

"""
    ∂T∂θ(θ, S, z, thermodynamics)

Derivative T_θ = (Σ_θθ - T η⁰_θθ) / η⁰_θ of the temperature with respect to θ.
"""
@inline ∂T∂θ(θ, S, z, thermodynamics) =
    (∂²h⁰∂θ²(θ, S, thermodynamics) - ∂²b∂θ²(θ, S, thermodynamics) * z -
     temperature(θ, S, z, thermodynamics) * ∂²η⁰∂θ²(θ, S, thermodynamics)) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
    ∂T∂S(θ, S, z, thermodynamics)

Derivative T_S = (Σ_θS - T η⁰_θS) / η⁰_θ of the temperature with respect to S.
"""
@inline ∂T∂S(θ, S, z, thermodynamics) =
    (∂²h⁰∂θ∂S(θ, S, thermodynamics) - ∂²b∂θ∂S(θ, S, thermodynamics) * z -
     temperature(θ, S, z, thermodynamics) * ∂²η⁰∂θ∂S(θ, S, thermodynamics)) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
    heat_capacity(θ, S, z, thermodynamics)

Specific heat capacity cₚ = T (∂η/∂T)_S,z = T η⁰_θ / T_θ.
"""
@inline heat_capacity(θ, S, z, thermodynamics) =
    temperature(θ, S, z, thermodynamics) * ∂η⁰∂θ(θ, S, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)

#=
Isothermal derivatives are (∂f/∂S)_T,z = f_S - f_θ T_S / T_θ.
(∂μ/∂z)_T,S = - (∂b/∂S)_T,z is a Maxwell relation.
=#

"""
    isothermal_∂b∂S(θ, S, z, thermodynamics)

Isothermal derivative (∂b/∂S)_T,z = b_S - b_θ T_S / T_θ of the buoyancy.
"""
@inline isothermal_∂b∂S(θ, S, z, thermodynamics) =
    ∂b∂S(θ, S, thermodynamics) - ∂b∂θ(θ, S, thermodynamics) * ∂T∂S(θ, S, z, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)

"""
    isothermal_∂μ∂S(θ, S, z, thermodynamics)

Isothermal derivative (∂μ/∂S)_T,z = Σ_SS - T η⁰_SS - η⁰_θ T_S² / T_θ of the chemical potential.
"""
@inline isothermal_∂μ∂S(θ, S, z, thermodynamics) =
    ∂²h⁰∂S²(θ, S, thermodynamics) - ∂²b∂S²(θ, S, thermodynamics) * z -
    temperature(θ, S, z, thermodynamics) * ∂²η⁰∂S²(θ, S, thermodynamics) -
    ∂η⁰∂θ(θ, S, thermodynamics) * ∂T∂S(θ, S, z, thermodynamics)^2 / ∂T∂θ(θ, S, z, thermodynamics)
