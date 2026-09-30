"""
    AbstractBoussinesqThermodynamics

Supertype for Boussinesq thermodynamics, defined by b(θ, S) (including g), h⁰(θ, S) and η⁰(θ, S),
where θ is a generic entropic variable. Unlike depth-dependent equations of state (e.g. TEOS-10 in Oceananigans),
b is independent of z: z enters only through Σ = h⁰ - b z. Subtypes implement the functions in `thermodynamic_interface`.
"""
abstract type AbstractBoussinesqThermodynamics end

const thermodynamic_interface = (:buoyancy,           :∂b∂θ,  :∂b∂S,  :∂²b∂θ²,  :∂²b∂θ∂S,  :∂²b∂S²,
                                 :potential_enthalpy, :∂h⁰∂θ, :∂h⁰∂S, :∂²h⁰∂θ², :∂²h⁰∂θ∂S, :∂²h⁰∂S²,
                                 :entropy,            :∂η⁰∂θ, :∂η⁰∂S, :∂²η⁰∂θ², :∂²η⁰∂θ∂S, :∂²η⁰∂S²)

#####
##### Linear (γ = 0) and cabbeling (γ ≠ 0) thermodynamics
#####

struct CabbelingBoussinesqThermodynamics{FT} <: AbstractBoussinesqThermodynamics
    gravitational_acceleration :: FT # g
    thermal_expansion :: FT          # α
    haline_contraction :: FT         # β
    cabbeling_coefficient :: FT      # γ
    reference_temperature :: FT      # θᵣ
    reference_salinity :: FT         # Sᵣ
    reference_heat_capacity :: FT    # cₚ⁰
    saline_entropy_constant :: FT    # R_η
end

"""
    CabbelingBoussinesqThermodynamics([FT;] thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                            reference_temperature, reference_salinity, reference_heat_capacity,
                            saline_entropy_constant = R / M_S, gravitational_acceleration = g)

Return thermodynamics with θ the absolute potential temperature and

    b  = g [α (θ - θᵣ) + ½ γ (θ - θᵣ)² - β (S - Sᵣ)]
    h⁰ = cₚ⁰ θ
    η⁰ = cₚ⁰ ln(θ / θᵣ) + η_sal(S),   η_sal = - R_η S (ln S - 1)

γ = 0 gives the linear model of Tailleux, Dubos & Hatton (2025, §4.2).
The default R_η = R / M_S ≈ 264.76 J kg⁻¹ K⁻¹ is the ideal entropy of mixing of standard seawater,
with M_S = 31.4038218 g mol⁻¹ (Millero et al. 2008, eq. 5.3) and S in kg/kg (use R / (1000 M_S) for g/kg).
It neglects non-ideality (osmotic coefficient ≈ 0.9).

```jldoctest
julia> using ThermoShenanigans

julia> CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
                               reference_salinity = 0.035, reference_heat_capacity = 4000)
CabbelingBoussinesqThermodynamics{Float64}(g=9.80665, α=0.0002, β=0.8, γ=0.0, θᵣ=283.0, Sᵣ=0.035, cₚ⁰=4000.0, R_η=264.76)
```
"""
function CabbelingBoussinesqThermodynamics(FT = Oceananigans.defaults.FloatType;
                                 thermal_expansion, haline_contraction, cabbeling_coefficient = 0,
                                 reference_temperature, reference_salinity, reference_heat_capacity,
                                 saline_entropy_constant = 8.314462618 / 31.4038218e-3,
                                 gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration)

    parameters = (gravitational_acceleration, thermal_expansion, haline_contraction, cabbeling_coefficient,
                  reference_temperature, reference_salinity, reference_heat_capacity, saline_entropy_constant)

    return CabbelingBoussinesqThermodynamics{FT}(convert.(FT, parameters)...)
end

const CT = CabbelingBoussinesqThermodynamics

@inline buoyancy(θ, S, t::CT) = t.gravitational_acceleration * (t.thermal_expansion * (θ - t.reference_temperature)
                                                                + t.cabbeling_coefficient * (θ - t.reference_temperature)^2 / 2
                                                                - t.haline_contraction * (S - t.reference_salinity))

@inline ∂b∂θ(θ, S, t::CT)    = t.gravitational_acceleration * (t.thermal_expansion + t.cabbeling_coefficient * (θ - t.reference_temperature))
@inline ∂b∂S(θ, S, t::CT)    = - t.gravitational_acceleration * t.haline_contraction
@inline ∂²b∂θ²(θ, S, t::CT)  = t.gravitational_acceleration * t.cabbeling_coefficient
@inline ∂²b∂θ∂S(θ, S, t::CT) = zero(θ)
@inline ∂²b∂S²(θ, S, t::CT)  = zero(θ)

@inline potential_enthalpy(θ, S, t::CT) = t.reference_heat_capacity * θ
@inline ∂h⁰∂θ(θ, S, t::CT)              = t.reference_heat_capacity
@inline ∂h⁰∂S(θ, S, t::CT)              = zero(θ)
@inline ∂²h⁰∂θ²(θ, S, t::CT)            = zero(θ)
@inline ∂²h⁰∂θ∂S(θ, S, t::CT)           = zero(θ)
@inline ∂²h⁰∂S²(θ, S, t::CT)             = zero(θ)

@inline entropy(θ, S, t::CT)  = t.reference_heat_capacity * log(θ / t.reference_temperature) - t.saline_entropy_constant * S * (log(S) - 1)
@inline ∂η⁰∂θ(θ, S, t::CT)    = t.reference_heat_capacity / θ
@inline ∂η⁰∂S(θ, S, t::CT)    = - t.saline_entropy_constant * log(S)
@inline ∂²η⁰∂θ²(θ, S, t::CT)  = - t.reference_heat_capacity / θ^2
@inline ∂²η⁰∂θ∂S(θ, S, t::CT) = zero(θ)
@inline ∂²η⁰∂S²(θ, S, t::CT)  = - t.saline_entropy_constant / S

#####
##### Derived from Σ = h⁰ - b z and η⁰ (TDH25 §3.1), with z the height above the surface
#####

@inline static_energy(θ, S, z, t)       = potential_enthalpy(θ, S, t) - buoyancy(θ, S, t) * z # Σ
@inline exner(θ, S, z, t)               = ∂h⁰∂θ(θ, S, t) - ∂b∂θ(θ, S, t) * z                  # π = Σ_θ
@inline chemical_potential(θ, S, z, t)  = ∂h⁰∂S(θ, S, t) - ∂b∂S(θ, S, t) * z                  # μ = Σ_S
@inline in_situ_temperature(θ, S, z, t) = exner(θ, S, z, t) / ∂η⁰∂θ(θ, S, t)                  # T = π / η⁰_θ

# T_θ = (Σ_θθ - T η⁰_θθ) / η⁰_θ  and  T_S = (Σ_θS - T η⁰_θS) / η⁰_θ
@inline ∂T∂θ(θ, S, z, t) = (∂²h⁰∂θ²(θ, S, t)  - ∂²b∂θ²(θ, S, t)  * z - in_situ_temperature(θ, S, z, t) * ∂²η⁰∂θ²(θ, S, t))  / ∂η⁰∂θ(θ, S, t)
@inline ∂T∂S(θ, S, z, t) = (∂²h⁰∂θ∂S(θ, S, t) - ∂²b∂θ∂S(θ, S, t) * z - in_situ_temperature(θ, S, z, t) * ∂²η⁰∂θ∂S(θ, S, t)) / ∂η⁰∂θ(θ, S, t)

@inline heat_capacity(θ, S, z, t)   = in_situ_temperature(θ, S, z, t) * ∂η⁰∂θ(θ, S, t) / ∂T∂θ(θ, S, z, t) # cₚ = T (∂η/∂T)_S,z
@inline isothermal_∂θ∂S(θ, S, z, t) = - ∂T∂S(θ, S, z, t) / ∂T∂θ(θ, S, z, t)                                # (∂θ/∂S)_T,z

# With m = μ - T η⁰_S, whose z-derivative (∂m/∂z)_T = - (∂b/∂S)_T is a Maxwell relation
@inline isothermal_∂b∂S(θ, S, z, t) = ∂b∂S(θ, S, t) + ∂b∂θ(θ, S, t) * isothermal_∂θ∂S(θ, S, z, t)                           # (∂b/∂S)_T,z
@inline isothermal_∂m∂S(θ, S, z, t) = ∂²h⁰∂S²(θ, S, t) - ∂²b∂S²(θ, S, t) * z - in_situ_temperature(θ, S, z, t) * ∂²η⁰∂S²(θ, S, t) +
                                      ∂η⁰∂θ(θ, S, t) * ∂T∂S(θ, S, z, t) * isothermal_∂θ∂S(θ, S, z, t)                   # (∂m/∂S)_T,z
