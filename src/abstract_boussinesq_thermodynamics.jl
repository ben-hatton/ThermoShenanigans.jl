"""
    AbstractBoussinesqThermodynamics

Supertype for Boussinesq thermodynamics, defined by b(θ, S) (including g), h⁰(θ, S) and η⁰(θ, S),
where θ is a generic entropic variable. Unlike depth-dependent equations of state (e.g. TEOS-10 in Oceananigans),
b is independent of z: z enters only through Σ = h⁰ - b (z - z₀) + p + g₀ z₀, where z₀ is the height at which the
pressure is the reference pressure P₀.

Subtypes implement, each as `f(θ, S, thermodynamics)`,

* the buoyancy `buoyancy` and its derivatives `∂b∂θ`, `∂b∂S`, `∂²b∂θ²`, `∂²b∂θ∂S`, `∂²b∂S²`,
* the potential enthalpy `potential_enthalpy` and its derivatives `∂h⁰∂θ`, `∂h⁰∂S`, `∂²h⁰∂θ²`, `∂²h⁰∂θ∂S`, `∂²h⁰∂S²`,
* the entropy `entropy` and its derivatives `∂η⁰∂θ`, `∂η⁰∂S`, `∂²η⁰∂θ²`, `∂²η⁰∂θ∂S`, `∂²η⁰∂S²`,

and, each as `f(thermodynamics)`, the `reference_pressure_height` z₀ and the `gravitational_acceleration` g₀.
"""
abstract type AbstractBoussinesqThermodynamics end

#####
##### Interface: b, h⁰ and η⁰ with their first and second derivatives
#####

"""
    buoyancy(θ, S, thermodynamics)

Buoyancy b(θ, S), including the gravitational acceleration.
"""
function buoyancy end

"""
    ∂b∂θ(θ, S, thermodynamics)

Derivative b_θ of the buoyancy with respect to θ.
"""
function ∂b∂θ end

"""
    ∂b∂S(θ, S, thermodynamics)

Derivative b_S of the buoyancy with respect to S.
"""
function ∂b∂S end

"""
    ∂²b∂θ²(θ, S, thermodynamics)

Second derivative b_θθ of the buoyancy.
"""
function ∂²b∂θ² end

"""
    ∂²b∂θ∂S(θ, S, thermodynamics)

Mixed second derivative b_θS of the buoyancy.
"""
function ∂²b∂θ∂S end

"""
    ∂²b∂S²(θ, S, thermodynamics)

Second derivative b_SS of the buoyancy.
"""
function ∂²b∂S² end

"""
    potential_enthalpy(θ, S, thermodynamics)

Potential enthalpy h⁰(θ, S), the enthalpy at the reference pressure.
"""
function potential_enthalpy end

"""
    ∂h⁰∂θ(θ, S, thermodynamics)

Derivative h⁰_θ of the potential enthalpy with respect to θ.
"""
function ∂h⁰∂θ end

"""
    ∂h⁰∂S(θ, S, thermodynamics)

Derivative h⁰_S of the potential enthalpy with respect to S.
"""
function ∂h⁰∂S end

"""
    ∂²h⁰∂θ²(θ, S, thermodynamics)

Second derivative h⁰_θθ of the potential enthalpy.
"""
function ∂²h⁰∂θ² end

"""
    ∂²h⁰∂θ∂S(θ, S, thermodynamics)

Mixed second derivative h⁰_θS of the potential enthalpy.
"""
function ∂²h⁰∂θ∂S end

"""
    ∂²h⁰∂S²(θ, S, thermodynamics)

Second derivative h⁰_SS of the potential enthalpy.
"""
function ∂²h⁰∂S² end

"""
    entropy(θ, S, thermodynamics)

Entropy η⁰(θ, S) at the reference pressure.
"""
function entropy end

"""
    ∂η⁰∂θ(θ, S, thermodynamics)

Derivative η⁰_θ of the entropy with respect to θ.
"""
function ∂η⁰∂θ end

"""
    ∂η⁰∂S(θ, S, thermodynamics)

Derivative η⁰_S of the entropy with respect to S.
"""
function ∂η⁰∂S end

"""
    ∂²η⁰∂θ²(θ, S, thermodynamics)

Second derivative η⁰_θθ of the entropy.
"""
function ∂²η⁰∂θ² end

"""
    ∂²η⁰∂θ∂S(θ, S, thermodynamics)

Mixed second derivative η⁰_θS of the entropy.
"""
function ∂²η⁰∂θ∂S end

"""
    ∂²η⁰∂S²(θ, S, thermodynamics)

Second derivative η⁰_SS of the entropy.
"""
function ∂²η⁰∂S² end

"""
    reference_pressure_height(thermodynamics)

Height z₀ at which the pressure is the reference pressure P₀, so that the hydrostatic reference pressure is
P_R(z) = P₀ - g₀ (z - z₀) / υ₀.
"""
function reference_pressure_height end

"""
    gravitational_acceleration(thermodynamics)

Gravitational acceleration g₀, the same as that included in the buoyancy b.
"""
function gravitational_acceleration end

#####
##### Derived functions of θ, S and z
#####

#=
Derived from Σ = h⁰ - b (z - z₀) + p + g₀ z₀ at fixed p, and η⁰. The functions below take the height z as their
argument `z`, with z₀ the `reference_pressure_height` of the thermodynamics, where the pressure is P₀.
Since dΣ = π dθ + m dS = T dη + μ dS at fixed z, T and μ are the temperature and chemical potential (T_B and μ_B of
Tailleux, Dubos & Hatton), and m = (∂Σ/∂S)_θ is the chemical potential analogue of de Szoeke (2000, J. Phys. Oceanogr.).
=#

# z - z₀, the height above the height z₀ at which the pressure is P₀
@inline height_above_z₀(z, thermodynamics) = z - reference_pressure_height(thermodynamics)

"""
$(TYPEDSIGNATURES)

Potential energy 𝒫 = Σ - p ∂Σ/∂p = h⁰ - b (z - z₀) + g₀ z₀ at height z (Tailleux 2024), with z₀ the
`reference_pressure_height`.
"""
@inline potential_energy(θ, S, z, thermodynamics) =
    potential_enthalpy(θ, S, thermodynamics) - buoyancy(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics) +
    gravitational_acceleration(thermodynamics) * reference_pressure_height(thermodynamics)

"""
$(TYPEDSIGNATURES)

Exner function `π = Σ_θ = h⁰_θ - b_θ (z - z₀)`.
"""
@inline exner(θ, S, z, thermodynamics) =
    ∂h⁰∂θ(θ, S, thermodynamics) - ∂b∂θ(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics)

"""
$(TYPEDSIGNATURES)

Chemical potential analogue `m = (∂Σ/∂S)_θ = h⁰_S - b_S (z - z₀)`.
"""
@inline chemical_potential_analogue(θ, S, z, thermodynamics) =
    ∂h⁰∂S(θ, S, thermodynamics) - ∂b∂S(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics)

"""
$(TYPEDSIGNATURES)

Temperature T = π / η⁰_θ.
"""
@inline temperature(θ, S, z, thermodynamics) = exner(θ, S, z, thermodynamics) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
$(TYPEDSIGNATURES)

Chemical potential μ = m - T η⁰_S.
"""
@inline chemical_potential(θ, S, z, thermodynamics) =
    chemical_potential_analogue(θ, S, z, thermodynamics) - temperature(θ, S, z, thermodynamics) * ∂η⁰∂S(θ, S, thermodynamics)

"""
$(TYPEDSIGNATURES)

Derivative `T_θ = (Σ_θθ - T η⁰_θθ) / η⁰_θ` of the temperature with respect to θ.
"""
@inline ∂T∂θ(θ, S, z, thermodynamics) =
    (∂²h⁰∂θ²(θ, S, thermodynamics) - ∂²b∂θ²(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics) -
     temperature(θ, S, z, thermodynamics) * ∂²η⁰∂θ²(θ, S, thermodynamics)) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
$(TYPEDSIGNATURES)

Derivative `T_S = (Σ_θS - T η⁰_θS) / η⁰_θ` of the temperature with respect to S.
"""
@inline ∂T∂S(θ, S, z, thermodynamics) =
    (∂²h⁰∂θ∂S(θ, S, thermodynamics) - ∂²b∂θ∂S(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics) -
     temperature(θ, S, z, thermodynamics) * ∂²η⁰∂θ∂S(θ, S, thermodynamics)) / ∂η⁰∂θ(θ, S, thermodynamics)

"""
$(TYPEDSIGNATURES)

Specific heat capacity `cₚ = T (∂η/∂T)_S,z = T η⁰_θ / T_θ`.
"""
@inline heat_capacity(θ, S, z, thermodynamics) =
    temperature(θ, S, z, thermodynamics) * ∂η⁰∂θ(θ, S, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)

#=
Isothermal derivatives are (∂f/∂S)_T,z = f_S - f_θ T_S / T_θ.
(∂μ/∂z)_T,S = - (∂b/∂S)_T,z is a Maxwell relation.
=#

"""
$(TYPEDSIGNATURES)

Isothermal derivative `(∂b/∂S)_T,z = b_S - b_θ T_S / T_θ` of the buoyancy.
"""
@inline isothermal_∂b∂S(θ, S, z, thermodynamics) =
    ∂b∂S(θ, S, thermodynamics) - ∂b∂θ(θ, S, thermodynamics) * ∂T∂S(θ, S, z, thermodynamics) / ∂T∂θ(θ, S, z, thermodynamics)

"""
$(TYPEDSIGNATURES)

Isothermal derivative `(∂μ/∂S)_T,z = m_S - T η⁰_SS - η⁰_θ T_S² / T_θ` of the chemical potential.
"""
@inline isothermal_∂μ∂S(θ, S, z, thermodynamics) =
    ∂²h⁰∂S²(θ, S, thermodynamics) - ∂²b∂S²(θ, S, thermodynamics) * height_above_z₀(z, thermodynamics) -
    temperature(θ, S, z, thermodynamics) * ∂²η⁰∂S²(θ, S, thermodynamics) -
    ∂η⁰∂θ(θ, S, thermodynamics) * ∂T∂S(θ, S, z, thermodynamics)^2 / ∂T∂θ(θ, S, z, thermodynamics)
