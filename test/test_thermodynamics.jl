using ForwardDiff: derivative
using ThermoShenanigans: thermodynamic_interface

parameters = (thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
              reference_salinity = 0.035, reference_heat_capacity = 4000)

linear    = CabbelingBoussinesqThermodynamics(; parameters...)
cabbeling = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, parameters...)

# A minimal user-defined thermodynamics with a non-zero isothermal ∂θ/∂S:
# h⁰ = c θ + a θ S, η⁰ = c ln θ - S (ln S - 1), b = gα θ
struct CrossThermodynamics <: AbstractBoussinesqThermodynamics
    c :: Float64
    a :: Float64
    gα :: Float64
end

import ThermoShenanigans: buoyancy, ∂b∂θ, ∂b∂S, ∂²b∂θ², ∂²b∂θ∂S, ∂²b∂S²,
                          potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                          entropy, ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²

for f in (:∂b∂S, :∂²b∂θ², :∂²b∂θ∂S, :∂²b∂S², :∂²h⁰∂θ², :∂²h⁰∂S², :∂²η⁰∂θ∂S)
    @eval $f(θ, S, ::CrossThermodynamics) = zero(θ)
end

buoyancy(θ, S, t::CrossThermodynamics)           = t.gα * θ
∂b∂θ(θ, S, t::CrossThermodynamics)               = t.gα
potential_enthalpy(θ, S, t::CrossThermodynamics) = t.c * θ + t.a * θ * S
∂h⁰∂θ(θ, S, t::CrossThermodynamics)              = t.c + t.a * S
∂h⁰∂S(θ, S, t::CrossThermodynamics)              = t.a * θ
∂²h⁰∂θ∂S(θ, S, t::CrossThermodynamics)           = t.a
entropy(θ, S, t::CrossThermodynamics)            = t.c * log(θ) - S * (log(S) - 1)
∂η⁰∂θ(θ, S, t::CrossThermodynamics)              = t.c / θ
∂η⁰∂S(θ, S, t::CrossThermodynamics)              = - log(S)
∂²η⁰∂θ²(θ, S, t::CrossThermodynamics)            = - t.c / θ^2
∂²η⁰∂S²(θ, S, t::CrossThermodynamics)            = - 1 / S

cross = CrossThermodynamics(4000, 100, 9.81 * 2e-4)

states = [(283.0, 0.035, 0.0), (275.5, 0.034, -50.0), (291.2, 0.0365, -1000.0)]

@testset "Thermodynamics" begin
    @testset "Derivatives match ForwardDiff [$(summary(t))]" for t in (linear, cabbeling, cross)
        for (θ, S, z) in states
            ∂θ(f) = derivative(θ -> f(θ, S, t), θ)
            ∂S(f) = derivative(S -> f(θ, S, t), S)

            @test ∂b∂θ(θ, S, t)      ≈ ∂θ(buoyancy)             atol=1e-12
            @test ∂b∂S(θ, S, t)      ≈ ∂S(buoyancy)             atol=1e-12
            @test ∂²b∂θ²(θ, S, t)    ≈ ∂θ(∂b∂θ)                 atol=1e-12
            @test ∂²b∂θ∂S(θ, S, t)   ≈ ∂S(∂b∂θ)                 atol=1e-12
            @test ∂²b∂S²(θ, S, t)    ≈ ∂S(∂b∂S)                 atol=1e-12
            @test ∂h⁰∂θ(θ, S, t)     ≈ ∂θ(potential_enthalpy)   atol=1e-12
            @test ∂h⁰∂S(θ, S, t)     ≈ ∂S(potential_enthalpy)   atol=1e-12
            @test ∂²h⁰∂θ²(θ, S, t)   ≈ ∂θ(∂h⁰∂θ)                atol=1e-12
            @test ∂²h⁰∂θ∂S(θ, S, t)  ≈ ∂S(∂h⁰∂θ)                atol=1e-12
            @test ∂²h⁰∂S²(θ, S, t)   ≈ ∂S(∂h⁰∂S)                atol=1e-12
            @test ∂η⁰∂θ(θ, S, t)     ≈ ∂θ(entropy)              atol=1e-12
            @test ∂η⁰∂S(θ, S, t)     ≈ ∂S(entropy)              atol=1e-12
            @test ∂²η⁰∂θ²(θ, S, t)   ≈ ∂θ(∂η⁰∂θ)                atol=1e-12
            @test ∂²η⁰∂θ∂S(θ, S, t)  ≈ ∂S(∂η⁰∂θ)                atol=1e-12
            @test ∂²η⁰∂S²(θ, S, t)   ≈ ∂S(∂η⁰∂S)                atol=1e-12

            # π = Σ_θ, μ = Σ_S, and T_θ, T_S, cₚ = T (∂η/∂T)_S,z from ForwardDiff of Σ, T and η⁰
            @test exner(θ, S, z, t)              ≈ derivative(θ -> static_energy(θ, S, z, t), θ)
            @test chemical_potential(θ, S, z, t) ≈ derivative(S -> static_energy(θ, S, z, t), S) atol=1e-12
            @test ∂T∂θ(θ, S, z, t) ≈ derivative(θ -> in_situ_temperature(θ, S, z, t), θ)
            @test ∂T∂S(θ, S, z, t) ≈ derivative(S -> in_situ_temperature(θ, S, z, t), S) atol=1e-12
            @test heat_capacity(θ, S, z, t) ≈ in_situ_temperature(θ, S, z, t) * ∂θ(entropy) /
                                              derivative(θ -> in_situ_temperature(θ, S, z, t), θ)
        end
    end

    @testset "Closed forms" begin
        g, α, β, γ, θᵣ, cₚ⁰ = cabbeling.gravitational_acceleration, cabbeling.thermal_expansion, cabbeling.haline_contraction,
                              cabbeling.cabbeling_coefficient, cabbeling.reference_temperature, cabbeling.reference_heat_capacity

        for (θ, S, z) in states
            @test in_situ_temperature(θ, S, z, linear) ≈ θ * (1 - g * α * z / cₚ⁰)
            @test chemical_potential(θ, S, z, linear)  ≈ g * β * z
            @test heat_capacity(θ, S, z, linear)       ≈ cₚ⁰

            π = cₚ⁰ - g * (α + γ * (θ - θᵣ)) * z
            @test exner(θ, S, z, cabbeling)               ≈ π
            @test in_situ_temperature(θ, S, z, cabbeling) ≈ π * θ / cₚ⁰
            @test heat_capacity(θ, S, z, cabbeling)       ≈ cₚ⁰ * π / (π - g * γ * z * θ)
            @test isothermal_∂θ∂S(θ, S, z, cabbeling) == 0

            # Maxwell relation (∂cₚ/∂p)_T = - T υ_TT gives ∂cₚ/∂z = g γ θ at the surface
            @test derivative(z -> heat_capacity(θ, S, z, cabbeling), 0.0) ≈ g * γ * θ

            c, a = cross.c, cross.a
            @test isothermal_∂θ∂S(θ, S, 0.0, cross) ≈ - a * θ / (c + a * S)
        end

        @test linear.saline_entropy_constant ≈ 8.314462618 / 31.4038218e-3
    end

    @testset "Float32" begin
        t = CabbelingBoussinesqThermodynamics(Float32; cabbeling_coefficient = 1e-5, parameters...)
        θ, S, z = 283f0, 0.035f0, -10f0
        @test all(getfield(ThermoShenanigans, f)(θ, S, t) isa Float32 for f in thermodynamic_interface)
        @test all(f(θ, S, z, t) isa Float32 for f in (static_energy, exner, chemical_potential, in_situ_temperature,
                                                       ∂T∂θ, ∂T∂S, heat_capacity, isothermal_∂θ∂S))
    end
end
