@testset "Thermodynamics" begin
    @testset "Derivatives match ForwardDiff [$(summary(thermodynamics))]" for thermodynamics in (linear, cabbeling, coupled)
        for (θ, S, z) in states
            ∂θ(f) = derivative(θ -> f(θ, S, thermodynamics), θ)
            ∂S(f) = derivative(S -> f(θ, S, thermodynamics), S)

            @test ∂b∂θ(θ, S, thermodynamics)      ≈ ∂θ(buoyancy)             atol=1e-12
            @test ∂b∂S(θ, S, thermodynamics)      ≈ ∂S(buoyancy)             atol=1e-12
            @test ∂²b∂θ²(θ, S, thermodynamics)    ≈ ∂θ(∂b∂θ)                 atol=1e-12
            @test ∂²b∂θ∂S(θ, S, thermodynamics)   ≈ ∂S(∂b∂θ)                 atol=1e-12
            @test ∂²b∂S²(θ, S, thermodynamics)    ≈ ∂S(∂b∂S)                 atol=1e-12
            @test ∂h⁰∂θ(θ, S, thermodynamics)     ≈ ∂θ(potential_enthalpy)   atol=1e-12
            @test ∂h⁰∂S(θ, S, thermodynamics)     ≈ ∂S(potential_enthalpy)   atol=1e-12
            @test ∂²h⁰∂θ²(θ, S, thermodynamics)   ≈ ∂θ(∂h⁰∂θ)                atol=1e-12
            @test ∂²h⁰∂θ∂S(θ, S, thermodynamics)  ≈ ∂S(∂h⁰∂θ)                atol=1e-12
            @test ∂²h⁰∂S²(θ, S, thermodynamics)   ≈ ∂S(∂h⁰∂S)                atol=1e-12
            @test ∂η⁰∂θ(θ, S, thermodynamics)     ≈ ∂θ(entropy)              atol=1e-12
            @test ∂η⁰∂S(θ, S, thermodynamics)     ≈ ∂S(entropy)              atol=1e-12
            @test ∂²η⁰∂θ²(θ, S, thermodynamics)   ≈ ∂θ(∂η⁰∂θ)                atol=1e-12
            @test ∂²η⁰∂θ∂S(θ, S, thermodynamics)  ≈ ∂S(∂η⁰∂θ)                atol=1e-12
            @test ∂²η⁰∂S²(θ, S, thermodynamics)   ≈ ∂S(∂η⁰∂S)                atol=1e-12

            # π = Σ_θ, Σ_S, and T_θ, T_S, cₚ = T (∂η/∂T)_S,z from ForwardDiff of Σ, T and η⁰
            @test exner(θ, S, z, thermodynamics)                       ≈ derivative(θ -> static_energy(θ, S, z, thermodynamics), θ)
            @test chemical_potential_analogue(θ, S, z, thermodynamics) ≈ derivative(S -> static_energy(θ, S, z, thermodynamics), S) atol=1e-12

            # T = (∂Σ/∂η)_S and μ = (∂Σ/∂S)_η, along θ(η, S)
            η = entropy(θ, S, thermodynamics)
            @test temperature(θ, S, z, thermodynamics)        ≈ derivative(η -> static_energy(isentropic_θ(η, S, thermodynamics; θ), S, z, thermodynamics), η)
            @test chemical_potential(θ, S, z, thermodynamics) ≈ derivative(S -> static_energy(isentropic_θ(η, S, thermodynamics; θ), S, z, thermodynamics), S)
            @test ∂T∂θ(θ, S, z, thermodynamics) ≈ derivative(θ -> temperature(θ, S, z, thermodynamics), θ)
            @test ∂T∂S(θ, S, z, thermodynamics) ≈ derivative(S -> temperature(θ, S, z, thermodynamics), S) atol=1e-12
            @test heat_capacity(θ, S, z, thermodynamics) ≈ temperature(θ, S, z, thermodynamics) * ∂θ(entropy) /
                                              derivative(θ -> temperature(θ, S, z, thermodynamics), θ)
        end
    end

    @testset "Closed forms" begin
        g, α, β, γ, θᵣ, cₚ⁰ = cabbeling.gravitational_acceleration, cabbeling.thermal_expansion, cabbeling.haline_contraction,
                              cabbeling.cabbeling_coefficient, cabbeling.reference_temperature, cabbeling.reference_heat_capacity

        for (θ, S, z) in states
            @test temperature(θ, S, z, linear) ≈ θ * (1 - g * α * z / cₚ⁰)
            @test chemical_potential_analogue(θ, S, z, linear) ≈ g * β * z
            @test heat_capacity(θ, S, z, linear)       ≈ cₚ⁰

            π = cₚ⁰ - g * (α + γ * (θ - θᵣ)) * z
            @test exner(θ, S, z, cabbeling)               ≈ π
            @test temperature(θ, S, z, cabbeling) ≈ π * θ / cₚ⁰
            @test heat_capacity(θ, S, z, cabbeling)       ≈ cₚ⁰ * π / (π - g * γ * z * θ)
            @test ∂T∂S(θ, S, z, cabbeling) == 0

            # Maxwell relation (∂cₚ/∂p)_T = - T υ_TT gives ∂cₚ/∂z = g γ θ at the surface
            @test derivative(z -> heat_capacity(θ, S, z, cabbeling), 0.0) ≈ g * γ * θ

            c, a = coupled.c, coupled.a
            @test - ∂T∂S(θ, S, 0.0, coupled) / ∂T∂θ(θ, S, 0.0, coupled) ≈ - a * θ / (c + a * S)
        end

        @test linear.saline_entropy_constant ≈ 8.314462618 / 31.4038218e-3
    end

    @testset "Float32" begin
        thermodynamics = CabbelingBoussinesqThermodynamics(Float32; cabbeling_coefficient = 1e-5, parameters...)
        θ, S, z = 283f0, 0.035f0, -10f0
        interface = (buoyancy,           ∂b∂θ,  ∂b∂S,  ∂²b∂θ²,  ∂²b∂θ∂S,  ∂²b∂S²,
                     potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                     entropy,            ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²)
        @test all(f(θ, S, thermodynamics) isa Float32 for f in interface)
        @test all(f(θ, S, z, thermodynamics) isa Float32 for f in (static_energy, exner, chemical_potential_analogue, temperature, chemical_potential,
                                                       ∂T∂θ, ∂T∂S, heat_capacity, isothermal_∂b∂S, isothermal_∂μ∂S))
    end
end
