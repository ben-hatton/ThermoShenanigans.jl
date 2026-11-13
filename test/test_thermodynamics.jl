@testset "Thermodynamics" begin
    @testset "Derivatives match ForwardDiff [$(summary(thermodynamics))]" for thermodynamics in (linear, cabbeling, coupled, shifted)
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

            # π = Σ_θ, m, and T_θ, T_S, cₚ = T (∂η/∂T)_S,z from ForwardDiff of Σ, T and η⁰
            @test exner(θ, S, z, thermodynamics)                       ≈ derivative(θ -> potential_energy(θ, S, z, thermodynamics), θ)
            @test chemical_potential_analogue(θ, S, z, thermodynamics) ≈ derivative(S -> potential_energy(θ, S, z, thermodynamics), S) atol=1e-12

            # T = (∂Σ/∂η)_S and μ = (∂Σ/∂S)_η, along θ(η, S)
            η = entropy(θ, S, thermodynamics)
            @test temperature(θ, S, z, thermodynamics)        ≈ derivative(η -> potential_energy(isentropic_θ(η, S, thermodynamics; θ), S, z, thermodynamics), η)
            @test chemical_potential(θ, S, z, thermodynamics) ≈ derivative(S -> potential_energy(isentropic_θ(η, S, thermodynamics; θ), S, z, thermodynamics), S)
            @test ∂T∂θ(θ, S, z, thermodynamics) ≈ derivative(θ -> temperature(θ, S, z, thermodynamics), θ)
            @test ∂T∂S(θ, S, z, thermodynamics) ≈ derivative(S -> temperature(θ, S, z, thermodynamics), S) atol=1e-12
            @test heat_capacity(θ, S, z, thermodynamics) ≈ temperature(θ, S, z, thermodynamics) * ∂θ(entropy) /
                                              derivative(θ -> temperature(θ, S, z, thermodynamics), θ)
        end
    end

    @testset "Closed forms" begin
        g, α, β, γ, T₀, cₚ₀ = cabbeling.gravitational_acceleration, cabbeling.thermal_expansion, cabbeling.haline_contraction,
                              cabbeling.cabbeling_coefficient, cabbeling.reference_temperature, cabbeling.reference_heat_capacity
        S₀, R_S = cabbeling.reference_salinity, cabbeling.saline_entropy_constant

        # Gauge of Tailleux, Dubos & Hatton: b, h⁰, η⁰ and μ - m vanish at (T₀, S₀)
        @test buoyancy(T₀, S₀, cabbeling) == 0
        @test potential_enthalpy(T₀, S₀, cabbeling) == 0
        @test entropy(T₀, S₀, cabbeling) == 0
        @test ∂η⁰∂S(T₀, S₀, cabbeling) == 0

        for (θ, S, z) in states
            @test temperature(θ, S, z, linear) ≈ θ * (1 - g * α * z / cₚ₀)
            @test chemical_potential_analogue(θ, S, z, linear) ≈ g * β * z
            @test heat_capacity(θ, S, z, linear)       ≈ cₚ₀

            π = cₚ₀ - g * (α + γ * (θ - T₀)) * z
            @test exner(θ, S, z, cabbeling)               ≈ π
            @test temperature(θ, S, z, cabbeling) ≈ π * θ / cₚ₀
            @test heat_capacity(θ, S, z, cabbeling)       ≈ cₚ₀ * π / (π - g * γ * z * θ)
            @test ∂T∂S(θ, S, z, cabbeling) == 0
            @test potential_energy(θ, S, z, cabbeling) ≈ cₚ₀ * (θ - T₀) - buoyancy(θ, S, cabbeling) * z
            @test entropy(θ, S, cabbeling) ≈ cₚ₀ * log(θ / T₀) - R_S * (S * log(S / S₀) - (S - S₀))
            @test chemical_potential(θ, S, z, cabbeling) ≈ g * β * z + R_S * π * θ / cₚ₀ * log(S / S₀)

            # Maxwell relation (∂cₚ/∂p)_T = - T υ_TT gives ∂cₚ/∂z = g γ θ at the surface
            @test derivative(z -> heat_capacity(θ, S, z, cabbeling), 0.0) ≈ g * γ * θ

            c, a = coupled.c, coupled.a
            @test - ∂T∂S(θ, S, 0.0, coupled) / ∂T∂θ(θ, S, 0.0, coupled) ≈ - a * θ / (c + a * S)

            # With z₀ ≠ 0, the functions of z depend on z - z₀ only, and 𝒫 gains the constant g z₀
            z₀ = reference_pressure_height(shifted)
            for f in (exner, chemical_potential_analogue, temperature, chemical_potential, heat_capacity)
                @test f(θ, S, z, shifted) ≈ f(θ, S, z - z₀, cabbeling)
            end
            @test potential_energy(θ, S, z, shifted) ≈ potential_energy(θ, S, z - z₀, cabbeling) + g * z₀
        end

        # The g of g z₀ in 𝒫 is the g in b: doubling g doubles 𝒫 - h⁰ = - b (z - z₀) + g z₀
        @test gravitational_acceleration(cabbeling) === g
        doubled_gravity = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, reference_pressure_height = -40,
                                                            gravitational_acceleration = 2g, parameters...)
        for (θ, S, z) in states
            @test potential_energy(θ, S, z, doubled_gravity) - potential_enthalpy(θ, S, doubled_gravity) ≈
                  2 * (potential_energy(θ, S, z, shifted) - potential_enthalpy(θ, S, shifted))
        end

        @test linear.saline_entropy_constant ≈ 8.314462618 / 31.4038218e-3
    end

    @testset "Float32" begin
        thermodynamics = CabbelingBoussinesqThermodynamics(Float32; cabbeling_coefficient = 1e-5, reference_pressure_height = -40,
                                                           parameters...)
        θ, S, z = 283f0, 0.035f0, -10f0
        @test reference_pressure_height(thermodynamics) isa Float32
        @test gravitational_acceleration(thermodynamics) isa Float32
        interface = (buoyancy,           ∂b∂θ,  ∂b∂S,  ∂²b∂θ²,  ∂²b∂θ∂S,  ∂²b∂S²,
                     potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                     entropy,            ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²)
        @test all(f(θ, S, thermodynamics) isa Float32 for f in interface)
        @test all(f(θ, S, z, thermodynamics) isa Float32 for f in (potential_energy, exner, chemical_potential_analogue, temperature, chemical_potential,
                                                       ∂T∂θ, ∂T∂S, heat_capacity, isothermal_∂b∂S, isothermal_∂μ∂S))
    end
end
