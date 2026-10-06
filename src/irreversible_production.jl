#####
##### Kinetic energy dissipation ε = ∂ⱼuᵢ τᵢⱼ, at (Center, Center, Center)
#####

# Ported from Oceanostics.jl (v0.17.2, `viscous_dissipation_rate_ccc`): each term is A δuᵢ Fᵢⱼ at the
# native location of the viscous flux Fᵢⱼ, so that Σ ε V is exactly the discrete kinetic energy sink.
@inline Ax_δu_F₁₁ᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) = - Axᶜᶜᶜ(i, j, k, grid) * δxᶜᵃᵃ(i, j, k, grid, fields.u) * viscous_flux_ux(i, j, k, grid, closure, K, clock, fields, b)
@inline Ay_δu_F₁₂ᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) = - Ayᶠᶠᶜ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.u) * viscous_flux_uy(i, j, k, grid, closure, K, clock, fields, b)
@inline Az_δu_F₁₃ᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, b) = - Azᶠᶜᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.u) * viscous_flux_uz(i, j, k, grid, closure, K, clock, fields, b)
@inline Ax_δv_F₂₁ᶠᶠᶜ(i, j, k, grid, closure, K, clock, fields, b) = - Axᶠᶠᶜ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.v) * viscous_flux_vx(i, j, k, grid, closure, K, clock, fields, b)
@inline Ay_δv_F₂₂ᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) = - Ayᶜᶜᶜ(i, j, k, grid) * δyᵃᶜᵃ(i, j, k, grid, fields.v) * viscous_flux_vy(i, j, k, grid, closure, K, clock, fields, b)
@inline Az_δv_F₂₃ᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, b) = - Azᶜᶠᶠ(i, j, k, grid) * δzᵃᵃᶠ(i, j, k, grid, fields.v) * viscous_flux_vz(i, j, k, grid, closure, K, clock, fields, b)
@inline Ax_δw_F₃₁ᶠᶜᶠ(i, j, k, grid, closure, K, clock, fields, b) = - Axᶠᶜᶠ(i, j, k, grid) * δxᶠᵃᵃ(i, j, k, grid, fields.w) * viscous_flux_wx(i, j, k, grid, closure, K, clock, fields, b)
@inline Ay_δw_F₃₂ᶜᶠᶠ(i, j, k, grid, closure, K, clock, fields, b) = - Ayᶜᶠᶠ(i, j, k, grid) * δyᵃᶠᵃ(i, j, k, grid, fields.w) * viscous_flux_wy(i, j, k, grid, closure, K, clock, fields, b)
@inline Az_δw_F₃₃ᶜᶜᶜ(i, j, k, grid, closure, K, clock, fields, b) = - Azᶜᶜᶜ(i, j, k, grid) * δzᵃᵃᶜ(i, j, k, grid, fields.w) * viscous_flux_wz(i, j, k, grid, closure, K, clock, fields, b)

@inline viscous_dissipation_rateᶜᶜᶜ(i, j, k, grid, args...) = # args = (closure, K, clock, fields, b)
    (Ax_δu_F₁₁ᶜᶜᶜ(i, j, k, grid, args...) + ℑxyᶜᶜᵃ(i, j, k, grid, Ay_δu_F₁₂ᶠᶠᶜ, args...) + ℑxzᶜᵃᶜ(i, j, k, grid, Az_δu_F₁₃ᶠᶜᶠ, args...) +
     ℑxyᶜᶜᵃ(i, j, k, grid, Ax_δv_F₂₁ᶠᶠᶜ, args...) + Ay_δv_F₂₂ᶜᶜᶜ(i, j, k, grid, args...) + ℑyzᵃᶜᶜ(i, j, k, grid, Az_δv_F₂₃ᶜᶠᶠ, args...) +
     ℑxzᶜᵃᶜ(i, j, k, grid, Ax_δw_F₃₁ᶠᶜᶠ, args...) + ℑyzᵃᶜᶜ(i, j, k, grid, Ay_δw_F₃₂ᶜᶠᶠ, args...) + Az_δw_F₃₃ᶜᶜᶜ(i, j, k, grid, args...)) /
    Vᶜᶜᶜ(i, j, k, grid)

#####
##### Contractions of the fluxes with gradients, formed at faces and averaged to (Center, Center, Center)
#####

@inline heat_entropy_coefficient(θ, S, z, t) = heat_capacity(θ, S, z, t) / temperature(θ, S, z, t)^2 # cₚ / T²
@inline salt_entropy_coefficient(θ, S, z, t) = isothermal_∂μ∂S(θ, S, z, t) / temperature(θ, S, z, t) # (∂μ/∂S)_T / T

for (dir, δ, ∂, ℑᶠ, A, V, wall_mask) in ((:x, δxᶠᵃᵃ, ∂xᶠᶜᶜ, ℑxᶠᵃᵃ, Axᶠᶜᶜ, Vᶠᶜᶜ, biharmonic_mask_x),
                                          (:y, δyᵃᶠᵃ, ∂yᶜᶠᶜ, ℑyᵃᶠᵃ, Ayᶜᶠᶜ, Vᶜᶠᶜ, biharmonic_mask_y),
                                          (:z, δzᵃᵃᶠ, ∂zᶜᶜᶠ, ℑzᵃᵃᶠ, Azᶜᶜᶠ, Vᶜᶜᶠ, biharmonic_mask_z))

    heat_flux                   = Symbol(:masked_heat_flux_, dir)
    salt_flux                   = Symbol(:masked_salt_flux_, dir)
    salt_gradient               = Symbol(:salt_gradient_, dir)
    face_contraction            = Symbol(:face_contraction_, dir)
    buoyancy_flux               = Symbol(:buoyancy_flux_, dir)
    face_entropy_production     = Symbol(:face_entropy_production_, dir)
    unmasked_entropy_production = Symbol(:unmasked_entropy_production_, dir)
    salt_entropy_production     = Symbol(:salt_entropy_production_, dir)

    @eval begin
        # A (j_θ δfθ + j_S δfS)
        @inline $face_contraction(i, j, k, grid, fθ, fS, cl, tb, C) =
            $A(i, j, k, grid) * ($heat_flux(i, j, k, grid, cl, tb, C) * $δ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, fθ, tb, C) +
                                 $salt_flux(i, j, k, grid, cl, tb, C) * $δ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, fS, tb, C))

        # j_b = ℑ(b_θ) j_θ + ℑ(b_S) j_S
        @inline $buoyancy_flux(i, j, k, grid, cl, tb, C) =
            $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_θ, tb, C) * $heat_flux(i, j, k, grid, cl, tb, C) +
            $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, buoyancy_S, tb, C) * $salt_flux(i, j, k, grid, cl, tb, C)

        # V [κ_T ℑ(cₚ/T²) (∂T)² + κ_S ℑ((∂μ/∂S)_T / T) (∂S - Γ)²] ≥ 0
        @inline $salt_entropy_production(i, j, k, grid, cl, tb, C) =
            cl.salt_diffusivity * $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, salt_entropy_coefficient, tb, C) *
                                  $salt_gradient(i, j, k, grid, tb, C)^2
        @inline $salt_entropy_production(i, j, k, grid, cl, tb::ConstantSalinityThermodynamicBuoyancy, C) = zero(grid)

        @inline $unmasked_entropy_production(i, j, k, grid, cl, tb, C) =
            $V(i, j, k, grid) * (cl.thermal_diffusivity * $ℑᶠ(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, heat_entropy_coefficient, tb, C) *
                                                          $∂(i, j, k, grid, thermodynamic_functionᶜᶜᶜ, temperature, tb, C)^2 +
                                 $salt_entropy_production(i, j, k, grid, cl, tb, C))

        @inline $face_entropy_production(i, j, k, grid, cl, tb, C) = $wall_mask(i, j, k, grid, $unmasked_entropy_production, cl, tb, C)
    end
end

# j_θ·∇fθ + j_S·∇fS at (Center, Center, Center); by summation by parts Σ V (j_θ·∇fθ) = - Σ V fθ ∇·j_θ exactly
@inline flux_contractionᶜᶜᶜ(i, j, k, grid, fθ, fS, cl, tb, C) =
    (ℑxᶜᵃᵃ(i, j, k, grid, face_contraction_x, fθ, fS, cl, tb, C) +
     ℑyᵃᶜᵃ(i, j, k, grid, face_contraction_y, fθ, fS, cl, tb, C) +
     ℑzᵃᵃᶜ(i, j, k, grid, face_contraction_z, fθ, fS, cl, tb, C)) / Vᶜᶜᶜ(i, j, k, grid)

#####
##### Irreversible production terms at (Center, Center, Center); p = (; viscous_closure, closure_fields, diffusivity, buoyancy)
#####

@inline dissipationᶜᶜᶜ(i, j, k, grid, clock, C, p) = viscous_dissipation_rateᶜᶜᶜ(i, j, k, grid, p.viscous_closure, p.closure_fields, clock, C, p.buoyancy)

# j_θ·∇π + j_S·∇Σ_S, the conversion of static energy by the fluxes
@inline static_energy_conversionᶜᶜᶜ(i, j, k, grid, clock, C, p) = flux_contractionᶜᶜᶜ(i, j, k, grid, exner, chemical_potential_analogue, p.diffusivity, p.buoyancy, C)

# σ_b_nonlin = j_θ·∇b_θ + j_S·∇b_S, nonzero only for a nonlinear equation of state
@inline nonlinear_buoyancy_productionᶜᶜᶜ(i, j, k, grid, clock, C, p) = flux_contractionᶜᶜᶜ(i, j, k, grid, buoyancy_θ, buoyancy_S, p.diffusivity, p.buoyancy, C)

# σ_θ = [ε - j_θ·∇π - j_S·∇Σ_S] / π, so that π (σ_θ - ∇·j_θ) - Σ_S ∇·j_S = ε - ∇·(π j_θ + Σ_S j_S):
# the static energy Σ gains exactly the kinetic energy lost to viscosity, plus a flux divergence (first law)
# The arguments follow the discrete-form `Forcing` signature, so σ_θ is used directly as the θ forcing
@inline θ_productionᶜᶜᶜ(i, j, k, grid, clock, C, p) =
    (dissipationᶜᶜᶜ(i, j, k, grid, clock, C, p) - static_energy_conversionᶜᶜᶜ(i, j, k, grid, clock, C, p)) /
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, p.buoyancy, C)

# σ_b = b_θ σ_θ + σ_b_nonlin, so that Db/Dt = σ_b - ∇·j_b
@inline buoyancy_productionᶜᶜᶜ(i, j, k, grid, clock, C, p) =
    thermodynamic_functionᶜᶜᶜ(i, j, k, grid, buoyancy_θ, p.buoyancy, C) * θ_productionᶜᶜᶜ(i, j, k, grid, clock, C, p) +
    nonlinear_buoyancy_productionᶜᶜᶜ(i, j, k, grid, clock, C, p)

# σ_η = ε / T + κ_T cₚ |∇T|² / T² + κ_S (∂μ/∂S)_T |∇S - Γ 𝐤|² / T ≥ 0
@inline entropy_productionᶜᶜᶜ(i, j, k, grid, clock, C, p) =
    dissipationᶜᶜᶜ(i, j, k, grid, clock, C, p) / thermodynamic_functionᶜᶜᶜ(i, j, k, grid, temperature, p.buoyancy, C) +
    (ℑxᶜᵃᵃ(i, j, k, grid, face_entropy_production_x, p.diffusivity, p.buoyancy, C) +
     ℑyᵃᶜᵃ(i, j, k, grid, face_entropy_production_y, p.diffusivity, p.buoyancy, C) +
     ℑzᵃᵃᶜ(i, j, k, grid, face_entropy_production_z, p.diffusivity, p.buoyancy, C)) / Vᶜᶜᶜ(i, j, k, grid)
