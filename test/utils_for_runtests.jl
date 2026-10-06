using ForwardDiff: derivative
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: location
using Oceananigans.Operators: volume
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
using ThermoShenanigans: barodiffusion_coefficient

# Set TEST_ARCHITECTURE=GPU to also run the architecture tests on a GPU; CUDA is loaded only then
const test_gpu = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU"
test_gpu && @eval using CUDA
archs = test_gpu ? (CPU(), GPU()) : (CPU(),)

#####
##### Thermodynamics
#####

parameters = (thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
              reference_salinity = 0.035, reference_heat_capacity = 4000)

linear    = CabbelingBoussinesqThermodynamics(; parameters...)
cabbeling = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, parameters...)

# Test-only thermodynamics with θ–S coupling, h⁰ = c θ + a θ S, η⁰ = c ln θ - S (ln S - 1), b = gα θ.
# T_S and the cross term in j_θ vanish for CabbelingBoussinesqThermodynamics, so this exercises the general formulas.
struct CoupledTestThermodynamics <: AbstractBoussinesqThermodynamics
    c :: Float64
    a :: Float64
    gα :: Float64
end

import ThermoShenanigans: buoyancy, ∂b∂θ, ∂b∂S, ∂²b∂θ², ∂²b∂θ∂S, ∂²b∂S²,
                          potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                          entropy, ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²

for f in (:∂b∂S, :∂²b∂θ², :∂²b∂θ∂S, :∂²b∂S², :∂²h⁰∂θ², :∂²h⁰∂S², :∂²η⁰∂θ∂S)
    @eval $f(θ, S, ::CoupledTestThermodynamics) = zero(θ)
end

buoyancy(θ, S, t::CoupledTestThermodynamics)           = t.gα * θ
∂b∂θ(θ, S, t::CoupledTestThermodynamics)               = t.gα
potential_enthalpy(θ, S, t::CoupledTestThermodynamics) = t.c * θ + t.a * θ * S
∂h⁰∂θ(θ, S, t::CoupledTestThermodynamics)              = t.c + t.a * S
∂h⁰∂S(θ, S, t::CoupledTestThermodynamics)              = t.a * θ
∂²h⁰∂θ∂S(θ, S, t::CoupledTestThermodynamics)           = t.a
entropy(θ, S, t::CoupledTestThermodynamics)            = t.c * log(θ) - S * (log(S) - 1)
∂η⁰∂θ(θ, S, t::CoupledTestThermodynamics)              = t.c / θ
∂η⁰∂S(θ, S, t::CoupledTestThermodynamics)              = - log(S)
∂²η⁰∂θ²(θ, S, t::CoupledTestThermodynamics)            = - t.c / θ^2
∂²η⁰∂S²(θ, S, t::CoupledTestThermodynamics)            = - 1 / S

coupled = CoupledTestThermodynamics(4000, 100, 9.81 * 2e-4)

states = [(283.0, 0.035, 0.0), (275.5, 0.034, -50.0), (291.2, 0.0365, -1000.0)] # (θ, S, z)

# θ on the isotherm T(θ, S, z) = T₀ (Newton; differentiable with ForwardDiff)
function isothermal_θ(T₀, S, z, t; θ = T₀ + zero(S * z))
    for _ in 1:30
        θ -= (temperature(θ, S, z, t) - T₀) / ∂T∂θ(θ, S, z, t)
    end
    return θ
end

# θ on the isentrope η⁰(θ, S) = η₀ (Newton; differentiable with ForwardDiff)
function isentropic_θ(η₀, S, t; θ = 283 + zero(η₀ * S))
    for _ in 1:30
        θ -= (entropy(θ, S, t) - η₀) / ∂η⁰∂θ(θ, S, t)
    end
    return θ
end

#####
##### Models
#####

# Tests share one grid type (only size and extent vary) and one model builder, so that models share compiled kernels
test_grid(; size = (6, 6, 6), extent = (1, 1, 1)) = RectilinearGrid(; size, extent, topology = (Bounded, Bounded, Bounded))

component_model(grid, thermo; tracers = (:θ, :S), momentum_advection = Centered(), boundary_conditions = NamedTuple(),
                κ_T = 1e-3, kw...) =
    NonhydrostaticModel(grid; tracers, momentum_advection, boundary_conditions,
                        thermodynamic_model_components(grid, thermo; viscosity = 1e-3, thermal_diffusivity = κ_T,
                                                       salt_diffusivity = 1e-4, kw...)...)

# Prescribed fluxes of θ and S at the top and bottom (positive upwards), and a passive tracer c
const flux_bc_values = (θ = (top = 1e-4, bottom = -2e-4), S = (top = -3e-6, bottom = 1e-6))

flux_bc_model(grid) =
    component_model(grid, cabbeling; tracers = (:θ, :S, :c),
                    boundary_conditions = map(Q -> FieldBoundaryConditions(top = FluxBoundaryCondition(Q.top),
                                                                           bottom = FluxBoundaryCondition(Q.bottom)), flux_bc_values))

function set_random_state!(model; S = true)
    U(x, y, z) = 1e-2 * randn()
    θᵢ(x, y, z) = 283 + rand()
    Sᵢ(x, y, z) = 0.035 + 1e-3 * rand()
    S ? set!(model, u = U, v = U, w = U, θ = θᵢ, S = Sᵢ) : set!(model, u = U, v = U, w = U, θ = θᵢ)
end

# Uniform T₀ on a single column, with S marched upwards so that δS / Δz = ℑΓ on every interior face
function set_resting_equilibrium!(model, t; T₀ = 283.0, S_bottom = 0.035)
    grid = model.grid
    z, Nz, Δz = znodes(grid, Center()), size(grid, 3), zspacings(grid, Center())[1]
    S, θ = zeros(Nz), zeros(Nz)
    S[1] = S_bottom
    θ[1] = isothermal_θ(T₀, S[1], z[1], t)
    Γ(k) = barodiffusion_coefficient(θ[k], S[k], z[k], t)
    for k in 2:Nz
        S[k], θ[k] = S[k-1], θ[k-1]
        for _ in 1:50
            θ[k] = isothermal_θ(T₀, S[k], z[k], t)
            S[k] = S[k-1] + Δz * (Γ(k-1) + Γ(k)) / 2
        end
    end
    set!(model, θ = reshape(θ, 1, 1, Nz), S = reshape(S, 1, 1, Nz))
    return S, Γ(1), Δz
end

#####
##### Discrete integrals and fluxes
#####

field(op) = compute!(Field(op))
computed(op) = Array(interior(field(op)))

interior_values(f::Field) = Array(interior(f))
interior_values(op) = computed(op)
cell_volumes(f) = computed(KernelFunctionOperation{location(f)...}(volume, f.grid, map(L -> L(), location(f))...))

# ∫ g(f₁, f₂, ...) dV over the interior, with the cell volumes at the fields' common location. Plain array sums
# avoid building Oceananigans operations from the diagnostics, whose large types make each one slow to construct.
integral(g::Function, fields...) = sum(g.(map(interior_values, fields)...) .* cell_volumes(first(fields)))
integral(f) = integral(identity, f)

# Arguments of the tracer `id`'s closure fluxes, as passed by the model
closure_args(m, id) = (m.closure, m.closure_fields, Val(id), m.tracers[id], m.clock, merge(m.velocities, m.tracers), m.buoyancy)

flux_divergence(m, id) = KernelFunctionOperation{Center, Center, Center}(∇_dot_qᶜ, m.grid, closure_args(m, id)...)

# Face fluxes of tracer `id`, including the wall faces of Bounded directions
z_flux(m, id) = computed(KernelFunctionOperation{Center, Center, Face}(diffusive_flux_z, m.grid, closure_args(m, id)...))
fluxes(m, id) = (computed(KernelFunctionOperation{Face, Center, Center}(diffusive_flux_x, m.grid, closure_args(m, id)...)),
                 computed(KernelFunctionOperation{Center, Face, Center}(diffusive_flux_y, m.grid, closure_args(m, id)...)),
                 z_flux(m, id))
