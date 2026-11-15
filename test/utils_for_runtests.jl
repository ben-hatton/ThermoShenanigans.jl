using ForwardDiff: derivative
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: location
using Oceananigans.Operators: volume
using Oceananigans.TimeSteppers: update_state!, compute_tendencies!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
using ThermoShenanigans: barodiffusion_coefficient, θ_flux_prefactor

# TEST_GROUP=basic (the default) runs a quick core set; TEST_GROUP=all runs the full suite
const test_group = get(ENV, "TEST_GROUP", "basic")

# The doctests and the explicit-imports check run with TEST_GROUP=all, and with TEST_GROUP=basic when
# TEST_DOCTESTS_AND_IMPORTS=true, after a change to a docstring, an export or an import
const test_doctests_and_imports = get(ENV, "TEST_DOCTESTS_AND_IMPORTS", "false") == "true"

# Test files of each group, in the order they run
test_files(group) = group == "all" ? ["test_thermodynamics.jl", "test_buoyancy.jl", "test_diffusivity.jl", "test_production.jl",
                                      "test_energy_conservation.jl", "test_energetics.jl", "test_architectures.jl"] :
                                     ["test_thermodynamics.jl", "test_basic.jl"]

# Each test file `test_name.jl` defines a function `test_name()` that runs its tests, compiled once per session. The call
# goes through `invokelatest` since the function is defined after the caller was compiled.
run_test_file(file) = Base.invokelatest(getproperty(@__MODULE__, Symbol(splitext(basename(file))[1])))

function run_doctests_and_imports()
    @testset "Explicit imports" begin
        @test isnothing(check_no_implicit_imports(ThermoShenanigans))
        # diffusive_flux_x/y/z are extended inside an @eval loop through interpolated Symbols, which ExplicitImports cannot see
        @test isnothing(check_no_stale_explicit_imports(ThermoShenanigans; ignore = (:diffusive_flux_x, :diffusive_flux_y, :diffusive_flux_z)))
    end

    doctest(ThermoShenanigans; manual = false)
end

# Set TEST_ARCHITECTURE=GPU to also run the architecture tests on a GPU. CUDA is not a test dependency, since installing
# and precompiling it dominates CPU-only runs, so it is added to the test environment only then, unless the environment
# already has it (as `benchmark/Project.toml`, used by `benchmark/gpu.slurm` on compute nodes without internet, does).
const test_gpu = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU"
if test_gpu
    if isnothing(Base.find_package("CUDA"))
        using Pkg: Pkg
        Pkg.add("CUDA")
    end
    @eval using CUDA
end
archs = test_gpu ? (CPU(), GPU()) : (CPU(),)

# Thermodynamics

parameters = (thermal_expansion = 2e-4, haline_contraction = 0.8, reference_temperature = 283,
              reference_salinity = 0.035, reference_heat_capacity = 4000)

linear    = CabbelingBoussinesqThermodynamics(; parameters...)
cabbeling = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, parameters...)

# As `cabbeling`, but with the reference pressure P₀ at z₀ = -40 rather than at z = 0
shifted = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, reference_pressure_height = -40, parameters...)

# Test-only thermodynamics with θ–S coupling, h⁰ = c θ + a θ S, η⁰ = c ln θ - S (ln S - 1), b = gα θ.
# T_S and the cross term in j_θ vanish for CabbelingBoussinesqThermodynamics, so this exercises the general formulas.
struct CoupledTestThermodynamics <: AbstractBoussinesqThermodynamics
    c :: Float64
    a :: Float64
    gα :: Float64
end

import ThermoShenanigans: buoyancy, ∂b∂θ, ∂b∂S, ∂²b∂θ², ∂²b∂θ∂S, ∂²b∂S²,
                          potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                          entropy, ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²,
                          reference_pressure_height, gravitational_acceleration

for f in (:∂b∂S, :∂²b∂θ², :∂²b∂θ∂S, :∂²b∂S², :∂²h⁰∂θ², :∂²h⁰∂S², :∂²η⁰∂θ∂S)
    @eval $f(θ, S, ::CoupledTestThermodynamics) = zero(θ)
end

buoyancy(θ, S, thermodynamics::CoupledTestThermodynamics)           = thermodynamics.gα * θ
∂b∂θ(θ, S, thermodynamics::CoupledTestThermodynamics)               = thermodynamics.gα
potential_enthalpy(θ, S, thermodynamics::CoupledTestThermodynamics) = thermodynamics.c * θ + thermodynamics.a * θ * S
∂h⁰∂θ(θ, S, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.c + thermodynamics.a * S
∂h⁰∂S(θ, S, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.a * θ
∂²h⁰∂θ∂S(θ, S, thermodynamics::CoupledTestThermodynamics)           = thermodynamics.a
entropy(θ, S, thermodynamics::CoupledTestThermodynamics)            = thermodynamics.c * log(θ) - S * (log(S) - 1)
∂η⁰∂θ(θ, S, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.c / θ
∂η⁰∂S(θ, S, thermodynamics::CoupledTestThermodynamics)              = - log(S)
∂²η⁰∂θ²(θ, S, thermodynamics::CoupledTestThermodynamics)            = - thermodynamics.c / θ^2
∂²η⁰∂S²(θ, S, thermodynamics::CoupledTestThermodynamics)            = - 1 / S

reference_pressure_height(::CoupledTestThermodynamics)  = 0.0
gravitational_acceleration(::CoupledTestThermodynamics) = 9.81

coupled = CoupledTestThermodynamics(4000, 100, 9.81 * 2e-4)

states = [(283.0, 0.035, 0.0), (275.5, 0.034, -50.0), (291.2, 0.0365, -1000.0)] # (θ, S, z)

# θ on the isotherm T(θ, S, z) = T_isotherm (Newton; differentiable with ForwardDiff)
function isothermal_θ(T_isotherm, S, z, thermodynamics; θ = T_isotherm + zero(S * z))
    for _ in 1:30
        θ -= (temperature(θ, S, z, thermodynamics) - T_isotherm) / ∂T∂θ(θ, S, z, thermodynamics)
    end
    return θ
end

# θ on the isentrope η⁰(θ, S) = η₀ (Newton; differentiable with ForwardDiff)
function isentropic_θ(η₀, S, thermodynamics; θ = 283 + zero(η₀ * S))
    for _ in 1:30
        θ -= (entropy(θ, S, thermodynamics) - η₀) / ∂η⁰∂θ(θ, S, thermodynamics)
    end
    return θ
end

# Models

# Tests share one grid type (only size and extent vary) and one model builder, so that models share compiled kernels
test_grid(; size = (6, 6, 6), extent = (1, 1, 1)) = RectilinearGrid(; size, extent, topology = (Bounded, Bounded, Bounded))

component_model(grid, thermo; tracers = (:θ, :S), momentum_advection = Centered(), boundary_conditions = NamedTuple(),
                κ_θ = 1e-3, model_kwargs...) =
    NonhydrostaticModel(grid; tracers, momentum_advection, boundary_conditions,
                        thermodynamic_model_components(grid, thermo; viscosity = 1e-3, thermal_diffusivity = κ_θ,
                                                       salt_diffusivity = 1e-4, model_kwargs...)...)

# Models on the test grid, z ∈ (-1, 0) with z₀ = 0, and on that grid translated up by `shift`, with z₀ = `shift` (the
# same dynamics) and with z₀ = 0 (different dynamics), all in the same smooth state
function translated_models(; shift = 10)
    models = map(((0, 0), (shift, shift), (shift, 0))) do (top, z₀)
        grid = RectilinearGrid(size = (6, 6, 6), x = (0, 1), y = (0, 1), z = (top - 1, top), topology = (Bounded, Bounded, Bounded))
        thermodynamics = CabbelingBoussinesqThermodynamics(; cabbeling_coefficient = 1e-5, reference_pressure_height = z₀,
                                                           parameters...)
        component_model(grid, thermodynamics)
    end

    reference = first(models)
    U(x, y, z) = 1e-2 * sin(2π * x) * cos(π * y) * cos(π * z)
    θᵢ(x, y, z) = 283 + sin(2π * x) * sin(π * y) + z
    Sᵢ(x, y, z) = 0.035 + 1e-3 * cos(2π * x) * cos(π * z)
    set!(reference, u = U, v = U, θ = θᵢ, S = Sᵢ)

    # The same values rather than the same functions of z - top, so that the translated states are identical
    values = map(field -> Array(interior(field)), merge(reference.velocities, reference.tracers))
    for model in models[2:end]
        set!(model; values...)
    end

    return models
end

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

# Uniform T_isotherm on a single column, with S marched upwards so that δS / Δz = - ℑΓ_S on every interior face
function set_resting_equilibrium!(model, thermodynamics; T_isotherm = 283.0, S_bottom = 0.035)
    grid = model.grid
    z, Nz, Δz = znodes(grid, Center()), size(grid, 3), zspacings(grid, Center())[1]
    S, θ = zeros(Nz), zeros(Nz)
    S[1] = S_bottom
    θ[1] = isothermal_θ(T_isotherm, S[1], z[1], thermodynamics)
    Γ_S(k) = barodiffusion_coefficient(θ[k], S[k], z[k], thermodynamics)
    for k in 2:Nz
        S[k], θ[k] = S[k-1], θ[k-1]
        for _ in 1:50
            θ[k] = isothermal_θ(T_isotherm, S[k], z[k], thermodynamics)
            S[k] = S[k-1] - Δz * (Γ_S(k-1) + Γ_S(k)) / 2
        end
    end
    set!(model, θ = reshape(θ, 1, 1, Nz), S = reshape(S, 1, 1, Nz))
    return S, Γ_S(1), Δz
end

# Discrete integrals and fluxes

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

# dE/dt = ∫ uᵢ Gᵢ dV + ∫ (π G_θ + m G_S) dV from the model's own tendencies, with two scales:
# the round-off scale ∫ |π G_θ| + |m G_S| dV (π ≈ cₚ makes this the largest term), and the size of the conversions ∫ |w b̃| + ε dV
function energy_tendency(model)
    update_state!(model)
    compute_tendencies!(model, [])
    G = model.timestepper.Gⁿ
    u, v, w = model.velocities
    π, m = field(thermodynamic_operation(model, exner)), field(thermodynamic_operation(model, chemical_potential_analogue))
    wb, ε = field(buoyancy_work(model)), field(viscous_dissipation(model))

    kinetic = integral(*, u, G.u) + integral(*, v, G.v) + integral(*, w, G.w)
    potential, round_off = integral(*, π, G.θ), integral((a, b) -> abs(a * b), π, G.θ)

    if haskey(G, :S)
        potential += integral(*, m, G.S)
        round_off += integral((a, b) -> abs(a * b), m, G.S)
    end

    conversions = integral(abs, wb) + integral(ε)
    return kinetic + potential, round_off, conversions
end
