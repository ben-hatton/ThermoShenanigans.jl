#=
Shared setup of the test files. Each test file `test_name.jl` runs on its own after this file,

    include("test/utils_for_runtests.jl")
    include("test/test_energetics.jl")
    run_test_file("test_energetics.jl")  # or test_energetics()

and test/runtests.jl runs the files of the selected groups (see test/test_groups.jl) in this way.
=#

using Test
using Oceananigans
using ThermoShenanigans
using ForwardDiff: derivative
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: location
using Oceananigans.Grids: znode
using Oceananigans.Operators: volume
using Oceananigans.TimeSteppers: update_state!, compute_tendencies!, compute_flux_bc_tendencies!
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, diffusive_flux_x, diffusive_flux_y, diffusive_flux_z
using ThermoShenanigans: barodiffusion_coefficient, θ_flux_prefactor, thermodynamic_functionᶜᶜᶜ

# Each test file `test_name.jl` defines a function `test_name()` that runs its tests, compiled once per session. The call
# goes through `invokelatest` since the function is defined after the caller was compiled.
run_test_file(file) = Base.invokelatest(getproperty(@__MODULE__, Symbol(splitext(basename(file))[1])))

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
    @eval $f(θ, S, z, ::CoupledTestThermodynamics) = zero(θ)
end

buoyancy(θ, S, z, thermodynamics::CoupledTestThermodynamics)           = thermodynamics.gα * θ
∂b∂θ(θ, S, z, thermodynamics::CoupledTestThermodynamics)               = thermodynamics.gα
potential_enthalpy(θ, S, z, thermodynamics::CoupledTestThermodynamics) = thermodynamics.c * θ + thermodynamics.a * θ * S
∂h⁰∂θ(θ, S, z, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.c + thermodynamics.a * S
∂h⁰∂S(θ, S, z, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.a * θ
∂²h⁰∂θ∂S(θ, S, z, thermodynamics::CoupledTestThermodynamics)           = thermodynamics.a
entropy(θ, S, z, thermodynamics::CoupledTestThermodynamics)            = thermodynamics.c * log(θ) - S * (log(S) - 1)
∂η⁰∂θ(θ, S, z, thermodynamics::CoupledTestThermodynamics)              = thermodynamics.c / θ
∂η⁰∂S(θ, S, z, thermodynamics::CoupledTestThermodynamics)              = - log(S)
∂²η⁰∂θ²(θ, S, z, thermodynamics::CoupledTestThermodynamics)            = - thermodynamics.c / θ^2
∂²η⁰∂S²(θ, S, z, thermodynamics::CoupledTestThermodynamics)            = - 1 / S

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
function isentropic_θ(η₀, S, z, thermodynamics; θ = 283 + zero(η₀ * S))
    for _ in 1:30
        θ -= (entropy(θ, S, z, thermodynamics) - η₀) / ∂η⁰∂θ(θ, S, z, thermodynamics)
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

# On the CPU in Float64 these have the types of the other test models, so time stepping compiles only once
architecture_grid(arch, FT) = RectilinearGrid(arch, FT, size = (4, 4, 4), extent = (1, 1, 1), topology = (Bounded, Bounded, Bounded))
architecture_thermodynamics(FT) = CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...)

# The Fickian ScalarDiffusivity(ν, κ) of consistent = false is the closure under test: a model type of its own
fickian_model(grid, thermodynamics; kwargs...) = component_model(grid, thermodynamics; consistent = false, kwargs...)

# The constant-salinity Fickian model, with κ = (; θ = κ_θ) and no S: a model type of its own, for the zero salt flux
fickian_constant_salinity_model(grid) = fickian_model(grid, cabbeling; tracers = (:θ,), constant_salinity = 0.035)

# Labels, as in `exported_diagnostics`, of the diagnostics that a Fickian model refuses
const fickian_refused = ("entropy_production", "potential_energy_conversion", "potential_energy", "internal_energy",
                         "thermodynamic_operation(potential_energy)", "background_potential_energy", "bpe_flux",
                         "bpe_advection", "spurious_bpe_production")

# `build()` throws the `MethodError` of the diagnostic `f` on the Fickian `ScalarDiffusivity` of its model
function test_refused(f, build)
    exception = try
        build()
        nothing
    catch caught
        caught
    end
    @test exception isa MethodError && exception.f === f && exception.args[1] isa ScalarDiffusivity
end

# Prescribed fluxes of θ and S at the x walls and at the top and bottom (positive in +x and +z), and a passive tracer c
const flux_bc_values = (θ = (west = 2e-4, east = -1e-4, top = 1e-4, bottom = -2e-4),
                        S = (west = -2e-6, east = 1e-6, top = -3e-6, bottom = 1e-6))

# The flux values are converted to the float type of the grid, for the Float32 models of the type-stability tests
flux_bc_model(grid, thermodynamics = cabbeling) =
    component_model(grid, thermodynamics; tracers = (:θ, :S, :c),
                    boundary_conditions = map(Q -> FieldBoundaryConditions(; map(value -> FluxBoundaryCondition(convert(eltype(grid), value)), Q)...),
                                              flux_bc_values))

#=
Σ_walls A (c₁ Q_θ + d₁ Q_S) over the x walls and the bottom and top of `grid`, with inward normals: the input through
the walls by the fluxes Q of `flux_bc_values` (positive in +x and +z), with the values of the interior arrays c and d
in the boundary cells
=#
function wall_input(c, d, grid; Q = flux_bc_values)
    Ax = yspacings(grid, Center())[1] * zspacings(grid, Center())[1]
    Az = xspacings(grid, Center())[1] * yspacings(grid, Center())[1]
    side_input(dims, index, Q_θ, Q_S) = sum(selectdim(c, dims, index) .* Q_θ .+ selectdim(d, dims, index) .* Q_S)
    return Ax * (side_input(1, 1, Q.θ.west, Q.S.west) - side_input(1, size(c, 1), Q.θ.east, Q.S.east)) +
           Az * (side_input(3, 1, Q.θ.bottom, Q.S.bottom) - side_input(3, size(c, 3), Q.θ.top, Q.S.top))
end

#=
A θ flux Q_θ = Q_H f(t) / (ρ₀ π(θ₁, S₁, z₁)) at the bottom (k₁ = 1) or top (k₁ = Nz), which carries the heat flux
Q_H f(t), positive upwards, with π of the boundary cell and f(t) = 1 + t: a discrete-form flux condition that reads the
model's fields and clock. With the same Q_H at the bottom and top, the energy input Σ_walls A π₁ Q_θ is zero.
=#
@inline function equal_heat_θ_flux(i, j, grid, clock, fields, parameters)
    π₁ = thermodynamic_functionᶜᶜᶜ(i, j, parameters.k, grid, exner, parameters.buoyancy_formulation, fields)
    return parameters.heat_flux * (1 + clock.time) / (parameters.reference_density * π₁)
end

# A model type of its own, for its field- and time-dependent discrete-form flux conditions of θ
function equal_heat_flux_model(grid; heat_flux = 400, reference_density = 1000)
    buoyancy_formulation = ThermodynamicBuoyancy(grid, cabbeling)
    θ_flux_condition(k) = FluxBoundaryCondition(equal_heat_θ_flux; discrete_form = true,
                                                parameters = (; heat_flux, reference_density, buoyancy_formulation, k))
    boundary_conditions = (; θ = FieldBoundaryConditions(bottom = θ_flux_condition(1), top = θ_flux_condition(size(grid, 3))))
    return component_model(grid, cabbeling; boundary_conditions)
end

#=
A Fickian model with the Flux conditions of `flux_bc_values` on the x walls, a ValueBoundaryCondition on the top of θ
and a GradientBoundaryCondition on the bottom of S: a model type of its own, since only a Fickian model accepts Value
and Gradient conditions. Its closure carries the wall fluxes of the Value and Gradient conditions, and the model adds
those of the Flux conditions separately.
=#
fickian_wall_model(grid, thermodynamics = cabbeling) =
    fickian_model(grid, thermodynamics;
                  boundary_conditions = (θ = FieldBoundaryConditions(west = FluxBoundaryCondition(flux_bc_values.θ.west),
                                                                     east = FluxBoundaryCondition(flux_bc_values.θ.east),
                                                                     top = ValueBoundaryCondition(283.5)),
                                         S = FieldBoundaryConditions(west = FluxBoundaryCondition(flux_bc_values.S.west),
                                                                     east = FluxBoundaryCondition(flux_bc_values.S.east),
                                                                     bottom = GradientBoundaryCondition(-1e-3))))

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

#=
A user θ forcing F_θ = q / π, the θ forcing of a heating q = π F_θ = q₁ sin(2π z / Lz), on a grid with z ∈ (-Lz, 0).
A discrete-form forcing that reads the model's tracers, with π evaluated as in `thermodynamic_operation(model, exner)`.
=#
@inline function θ_forcing_by_heatingᶜᶜᶜ(i, j, k, grid, clock, fields, parameters)
    z = znode(i, j, k, grid, Center(), Center(), Center())
    q = parameters.heating_amplitude * sinpi(2 * z / grid.Lz)
    return q / thermodynamic_functionᶜᶜᶜ(i, j, k, grid, exner, parameters.buoyancy_formulation, fields)
end

function θ_forcing_by_heating(grid, thermodynamics; heating_amplitude = 100)
    parameters = (; heating_amplitude = convert(eltype(grid), heating_amplitude),
                    buoyancy_formulation = ThermodynamicBuoyancy(grid, thermodynamics))
    return Forcing(θ_forcing_by_heatingᶜᶜᶜ; discrete_form = true, parameters)
end

# The θ tendency G_θ of `model`, with all of its forcings
function θ_tendency(model)
    update_state!(model)
    compute_tendencies!(model, [])
    return Array(interior(model.timestepper.Gⁿ.θ))
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

# dE/dt = ∫ uᵢ Gᵢ dV + ∫ (π G_θ + m G_S) dV from the model's own tendencies, including those of its
# FluxBoundaryConditions, with two scales: the round-off scale ∫ |π G_θ| + |m G_S| dV (π ≈ cₚ makes this the largest
# term), and the size of the conversions ∫ |w b̃| + ε dV
function energy_tendency(model)
    update_state!(model)
    compute_tendencies!(model, [])
    compute_flux_bc_tendencies!(model)
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

# The indices of the cell (3, 3, 3) of `operation`, at index 1 in its reduced (`Nothing`) directions
cell_indices(operation) = map(L -> L === Nothing ? 1 : 3, location(operation))

# Bytes allocated by evaluating `operation` in one cell, behind a function barrier so that the result is not boxed
function cell_allocations(operation)
    i, j, k = cell_indices(operation)
    operation[i, j, k]
    return @allocated operation[i, j, k]
end

# The interior of a data array of the reference state at (LX, LY, LZ)
interior_data(data, LX, LY, LZ, grid) = Array(interior(Field{LX, LY, LZ}(grid; data)))

# ∇·j of face fluxes j = (x, y, z) on a regular, Bounded grid, from the interiors including the wall faces
function divergence(j, grid)
    Δx, Δy, Δz = xspacings(grid, Center())[1], yspacings(grid, Center())[1], zspacings(grid, Center())[1]
    jx, jy, jz = map(computed, (j.x, j.y, j.z))
    return diff(jx, dims = 1) ./ Δx .+ diff(jy, dims = 2) ./ Δy .+ diff(jz, dims = 3) ./ Δz
end

# ℑ to the cell centres of values on the faces of direction `dims`, including the walls
center_average(a, dims) = (selectdim(a, dims, 1:size(a, dims)-1) .+ selectdim(a, dims, 2:size(a, dims))) ./ 2

# Whether the components of the face flux j normal to the walls vanish there
function zero_at_walls(j)
    jx, jy, jz = map(computed, (j.x, j.y, j.z))
    return all(iszero, jx[[1, end], :, :]) && all(iszero, jy[:, [1, end], :]) && all(iszero, jz[:, :, [1, end]])
end

# Every exported diagnostic

# Grouped by their arguments: the model, the reference height z★, or both. Face fluxes return their (x, y, z) components.
const model_diagnostics = (potential_energy, internal_energy, gravitational_energy, viscous_dissipation, θ_production,
                           buoyancy_production, heating_buoyancy_production, nonlinear_buoyancy_production,
                           entropy_production, potential_energy_conversion, buoyancy_work, buoyancy_tendency,
                           advective_buoyancy_tendency, kinetic_energy, ke_advection)
const model_fluxes = (θ_flux, salt_flux, buoyancy_flux)
const reference_state_diagnostics = (reference_displacement, reference_potential, reference_buoyancy, reference_work,
                                     available_potential_energy)
const reference_diagnostics = (background_potential_energy, background_gravitational_energy, ape_production,
                               heating_ape_production, nonlinear_ape_production, ape_dissipation, ape_advection,
                               available_buoyancy_work, spurious_ape_production, reference_tendency_work, bpe_advection,
                               spurious_bpe_production)
const reference_fluxes = (ape_flux, ke_flux, bpe_flux)

# The functions named in the docstring of `thermodynamic_operation`, and the internal energy ℐ = h⁰
const operation_functions = (potential_energy, temperature, exner, chemical_potential_analogue, chemical_potential,
                             potential_enthalpy)

function_label(f) = string(nameof(f))

#=
`label => diagnostic` for every exported diagnostic of `model` but those labelled in `refused`, with one z★ shared by
those of the reference state. Thermodynamic operations are labelled `"thermodynamic_operation(name)"`, and the
components of a face flux `"name.x"`, `"name.y"` and `"name.z"`, skipped together by the name of the flux.
=#
function exported_diagnostics(model; refused = ())
    z★ = reference_height(model)
    diagnostics = Pair{String, Any}["reference_height" => z★]
    is_kept(label) = !(label in refused)

    for f in operation_functions
        label = "thermodynamic_operation($(function_label(f)))"
        is_kept(label) && push!(diagnostics, label => thermodynamic_operation(model, f))
    end

    for diagnostic in model_diagnostics
        is_kept(function_label(diagnostic)) && push!(diagnostics, function_label(diagnostic) => diagnostic(model))
    end

    for diagnostic in reference_state_diagnostics
        is_kept(function_label(diagnostic)) && push!(diagnostics, function_label(diagnostic) => diagnostic(z★))
    end

    for diagnostic in reference_diagnostics
        is_kept(function_label(diagnostic)) && push!(diagnostics, function_label(diagnostic) => diagnostic(model, z★))
    end

    for flux in model_fluxes
        is_kept(function_label(flux)) || continue
        for (component, operation) in pairs(flux(model))
            push!(diagnostics, "$(function_label(flux)).$component" => operation)
        end
    end

    for flux in reference_fluxes
        is_kept(function_label(flux)) || continue
        for (component, operation) in pairs(flux(model, z★))
            push!(diagnostics, "$(function_label(flux)).$component" => operation)
        end
    end

    return diagnostics
end
