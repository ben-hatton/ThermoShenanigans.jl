# A deterministic state, so that models on different architectures can be compared
function set_smooth_state!(model)
    U(x, y, z) = 1e-2 * sin(2π * x) * cos(π * y) * cos(π * z)
    θᵢ(x, y, z) = 283 + sin(2π * x) * sin(π * y) + z
    Sᵢ(x, y, z) = 0.035 + 1e-3 * cos(2π * x) * cos(π * z)
    set!(model, u = U, v = U, θ = θᵢ, S = Sᵢ)
    return nothing
end

# On the CPU in Float64 these have the types of the other test models, so time stepping compiles only once
architecture_grid(arch, FT) = RectilinearGrid(arch, FT, size = (4, 4, 4), extent = (1, 1, 1), topology = (Bounded, Bounded, Bounded))
architecture_thermodynamics(FT) = CabbelingBoussinesqThermodynamics(FT; cabbeling_coefficient = 1e-5, parameters...)

function architecture_model(arch, FT = Float64; step = true)
    model = component_model(architecture_grid(arch, FT), architecture_thermodynamics(FT))
    set_smooth_state!(model)
    step && time_step!(model, 1e-3)
    return model
end

#####
##### Every exported diagnostic, for the comparison of architectures
#####

# Grouped by their arguments: the model, the reference height z★, or both. Face fluxes return their (x, y, z) components.
const model_diagnostics = (viscous_dissipation, θ_production, buoyancy_production, heating_buoyancy_production,
                           nonlinear_buoyancy_production, entropy_production, potential_energy_conversion, buoyancy_work,
                           buoyancy_tendency, advective_buoyancy_tendency, kinetic_energy, ke_advection)
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
                             HeightIndependent(potential_enthalpy))

function_label(f) = string(nameof(f))
function_label(f::HeightIndependent) = string("HeightIndependent(", nameof(f.func), ")")

# `label => diagnostic` for every exported diagnostic of `model`, with one z★ shared by those of the reference state
function exported_diagnostics(model)
    z★ = reference_height(model)
    diagnostics = Pair{String, Any}["reference_height" => z★]

    for f in operation_functions
        push!(diagnostics, "thermodynamic_operation($(function_label(f)))" => thermodynamic_operation(model, f))
    end

    for diagnostic in model_diagnostics
        push!(diagnostics, function_label(diagnostic) => diagnostic(model))
    end

    for diagnostic in reference_state_diagnostics
        push!(diagnostics, function_label(diagnostic) => diagnostic(z★))
    end

    for diagnostic in reference_diagnostics
        push!(diagnostics, function_label(diagnostic) => diagnostic(model, z★))
    end

    for flux in model_fluxes, (component, operation) in pairs(flux(model))
        push!(diagnostics, "$(function_label(flux)).$component" => operation)
    end

    for flux in reference_fluxes, (component, operation) in pairs(flux(model, z★))
        push!(diagnostics, "$(function_label(flux)).$component" => operation)
    end

    return diagnostics
end

# The interior values, on the CPU, of a diagnostic: a Field, or an operation computed into one
diagnostic_values(field::Field) = Array(interior(compute!(field)))
diagnostic_values(operation) = diagnostic_values(Field(operation))

#=
Models on `arch` and on the CPU in the same random state, each after one time step. The smooth state is symmetric in
y, so that pairs of cells share a buoyancy to within round-off, and the round-off differences of the two
architectures could order them differently in the sorted reference state. No two cells of a random state are that
close in buoyancy, so the reference states of the two architectures agree.
=#
function comparison_models(arch, FT)
    cpu = component_model(architecture_grid(CPU(), FT), architecture_thermodynamics(FT))
    set_random_state!(cpu)
    values = map(field -> Array(interior(field)), merge(cpu.velocities, cpu.tracers))

    device = component_model(architecture_grid(arch, FT), architecture_thermodynamics(FT))
    set!(device; values...)

    for model in (device, cpu)
        time_step!(model, 1e-3)
    end

    return device, cpu
end

@testset "Every exported diagnostic is compared between architectures" begin
    # Exported names that are not diagnostics: types, thermodynamic functions f(θ, S, thermodynamics) and
    # f(θ, S, z, thermodynamics), and the constructor of the model components
    thermodynamics_type = typeof(cabbeling)
    is_thermodynamic_function(f) = hasmethod(f, Tuple{Float64, Float64, thermodynamics_type}) ||
                                   hasmethod(f, Tuple{Float64, Float64, Float64, thermodynamics_type})
    exported = filter(name -> name != :ThermoShenanigans, names(ThermoShenanigans))
    exported_values = map(name -> getproperty(ThermoShenanigans, name), exported)
    is_diagnostic = map(value -> !(value isa Type || value === thermodynamic_model_components ||
                                   is_thermodynamic_function(value)), exported_values)

    compared = (reference_height, thermodynamic_operation, model_diagnostics..., model_fluxes...,
                reference_state_diagnostics..., reference_diagnostics..., reference_fluxes...)
    @test Set(exported[is_diagnostic]) == Set(nameof.(compared))
end

# On GPUs, every exported diagnostic after a time step. On the CPU, the main diagnostics after a time step in Float64, and
# a few without a time step in Float32, which keeps compilation short; the energetics tests check every diagnostic on
# the CPU in Float64.
@testset "Architectures [$(summary(arch)), $FT]" for arch in archs, FT in (Float64, Float32)
    full = FT == Float64 || arch isa GPU
    model = architecture_model(arch, FT; step = full)
    @test model.closure[2] isa ThermodynamicDiffusivity{FT}

    if arch isa GPU
        # The values in Float64 are compared with those on the CPU in "GPU diagnostics match the CPU"
        @testset "$label" for (label, diagnostic) in exported_diagnostics(model)
            @test all(isfinite, diagnostic_values(diagnostic))
        end

        reference = architecture_model(CPU(), FT)
        for name in (:θ, :S)
            @test Array(interior(model.tracers[name])) ≈ interior(reference.tracers[name])
        end
    else
        thermodynamic_functions = full ? (potential_energy, temperature, exner, chemical_potential_analogue, chemical_potential) :
                                         (temperature,)
        for f in thermodynamic_functions
            @test all(isfinite, computed(thermodynamic_operation(model, f)))
        end
        productions = full ? (viscous_dissipation, θ_production, buoyancy_production, nonlinear_buoyancy_production,
                              entropy_production, potential_energy_conversion, buoyancy_work) : (θ_production, entropy_production)
        for diagnostic in productions
            @test all(isfinite, computed(diagnostic(model)))
        end
        for flux in (full ? (θ_flux, salt_flux, buoyancy_flux) : (θ_flux,))
            @test all(isfinite, computed(flux(model).z))
        end
    end
end

# The prognostic fields and every exported diagnostic of `device` against those of `cpu`, one testset per diagnostic
function test_architectures_match(device, cpu)
    device_fields, cpu_fields = merge(device.velocities, device.tracers), merge(cpu.velocities, cpu.tracers)
    @testset "$name" for name in keys(cpu_fields)
        @test Array(interior(device_fields[name])) ≈ interior(cpu_fields[name])
    end

    device_diagnostics, cpu_diagnostics = exported_diagnostics(device), exported_diagnostics(cpu)
    @test first.(device_diagnostics) == first.(cpu_diagnostics)
    cpu_values = Dict(label => diagnostic_values(diagnostic) for (label, diagnostic) in cpu_diagnostics)

    # 𝒩_ℬ is a small residual of terms the size of ∇·(u ℬ) ≈ cₚ ∇·(u θ), so that differences of a few ulps in the state
    # change it by about 10⁻⁷ of itself; its tolerance is set by the round-off of those terms instead
    root_sum_of_squares(values) = sqrt(sum(abs2, values))
    round_off_scales = Dict("spurious_bpe_production" => root_sum_of_squares(cpu_values["bpe_advection"]))

    @testset "$label" for (label, device_diagnostic) in device_diagnostics
        atol = sqrt(eps(Float64)) * get(round_off_scales, label, 0.0)
        @test diagnostic_values(device_diagnostic) ≈ cpu_values[label] atol = atol
    end
end

#=
Float64 only: in Float32, π ≈ cₚ = 4000 J kg⁻¹ K⁻¹ is resolved to 2.4 × 10⁻⁴ J kg⁻¹ K⁻¹, while π differs between
neighbouring cells of `architecture_model` by about 5 × 10⁻⁴ J kg⁻¹ K⁻¹ vertically and 1.5 × 10⁻⁵ J kg⁻¹ K⁻¹
horizontally, so that the diagnostics formed from these differences (such as σ_θ and j_θ·∇π) are dominated by
round-off, which differs between architectures. The Float32 diagnostics on the GPU are checked to be finite above.
=#
if test_gpu
    @testset "GPU diagnostics match the CPU [Float64]" begin
        test_architectures_match(comparison_models(GPU(), Float64)...)
    end
end
