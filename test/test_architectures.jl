# A deterministic state, so that models on different architectures can be compared
function set_smooth_state!(model)
    U(x, y, z) = 1e-2 * sin(2π * x) * cos(π * y) * cos(π * z)
    θᵢ(x, y, z) = 283 + sin(2π * x) * sin(π * y) + z
    Sᵢ(x, y, z) = 0.035 + 1e-3 * cos(2π * x) * cos(π * z)
    set!(model, u = U, v = U, θ = θᵢ, S = Sᵢ)
    return nothing
end

function architecture_model(arch, FT = Float64; step = true)
    model = component_model(architecture_grid(arch, FT), architecture_thermodynamics(FT))
    set_smooth_state!(model)
    step && time_step!(model, 1e-3)
    return model
end

# The interior values, on the CPU, of a diagnostic: a Field, or an operation computed into one
diagnostic_values(field::Field) = Array(interior(compute!(field)))
diagnostic_values(operation) = diagnostic_values(Field(operation))

#=
Models built by `build(grid, thermodynamics; model_kwargs...)` on `arch` and on the CPU in the same random state, each
after one time step. The smooth state is symmetric in y, so that pairs of cells share a buoyancy to within round-off,
and the round-off differences of the two architectures could order them differently in the sorted reference state. No
two cells of a random state are that close in buoyancy, so the reference states of the two architectures agree.
=#
function comparison_models(arch, FT; build = component_model, model_kwargs...)
    cpu = build(architecture_grid(CPU(), FT), architecture_thermodynamics(FT); model_kwargs...)
    set_random_state!(cpu)
    values = map(field -> Array(interior(field)), merge(cpu.velocities, cpu.tracers))

    device = build(architecture_grid(arch, FT), architecture_thermodynamics(FT); model_kwargs...)
    set!(device; values...)

    for model in (device, cpu)
        time_step!(model, 1e-3)
    end

    return device, cpu
end

# The prognostic fields and every exported diagnostic of `device` against those of `cpu`, one testset per diagnostic,
# but those labelled in `refused`
function test_architectures_match(device, cpu; refused = ())
    device_fields, cpu_fields = merge(device.velocities, device.tracers), merge(cpu.velocities, cpu.tracers)
    @testset "$name" for name in keys(cpu_fields)
        @test Array(interior(device_fields[name])) ≈ interior(cpu_fields[name])
    end

    device_diagnostics, cpu_diagnostics = exported_diagnostics(device; refused), exported_diagnostics(cpu; refused)
    @test first.(device_diagnostics) == first.(cpu_diagnostics)
    cpu_values = Dict(label => diagnostic_values(diagnostic) for (label, diagnostic) in cpu_diagnostics)

    # 𝒩_ℬ is a small residual of terms the size of ∇·(u ℬ) ≈ cₚ ∇·(u θ), so that differences of a few ulps in the state
    # change it by about 10⁻⁷ of itself; its tolerance is set by the round-off of those terms instead. Both are absent
    # for a Fickian model.
    root_sum_of_squares(values) = sqrt(sum(abs2, values))
    round_off_scales = Dict("spurious_bpe_production" => root_sum_of_squares(get(cpu_values, "bpe_advection", Float64[])))

    @testset "$label" for (label, device_diagnostic) in device_diagnostics
        atol = sqrt(eps(Float64)) * get(round_off_scales, label, 0.0)
        @test diagnostic_values(device_diagnostic) ≈ cpu_values[label] atol = atol
    end
end

function test_architectures()
    @testset "Every exported diagnostic is compared between architectures" begin
        # Exported names that are not diagnostics: types, thermodynamic functions f(θ, S, z, thermodynamics), and the
        # constructors of the model components and of the θ forcing σ_θ
        thermodynamics_type = typeof(cabbeling)
        is_thermodynamic_function(f) = hasmethod(f, Tuple{Float64, Float64, Float64, thermodynamics_type})
        exported = filter(name -> name != :ThermoShenanigans, names(ThermoShenanigans))
        exported_values = map(name -> getproperty(ThermoShenanigans, name), exported)
        is_diagnostic = map(value -> !(value isa Type || value in (thermodynamic_model_components, θ_production_forcing) ||
                                       is_thermodynamic_function(value)), exported_values)

        compared = (reference_height, thermodynamic_operation, model_diagnostics..., model_fluxes...,
                    reference_state_diagnostics..., reference_diagnostics..., reference_fluxes...)
        @test Set(exported[is_diagnostic]) == Set(nameof.(compared))
    end

    #=
    The status of every exported diagnostic for a Fickian (`consistent = false`) model: supported, supported and
    identically zero, or refused with a `MethodError`. This testset defines the list in the docstring of
    `thermodynamic_model_components`.
    =#
    @testset "Fickian status of every exported diagnostic" begin
        consistent = architecture_model(CPU())
        fickian = fickian_model(architecture_grid(CPU(), Float64), architecture_thermodynamics(Float64))
        set_smooth_state!(fickian)
        time_step!(fickian, 1e-3)

        # The label of a diagnostic, or of the face flux of which it is a component
        flux_label(label) = first(split(label, '.'))
        consistent_labels = first.(exported_diagnostics(consistent))
        supported = exported_diagnostics(fickian; refused = fickian_refused)
        @test issubset(fickian_refused, flux_label.(consistent_labels))
        @test first.(supported) == filter(label -> !(flux_label(label) in fickian_refused), consistent_labels)

        values = Dict(label => diagnostic_values(diagnostic) for (label, diagnostic) in supported)
        @testset "$label" for (label, _) in supported
            @test all(isfinite, values[label])
        end

        zero_labels = filter(label -> all(iszero, values[label]), first.(supported))
        @test zero_labels == ["θ_production", "heating_buoyancy_production", "heating_ape_production"]
        @info "Fickian diagnostics: supported $(filter(label -> !(label in zero_labels), first.(supported))); " *
              "identically zero $zero_labels; refused $(collect(fickian_refused))"
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

    #=
    Float64 only: in Float32, π ≈ cₚ = 4000 J kg⁻¹ K⁻¹ is resolved to 2.4 × 10⁻⁴ J kg⁻¹ K⁻¹, while π differs between
    neighbouring cells of `architecture_model` by about 5 × 10⁻⁴ J kg⁻¹ K⁻¹ vertically and 1.5 × 10⁻⁵ J kg⁻¹ K⁻¹
    horizontally, so that the diagnostics formed from these differences (such as σ_θ and j_θ·∇π) are dominated by
    round-off, which differs between architectures. The Float32 diagnostics on the GPU are checked to be finite above.
    =#
    if test_gpu
        #=
        With Flux conditions (`flux_bc_model`, tracers θ, S and c): a time step with the conditions on the device, the
        zero flux of the passive tracer c (`tracer_flux_*(…, ::Val, …)`), and the wall fluxes of the diagnostics, from
        `wall_conditions(model)` adapted to the device: `lateral_boundary_flux` (`getbc`) on the x walls,
        `vertical_boundary_flux` (`explicit_flux`) on the bottom and top, `boundary_condition_flux_*`,
        `boundary_cell_value_*`, and the wall values of j_θ, j_S, j_b, j_𝒜, j_ℬ, Ḃ and R★
        =#
        @testset "GPU diagnostics match the CPU [$label, Float64]" for
                (label, build) in (("no flux", component_model), ("Flux conditions", flux_bc_model))
            test_architectures_match(comparison_models(GPU(), Float64; build)...)
        end

        #=
        The Fickian fluxes of Oceananigans' `diffusive_flux_*` through the wall masks, the zero σ_θ, and every
        diagnostic that a Fickian model supports. With wall conditions (`fickian_wall_model`), the closure's own wall
        flux of a Value and a Gradient condition (`closure_wall_flux_*`, through `x/y/z_peripheral_node`), added to the
        wall fluxes of Flux conditions on the x walls
        =#
        @testset "GPU Fickian diagnostics match the CPU [$label, Float64]" for
                (label, build) in (("no flux", fickian_model), ("wall conditions", fickian_wall_model))
            test_architectures_match(comparison_models(GPU(), Float64; build)...; refused = fickian_refused)
        end

        # The θ forcing as `MultipleForcings` of a user forcing and σ_θ
        @testset "GPU time step with a user θ forcing matches the CPU [Float64]" begin
            heating = θ_forcing_by_heating(architecture_grid(CPU(), Float64), architecture_thermodynamics(Float64))
            device, cpu = comparison_models(GPU(), Float64; forcing = (; θ = heating))
            device_fields, cpu_fields = merge(device.velocities, device.tracers), merge(cpu.velocities, cpu.tracers)
            @testset "$name" for name in keys(cpu_fields)
                @test Array(interior(device_fields[name])) ≈ interior(cpu_fields[name])
            end
        end
    end
end
