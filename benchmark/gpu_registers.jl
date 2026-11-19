# Registers per thread and theoretical occupancy of the GPU kernels that decide whether closure fields for the
# cell-centre thermodynamic coefficients (cₚ/π, T_S/T_θ, Γ_S) are worth prototyping (performance/log.md, Decisions):
# the θ tendency kernel of a consistent and of a Fickian model, and the two kernels of a reference tendency.
#
#     julia --project=benchmark benchmark/gpu_registers.jl
#
# Each kernel is compiled for the device with the arguments and the launch configuration that Oceananigans' `launch!`
# gives it (Oceananigans 0.116.1, src/Utils/kernel_launching.jl:348-366), and its compiled function is queried with
# `CUDA.registers` and `CUDA.occupancy`, as CUDA.jl's KernelAbstractions backend compiles it before a launch
# (src/CUDAKernels.jl, `(obj::KA.Kernel{CUDABackend})(args...)`). A θ tendency kernel whose consistent version has
# markedly more registers and a lower occupancy than its Fickian version is register-bound.

using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation, _compute!
using Oceananigans.Architectures: KA, convert_to_device
using Oceananigans.Models: interior_tendency_kernel_parameters
using Oceananigans.Models.NonhydrostaticModels: compute_Gc!
using Oceananigans.Utils: configure_kernel
using ThermoShenanigans
using ThermoShenanigans: total_buoyancy_tendencyᶜᶜᶜ, velocities_and_tracers, production_parameters, wall_conditions,
                         _compute_parcel_reference_tendency!
using CUDA
using Printf

N = 128

thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                   cabbeling_coefficient = 1e-5, reference_temperature = 283,
                                                   reference_salinity = 0.035, reference_heat_capacity = 4000)

function gpu_model(; consistent)
    grid = RectilinearGrid(GPU(); size = (N, N, N), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Periodic, Periodic, Bounded))
    components = thermodynamic_model_components(grid, thermodynamics; consistent, viscosity = 1e-6,
                                                thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7)
    return NonhydrostaticModel(grid; tracers = (:θ, :S), components...)
end

# The compiled `HostKernel` of `kernel!` launched by Oceananigans' `launch!(arch, grid, workspec, kernel!, args...)`,
# and its number of threads per block
function compiled_kernel(grid, workspec, kernel!, args...)
    arch = architecture(grid)
    workspec = workspec isa Symbol ? Val(workspec) : workspec
    location = Oceananigans.instantiated_location(first(args))
    loop!, _ = configure_kernel(arch, grid, workspec, kernel!, Val(false); location)
    ndrange, _, iterspace, _ = KA.launch_config(loop!, nothing, nothing)
    context = KA.mkcontext(loop!, ndrange, iterspace)
    threads = length(KA.workitems(iterspace))
    kernel = @cuda launch=false always_inline=KA.backend(loop!).always_inline maxthreads=threads loop!.f(context,
                                                                                     convert_to_device(arch, args)...)
    return kernel, threads
end

# The θ tendency kernel, with the arguments of Oceananigans' `launch_tracer_tendency!`
# (src/Models/NonhydrostaticModels/compute_nonhydrostatic_tendencies.jl:91-105)
function θ_tendency_kernel(model)
    grid = model.grid
    return compiled_kernel(grid, interior_tendency_kernel_parameters(architecture(grid), grid), compute_Gc!,
                           model.timestepper.Gⁿ.θ, grid, Val(1), Val(:θ), model.advection.θ, model.closure,
                           model.tracers.θ.boundary_conditions.immersed, model.buoyancy, model.biogeochemistry,
                           model.background_fields, model.velocities, model.tracers, model.auxiliary_fields,
                           model.closure_fields, model.clock, model.forcing.θ)
end

# The two kernels of `compute!` of R★ (`reference_state.jl` and `sorting.jl`): the full buoyancy tendency Ḃ through
# Oceananigans' `_compute!`, and Ṗ★(z★) from the bands and Ċ
function reference_tendency_kernels(model)
    grid = model.grid
    z★ = reference_height(model)
    workspace = z★.operand.workspace
    operation = KernelFunctionOperation{Center, Center, Center}(total_buoyancy_tendencyᶜᶜᶜ, grid, model.advection,
                                                                model.clock, velocities_and_tracers(model),
                                                                production_parameters(model), wall_conditions(model))
    parcel_tendency = Field{Center, Center, Center}(grid)
    return ("Ḃ (_compute!)" => compiled_kernel(grid, :xyz, _compute!, z★.operand.buoyancy_tendency, operation),
            "Ṗ★(z★)" => compiled_kernel(grid, :xyz, _compute_parcel_reference_tendency!, parcel_tendency.data,
                                        workspace.band_first, workspace.band_last, workspace.cumulative_tendency,
                                        workspace.slot_thickness))
end

consistent, fickian = gpu_model(consistent = true), gpu_model(consistent = false)

kernels = ("θ tendency, consistent" => θ_tendency_kernel(consistent),
           "θ tendency, Fickian" => θ_tendency_kernel(fickian),
           map(((name, kernel),) -> "R★: " * name => kernel, reference_tendency_kernels(consistent))...)

println("Registers and occupancy on ", CUDA.name(CUDA.device()), ", ", N, "³, Float64")
@printf("%-28s %10s %10s %10s %12s\n", "kernel", "registers", "threads", "occupancy", "local bytes")
for (name, (kernel, threads)) in kernels
    @printf("%-28s %10d %10d %9.0f%% %12d\n", name, CUDA.registers(kernel), threads,
            100 * CUDA.occupancy(kernel.fun, threads), CUDA.memory(kernel).local)
end
