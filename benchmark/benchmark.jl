# Benchmark of a thermodynamically consistent model: one time step, and one `compute!` of the reference height z★,
# the reference term R★ of the APE budget, the θ production σ_θ and the spurious APE production 𝒩, on 64³ and 128³ grids;
# and one time step of the Fickian model of `thermodynamic_model_components(...; consistent = false)`, for the ratio of
# the two time steps.
#
#     julia --project=benchmark benchmark/benchmark.jl [CPU | GPU] [trials]
#
# The architecture defaults to CPU and the number of trials to 10. The first call of each operation, which compiles it
# on the first grid, is timed on its own, then `trials` later calls, of which the minimum and median are reported; on
# the GPU each call is synchronised with the device before its time is taken. On the CPU, set the number of threads
# with `julia --threads`.
#
# R★ and 𝒩 depend on the reference state, so computing either one computes z★ again: their times include that of z★.
# R★ also computes the buoyancy tendency of every cell and sorts it with the buoyancy.

using Oceananigans
using Oceananigans.Architectures: synchronize
using ThermoShenanigans
using Printf
using Statistics: median

architecture_name = uppercase(get(ARGS, 1, "CPU"))
architecture_name in ("CPU", "GPU") ||
    throw(ArgumentError("The architecture must be CPU or GPU, but got $(ARGS[1])."))

trials = parse(Int, get(ARGS, 2, "10"))

if architecture_name == "GPU"
    using CUDA
end

arch = architecture_name == "GPU" ? GPU() : CPU()

sizes = (64, 128)
Δt = 0.1

thermodynamics = CabbelingBoussinesqThermodynamics(thermal_expansion = 2e-4, haline_contraction = 0.8,
                                                   cabbeling_coefficient = 1e-5, reference_temperature = 283,
                                                   reference_salinity = 0.035, reference_heat_capacity = 4000)

# A stably stratified cube with small random perturbations, so that no two cells share a buoyancy
function benchmark_model(arch, N; consistent = true)
    grid = RectilinearGrid(arch; size = (N, N, N), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Periodic, Periodic, Bounded))
    components = thermodynamic_model_components(grid, thermodynamics; consistent, viscosity = 1e-6,
                                                thermal_diffusivity = 1e-7, salt_diffusivity = 1e-7)
    model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

    U(x, y, z) = 1e-3 * randn()
    θᵢ(x, y, z) = 283 + z + 1e-3 * rand()
    Sᵢ(x, y, z) = 0.035 + 1e-5 * rand()
    set!(model, u = U, v = U, w = U, θ = θᵢ, S = Sᵢ)

    return model
end

# Wall-clock time in seconds of one call of `operation!`
function wall_time(operation!, arch)
    start = time_ns()
    operation!()
    synchronize(arch)
    return (time_ns() - start) / 1e9
end

# Wall-clock times in seconds of the first call of `operation!` and of `trials` later calls
wall_times(operation!, arch, trials) = wall_time(operation!, arch), map(_ -> wall_time(operation!, arch), 1:trials)

function print_header(arch)
    device = arch isa GPU ? CUDA.name(CUDA.device()) : string(Sys.cpu_info()[1].model, ", ", Threads.nthreads(), " threads")
    println("ThermoShenanigans benchmark on ", summary(arch), " (", device, "), Julia ", VERSION, ", Float64, ",
            "first call and ", trials, " later calls")
    @printf("%-6s %-26s %12s %12s %12s\n", "grid", "operation", "first call", "minimum", "median")
end

prettyseconds(t) = t < 1e-3 ? @sprintf("%8.1f μs", 1e6 * t) : t < 1 ? @sprintf("%8.2f ms", 1e3 * t) : @sprintf("%8.3f s", t)

print_header(arch)

for N in sizes
    model = benchmark_model(arch, N)
    fickian = benchmark_model(arch, N; consistent = false)
    z★ = reference_height(model)
    R★ = Field(reference_tendency_work(model, z★))
    σ_θ = Field(θ_production(model))
    𝒩 = Field(spurious_ape_production(model, z★))

    operations = ("time_step!"              => () -> time_step!(model, Δt),
                  "time_step!, Fickian"     => () -> time_step!(fickian, Δt),
                  "reference_height"        => () -> compute!(z★),
                  "reference_tendency_work" => () -> compute!(R★),
                  "θ_production"            => () -> compute!(σ_θ),
                  "spurious_ape_production" => () -> compute!(𝒩))

    for (name, operation!) in operations
        first_time, times = wall_times(operation!, arch, trials)
        @printf("%-6s %-26s %12s %12s %12s\n", "$(N)³", name, prettyseconds(first_time), prettyseconds(minimum(times)),
                prettyseconds(median(times)))
    end
end
