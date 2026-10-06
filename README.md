# ThermoShenanigans.jl

Thermodynamically consistent molecular fluxes and irreversible production terms for the Boussinesq
`NonhydrostaticModel` of [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl), following
Tailleux, Dubos & Hatton ([arXiv:2512.18461](https://arxiv.org/abs/2512.18461)).

Standard ocean models diffuse temperature and salinity down their own gradients (Fick's law), drop the heating
caused by viscous dissipation, and ignore pressure effects on diffusion. ThermoShenanigans replaces this with
heat and salt fluxes derived from a full thermodynamic description of the fluid, so that

* total energy (kinetic + potential) is conserved exactly, including in the discrete equations, and
* entropy production is non-negative, as the second law requires.

Choose the thermodynamics by supplying the buoyancy $b(\theta, S)$, potential enthalpy $h^0(\theta, S)$ and
entropy $\eta^0(\theta, S)$. Everything else (temperature, heat capacity, chemical potential, fluxes, production
terms) is derived from them.

## Quick start

```julia
using Pkg
Pkg.add(url = "https://github.com/ben-hatton/ThermoShenanigans.jl")
```

```julia
using Oceananigans
using ThermoShenanigans

grid = RectilinearGrid(size = (64, 64), x = (0, 0.1), z = (-0.1, 0), topology = (Periodic, Flat, Bounded))

thermo = CabbelingBoussinesqThermodynamics(
   thermal_expansion = 2e-4,
   haline_contraction = 0.8,
   cabbeling_coefficient = 1e-5,
   reference_temperature = 283.15,
   reference_salinity = 0.035,
   reference_heat_capacity = 3991.87
   )

components = thermodynamic_model_components(grid, thermo;
   viscosity = 1e-6,
   thermal_diffusivity = 1.4e-7,
   salt_diffusivity = 1e-9
   )
   
model = NonhydrostaticModel(grid; tracers = (:θ, :S), components...)

set!(model, θ = (x, z) -> 283.15 + 2 * (z > -0.05), S = 0.035)

simulation = Simulation(model; Δt = 0.1, stop_time = 60)
run!(simulation)

diagnostics = thermodynamic_diagnostics(model)
σ_η = Field(diagnostics.σ_η) # entropy production diagnostic
```

`thermodynamic_model_components` returns the `buoyancy`, `closure` and `forcing` keyword arguments of
`NonhydrostaticModel`, so the result is an ordinary Oceananigans model.

## Making the Boussinesq approximation

Following [Tailleux & Dubos (2024)](https://doi.org/10.1016/j.ocemod.2024.102339), the dynamics and thermodynamics
of a compressible fluid can be written in terms of a single potential, the static energy
$\Sigma(\eta, S, p, \Phi) = h(\eta, S, p) + \Phi$, where $h$ is the specific enthalpy and $\Phi = g z$ the
geopotential. Approximating $\Sigma$, rather than individual terms in the equations, keeps the energetics and
thermodynamics consistent. The choice

$$
\Sigma \approx h^0(\theta, S) - b(\theta, S)\, z
$$

yields the simple Boussinesq approximation implemented here.

## Thermodynamics

The thermodynamics is defined by three functions of $\theta$ and $S$:

| Symbol | Name |
|---|---|
| $b(\theta, S)$ | buoyancy |
| $h^0(\theta, S)$ | potential enthalpy |
| $\eta^0(\theta, S)$ | specific entropy |

$\theta$ is a generic entropic variable, such as potential temperature or Conservative Temperature; its meaning is
fixed by the chosen thermodynamics.

### Static energy
$$
\Sigma(\theta, S, z) = h^0(\theta, S) - b(\theta, S)\, z ,
$$

where $z$ is the height relative to the surface. Its derivatives give the thermodynamic conjugate variables
relative to $(\theta, S, z)$:

| Derivative | Expression | Name |
|---|---|---|
| $\Sigma_\theta$ | $h^0_\theta(\theta, S) - b_\theta(\theta, S) z$ | Exner function, $\pi$ |
| $\Sigma_S$ | $h^0_S(\theta, S) - b_S(\theta, S)z$ | Chemical potential analogue [(de Szoeke, 2000)](https://journals.ametsoc.org/view/journals/phoc/30/11/1520-0485_2001_031_2814__2.0.co_2.xml) |
| $\Sigma_z$ | $-b(\theta, S)$ | Minus the buoyancy |

### Temperature and chemical potential

The temperature $T$ and chemical potential $\mu$ are rather the conjugates relative to $(\eta, S, z)$. Their
expressions require the extra knowledge of $\eta^0(\theta, S)$, which defines entropy as a function of the model's
entropic variable and salinity.
$$
\begin{aligned}
T(\theta, S, z) &= \left(\frac{\partial \Sigma}{\partial \eta}\right)_{S,z} 
&& = \frac{\pi(\theta, S, z)}{\eta^0_\theta(\theta, S)} \\
\mu(\theta, S, z) &= \left(\frac{\partial \Sigma}{\partial S}\right)_{\eta,z} 
&& = \Sigma_S(\theta, S, z) - T(\theta, S, z)\, \eta^0_S(\theta, S)
\end{aligned}
$$

The specific heat capacity is 

$$
c_p(\theta, S, z) = \frac{T(\theta, S, z)\, \eta^0_\theta(\theta, S)}{T_\theta(\theta, S, z)} .
$$

## Fluxes and tracer equations

Salt diffuses down the isothermal gradient of the chemical potential,

$$
(\nabla \mu)_T = \left(\frac{\partial \mu}{\partial S}\right)_{T,z} \nabla S + \left(\frac{\partial \mu}{\partial z}\right)_{T,S} \nabla z
= \left(\frac{\partial \mu}{\partial S}\right)_{T,z} \left( \nabla S - \Gamma\, \mathbf{k} \right) ,
\qquad
\Gamma = -\frac{(\partial \mu / \partial z)_{T,S}}{(\partial \mu / \partial S)_{T,z}} ,
$$

where $\mathbf{k} = \nabla z$ is the upward unit vector and $\Gamma$ is the barodiffusion gradient. Isothermal
derivatives of a function $f(\theta, S, z)$ can be written as follows:
$$
\left(\frac{\partial f}{\partial S}\right)_{T,z} = f_S - f_\theta\, \frac{T_S}{T_\theta} ,
\qquad
\left(\frac{\partial f}{\partial z}\right)_{T,S} = f_z - f_\theta\, \frac{T_z}{T_\theta} .
$$

Ignoring cross-diffusive (Soret/Dufour) terms, the salt flux $\mathbf{j}_S$ and heat flux $\mathbf{j}_H$ are given by

$$
\mathbf{j}_S = -\kappa_S \left( \nabla S - \Gamma\, \mathbf{k} \right), 
\quad \mathbf{j}_H = - c_p \kappa_T \nabla T.
$$
where $\kappa_S$ and $\kappa_T$ are the molecular diffusivities. From this, the flux of $\theta$ is given by

$$
\mathbf{j}_\theta = \frac{1}{\pi}\mathbf{j}_H - \frac{T_s}{T_\theta} \mathbf{j}_S.
$$


With these, the tracers evolve as 

$$
\frac{D\theta}{Dt} = -\nabla \cdot \mathbf{j}_\theta + \sigma_\theta ,
\qquad
\frac{DS}{Dt} = -\nabla \cdot \mathbf{j}_S ,
$$

with the production of $\theta$ given by
$$
\sigma_\theta = \frac{\varepsilon - \mathbf{j}_\theta \cdot \nabla \pi - \mathbf{j}_S \cdot \nabla \Sigma_S}{\pi},
$$

where for viscous stress tensor $\tau_{ij}$, the viscous dissipation rate is $\varepsilon = \tau_{ij}\, \partial_j u_i $.

The total energy $E = \tfrac{1}{2}|\mathbf{u}|^2 + \Sigma$ satisfies the following conservation equation:
$$
\frac{DE}{Dt} + \nabla \cdot \left(\pi\, \mathbf{j}_\theta + \Sigma_S\, \mathbf{j}_S + p\, \mathbf{u} - \mathbf{u} \cdot \boldsymbol{\tau}\right) = 0 .
$$

The entropy
production is

$$
\sigma_\eta = \frac{\varepsilon}{T}
+ \kappa_T\, \frac{c_p}{T^2}\, |\nabla T|^2
+ \kappa_S\, \frac{(\partial \mu / \partial S)_{T,z}}{T}\, \left| \nabla S - \Gamma\, \mathbf{k} \right|^2.
$$
With `consistent = false`, `thermodynamic_model_components` instead uses the standard Fickian closure:


$$
\mathbf{j}_\theta = -\kappa_T \nabla\theta,
\quad
\mathbf{j}_S = -\kappa_S \nabla S,
\quad
\sigma_\theta = 0.
$$

### Built-in thermodynamics

`CabbelingBoussinesqThermodynamics` has $\theta$ the absolute potential temperature (K) and $S$ in kg/kg:

$$
\begin{aligned}
b &= g \left[ \alpha (\theta - \theta_r) + \tfrac{1}{2} \gamma (\theta - \theta_r)^2 - \beta (S - S_r) \right] , \\
h^0 &= c_p^0\, \theta , \\
\eta^0 &= c_p^0 \ln(\theta / \theta_r) - R_\eta\, S (\ln S - 1) .
\end{aligned}
$$

The default $R_\eta = R / M_S \approx 264.76\ \mathrm{J\,kg^{-1}\,K^{-1}}$
is the ideal entropy of mixing of sea salt, with $M_S = 31.4038218\ \mathrm{g\,mol^{-1}}$ (Millero et al. 2008).
For this thermodynamics, $T = \pi \theta / c_p^0$ and $\Gamma = -g \beta S / (R_\eta T)$.

### Your own thermodynamics

Subtype `AbstractBoussinesqThermodynamics` and add methods for $b$, $h^0$, $\eta^0$ and their first and second
derivatives (18 functions in all):

```julia
using ThermoShenanigans
import ThermoShenanigans: buoyancy, ∂b∂θ, ∂b∂S, ∂²b∂θ², ∂²b∂θ∂S, ∂²b∂S²,
                          potential_enthalpy, ∂h⁰∂θ, ∂h⁰∂S, ∂²h⁰∂θ², ∂²h⁰∂θ∂S, ∂²h⁰∂S²,
                          entropy, ∂η⁰∂θ, ∂η⁰∂S, ∂²η⁰∂θ², ∂²η⁰∂θ∂S, ∂²η⁰∂S²

struct MyThermodynamics{FT} <: AbstractBoussinesqThermodynamics
    # parameters...
end

@inline buoyancy(θ, S, t::MyThermodynamics) = ...
@inline ∂b∂θ(θ, S, t::MyThermodynamics)     = ...
# ... and so on for all 18 functions
```

The functions are evaluated inside GPU kernels, so they must be `@inline`, type-stable and allocation-free.
The tests check every derivative of the built-in thermodynamics against ForwardDiff, which is an easy way to check
your own.

## Discretisation

All thermodynamic functions are evaluated at cell centres from the centred $\theta$, $S$ and $z$, and then
differenced or interpolated to faces. The fluxes are zero at walls, where Oceananigans applies the boundary
conditions.

The discrete equations conserve energy exactly (up to round-off and the time-stepping error) because of three choices:

1. **Buoyancy force.** The $w$ equation uses the discrete-gradient buoyancy

   $$
   \tilde b = \frac{\delta_z(b\, z) - \overline{b_\theta\, z}^z\, \delta_z \theta - \overline{b_S\, z}^z\, \delta_z S}{\Delta z}
   $$

   at $w$-points, where $\delta_z$ is the vertical difference and $\overline{(\cdot)}^z$ the vertical average.
   With second-order centred tracer advection, the buoyancy work $\sum w \tilde b\, \Delta V$ then exactly balances
   the advective change of $\sum \Sigma\, \Delta V$ whenever $h^0$ and $b$ are at most quadratic in $(\theta, S)$.
   For a linear $b$, $\tilde b$ is the usual average $\overline{b}^z$.
2. **Dissipation.** $\varepsilon$ is assembled from the viscous fluxes at their native locations, so that
   $\sum \varepsilon\, \Delta V$ is exactly the kinetic energy removed by the viscous closure.
3. **Flux conversions.** $\mathbf{j} \cdot \nabla \pi$ is formed at faces and averaged to centres, so that by summation
   by parts $\sum (\mathbf{j}_\theta \cdot \nabla \pi)\, \Delta V = -\sum \pi\, (\nabla \cdot \mathbf{j}_\theta)\, \Delta V$.

Exact conservation requires Oceananigans' default `Centered(order = 2)` advection for momentum and tracers.
$\sigma_\theta$ evaluates $\varepsilon$ without closure fields, so momentum closures that compute their own fields
(such as Smagorinsky) are not yet supported.

**Boundary conditions.** Wall fluxes of $\theta$ and $S$ are set only by `FluxBoundaryCondition` (the default is
no flux); `ValueBoundaryCondition` and `GradientBoundaryCondition` are not supported with the consistent closure
(they are with `consistent = false`). A prescribed flux is a flux of $\theta$, not of heat: the energy input is
$\pi Q_\theta + \Sigma_S Q_S$ per unit area at the boundary cell, so a heat flux $Q_H$ corresponds to
$Q_\theta = Q_H / (\rho_0 \pi)$.

**Known Oceananigans bug on stretched vertical grids.** On grids with variable vertical spacing, Oceananigans'
centred momentum advection (tested with v0.113.4) does not conserve kinetic energy, even with a divergence-free
velocity. In `src/Advection/centered_advective_fluxes.jl`, the horizontal fluxes of $w$-momentum are, schematically,

```julia
advective_momentum_flux_Uw = Axᶠᶜᶠ * ℑzᵃᵃᶠ(u) * ℑxᶠᵃᵃ(w)
advective_momentum_flux_Vw = Ayᶜᶠᶠ * ℑzᵃᵃᶠ(v) * ℑyᵃᶠᵃ(w)
```

that is, $A_x^{(w)}\, \overline{u}^z$ with the face area $A_x^{(w)} = \Delta y\, \Delta z^{(w)}$ of the $w$-cell,
rather than the
volume flux $\overline{A_x u}^z$ that the docstring of `div_𝐯w` states. The two agree only when neighbouring cells
have equal thickness. Otherwise the velocity transporting $w$-momentum is not discretely divergence-free over the
$w$-cells, and the advection term leaks kinetic energy. The $u$ and $v$ equations are unaffected. In a stretched-grid
test with no buoyancy and no closure, $\int w\, G_w\, dV \approx 8 \times 10^{-9}$ while $\int u\, G_u\, dV$ and
$\int v\, G_v\, dV$ are at round-off ($10^{-22}$). The energy drift on stretched grids therefore comes from
Oceananigans, not from ThermoShenanigans; the stretched-grid energy test turns off momentum advection for this reason.

## Examples


| Example | What it shows |
|---|---|
| `examples/energy_conservation` | 2D Rayleigh–Taylor instability with strong cabbeling; closes the energy budget and shows the third-order (RK3) fall of the energy error with $\Delta t$ |
| `examples/two_layer_cabbeling` | 2D two-layer cabbeling instability after Bisits, Zika & Sohail (2025, JFM 1011 A17) |
| `examples/two_layer_cabbeling/gpu` | the same experiment in 3D on a GPU, with a Slurm script and offline plotting |

Run them with, for example,

```bash
julia --project=examples examples/energy_conservation/energy_conservation.jl
```

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test(julia_args = ["--check-bounds=auto"])'
```

`Pkg.test()` on its own forces `--check-bounds=yes`, which roughly doubles the time spent compiling the model kernels.
Set `TEST_ARCHITECTURE=GPU` to also run the architecture tests on a GPU.

The tests check every thermodynamic derivative against ForwardDiff, closed forms and Maxwell relations for the
built-in thermodynamics, zero fluxes in a resting isothermal column, wall and conservation properties of the
closure, $\sigma_\eta \ge 0$, the discrete energy identity on uniform and stretched grids, third-order convergence
of the RK3 energy error, CPU (and, where available, GPU) model steps in Float64 and Float32, doctests and explicit imports.
