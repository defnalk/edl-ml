# Scientific background

## The electric double layer

At the interface between a charged electrode and an ionic electrolyte, ions
rearrange to screen the electrode charge. The classical Gouy-Chapman-Stern
(GCS) picture decomposes the interface into two regions:

- **Stern layer** — a compact molecular condenser of thickness `d_H` (the
  closest approach of a hydrated ion) with a reduced dielectric constant due
  to water orientation, giving a constant capacitance
  `C_H = ε_H ε₀ / d_H`.
- **Diffuse layer** — a thermally smeared cloud of ions obeying Poisson-
  Boltzmann statistics, with a potential-dependent capacitance `C_d`.

The total differential capacitance is the series combination:

$$
\frac{1}{C_{\mathrm{dl}}} = \frac{1}{C_H} + \frac{1}{C_d}.
$$

## Poisson-Boltzmann equation

For a symmetric :math:`z:z` electrolyte with bulk number density `n_0`, the
one-dimensional PB equation is

$$
\frac{d^2 \psi}{dx^2} = \frac{2 z e n_0}{\varepsilon_r \varepsilon_0}
    \sinh\!\left(\frac{z e \psi}{k_B T}\right).
$$

Subject to `ψ(∞) = ψ'(∞) = 0` it admits the first-integral closed form

$$
\frac{d\psi}{dx} = -\,\operatorname{sgn}(\psi)\,
    \frac{2 k_B T \kappa}{z e}\,
    \sinh\!\left(\frac{z e \psi}{2 k_B T}\right),
$$

which `edl-ml` integrates forward from `x = 0` using `scipy.integrate.solve_ivp`
(LSODA, rtol `1e-10`). This formulation is numerically stable because the
bulk state is an *attracting* fixed point under forward integration, unlike
the second-order form whose `ψ = 0` equilibrium is hyperbolic.

## Grahame equation

Integrating Gauss's law across the diffuse layer yields the closed-form
diffuse-layer surface charge density

$$
\sigma_d = \operatorname{sgn}(\psi_d)\,
    \sqrt{8 \varepsilon_r \varepsilon_0 n_0 k_B T}\,
    \sinh\!\left(\frac{z e \psi_d}{2 k_B T}\right),
$$

used directly by [`solve_poisson_boltzmann`][edl_ml.physics.pb.solve_poisson_boltzmann]
and [`gouy_chapman_stern`][edl_ml.physics.gcs.gouy_chapman_stern].

Differentiating with respect to `ψ_d` gives the diffuse-layer differential
capacitance

$$
C_d = \varepsilon_r \varepsilon_0 \kappa
      \cosh\!\left(\frac{z e \psi_d}{2 k_B T}\right).
$$

## Self-consistent GCS split

The electrode potential `E` (relative to the point of zero charge) is split
between the two layers,

$$
E = \psi_H + \psi_d, \qquad \psi_H = \sigma / C_H,
$$

with `σ` defined by the Grahame equation as a function of `ψ_d`. The
transcendental equation `E - ψ_d - σ(ψ_d) / C_H = 0` is monotone in `ψ_d`
and is solved by bisection in
[`gouy_chapman_stern`][edl_ml.physics.gcs.gouy_chapman_stern]. The resulting
residual is `|E − ψ_H − ψ_d| < 10⁻⁹` V across the full sampling box.

## Validation

`tests/test_pb.py::test_pb_profile_matches_analytical_gouy_chapman` asserts
`max |ψ_numeric − ψ_analytic| < 10⁻⁸ V` over the entire domain for
`c = 50 mM`, `ψ_d = 50 mV`, where the analytical profile is

$$
\psi(x) = \frac{4 k_B T}{z e}
    \operatorname{arctanh}\!\left(
        \tanh\!\left(\frac{z e \psi_d}{4 k_B T}\right) e^{-\kappa x}
    \right).
$$

`tests/test_pb.py::test_pb_grahame_consistency` asserts that the surface
charge returned by the solver matches the closed-form Grahame equation to
1 ppm relative tolerance across ~25 randomly sampled diffuse-layer
potentials.
