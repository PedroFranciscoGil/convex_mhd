# Convex MHD — v0 specification (frozen)

Implementation of the convex reformulation of 3D finite-beta MHD force balance
from `linearized_mhd.pdf` (Proposal A: prox-linear sequential conic programming).
This file freezes the choices the proposal leaves open (Phase 0, step 1).

## Host & environment
- **Host code:** DESC (Fourier–Zernike pseudospectral, JAX). Chosen to follow the
  proposal's Section 10 verbatim. (Alternative considered: the local `vmec_jax`.)
- **Env:** conda env `desc` (Python 3.12, `desc-opt` from PyPI).

## Problem class (proposal assumptions A1–A4)
- Fixed-boundary, nested-surface, finite-beta ideal MHD. Stellarator symmetry ON.
- Unknowns: spectral coeffs `a` of `R(ρ,θ,ζ)`, `Z(ρ,θ,ζ)`; toroidal angle ζ undisplaced.
- Prescribed data: toroidal flux `2πψ(ρ)`, rotational transform `ι(ρ)`, pressure `p(ρ)`.
- Straight-field-line angles; ζ = geometric toroidal angle.

## Discretization
- Basis: Fourier–Zernike for R, Z (DESC default).
- Quadrature: Gauss–Legendre in ρ; uniform angular grids, trapezoidal weights.
- Starting resolution (M0/M1): L = M = N = 6, bump to ~8–12 for convergence-order tests.

## Energy (Eq. 8) — as corrected in Phase 0, see `docs/conventions.md`
`W = ∫ [ c(ρ) ‖u‖²/J − p(ρ) J ] dρ dθ dζ`,  `c = ½ ψ'(ρ)²/µ₀`,
`u = F m(λ)`,  `J = det F = R τ`,  `τ = R_θ Z_ρ − R_ρ Z_θ`.

Three deviations from the PDF, each measured rather than assumed:
- **pressure sign is −**, not the `+p det F` of Eq. (4). Verified by stationarity
  at DESC equilibria on 3 cases (ratio 45–2e5). Prop. 1 is unaffected — the term
  is linear in J either way, so all convex structure survives.
- **µ₀** folded into `c` so W matches DESC's SI `W_B`. Pass `mu0=1.0` for the
  PDF's units.
- **λ is required**, not optional: `m(λ) = (0, ι − λ_ζ, 1 + λ_θ)` (§6.6). The
  bare `m = (0, ι, 1)` of Eq. (3) misprices `W_B` by 2.9% on stock DESC
  equilibria, which carry `max|λ_θ| ≈ 0.45`.

## Convex model (Proposal A)
- Linearize only J: `L_k = J_k + cof F_k : (F − F_k)`  (Eq. 12; cofactor via Eq. 11).
- Per-node rotated second-order cone: `‖u_q‖² ≤ s_q L_{k,q}` (Lemma 1).
- Relative Jacobian floor `L ≥ κ J_k`, κ ≈ 0.2.
- Spectrally-scaled prox term; μ adapted by trust-region ratio test on the TRUE W.
- Gauge: pin poloidal angle (spectral condensation) or optional λ block (§6.6).

## Benchmark ladder → DESC examples (§9)
| Proposal case                    | DESC example        |
|----------------------------------|---------------------|
| axisym Solov'ev (analytic)       | `SOLOVEV`           |
| shaped finite-β tokamak          | `DSHAPE` / `HELIOTRON` cross-check |
| heliotron w/ Shafranov shift     | `HELIOTRON`         |
| strongly-shaped stellarator, β↑  | `W7-X`              |
| rotating-ellipse stellarator     | build from boundary |

## Milestones (§10)
- **M0** — gradient + operator tests pass to ~machine precision.  ← current target
- **M1** — axisym prototype: DESC-tol on Solov'ev/tokamak in ≤30 outer iters, monotone descent.
- **M2** — full 3D + β-continuation; ≥90% unattended convergence on perturbed ensemble.
- **M3** — matrix-free PDHG/ADMM backend; surrogate mode ≥10× cheaper.
- **M4** — prescribed-current + free-boundary parity with VMEC modes.
- **M5** — Proposal B lifted moment relaxation (research track).

## Status
**M0 reached** (Phase 0 steps 1–4 complete, 11/11 tests green on DESC 0.17.2).

- `src/convex_mhd/energy.py` — `W(a)` over DESC Fourier–Zernike transforms in
  JAX, autodiff gradient. Reproduces DESC `W_B`/`W_p` to 10 digits.
- `src/convex_mhd/operators.py` — node kernels: `u_q(a)` linear, `L_{k,q}(a)`
  affine via the Eq. (11) cofactor form; matches autodiff of `det F` to 1e-11
  and is second-order accurate (measured order 2.0).
- `docs/conventions.md` — the four pinned conventions, incl. the Eq. (4)
  pressure-sign correction.

Variational identity was checked in the *stationarity* form (∇W ≈ 0 along
boundary-preserving variations at a DESC equilibrium) rather than by
finite-differencing δW against the collocated force residual. That is the
stronger and cheaper check; the residual-vector comparison is still worth doing
if per-component force diagnostics are ever needed.

**Next (M1, Phase 1 step 5):** assemble subproblem (14) in CVXPY + Clarabel —
rotated cones per node, boundary equalities, Jacobian floor `L ≥ κJ_k`, prox
term — then the trust-region outer loop on the true W. Open question to settle
first: gauge handling (spectral condensation vs. the λ block), where §6.6's
"lean option" now looks less attractive given finding 3.
