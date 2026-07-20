# convex_mhd

A convex reformulation of 3D finite-beta MHD force balance, built on
[DESC](https://github.com/PlasmaControl/DESC).

Written in inverse (flux) coordinates, the ideal-MHD energy density is a
*perspective function* — jointly convex in the Jacobian matrix `F` and its
determinant `det F`. Every source of nonconvexity in the finite-beta problem is
therefore funnelled into the single cubic identification `J = det F`. Linearize
only that, and each solver step becomes a second-order cone program with the
exact convex physics retained, including the `1/det F` barrier that structurally
prevents flux-surface collapse.

This repo implements **Proposal A** — prox-linear sequential conic programming.
See [`SPEC_v0.md`](SPEC_v0.md) for the frozen v0 specification and milestones.

## Status

**M0 reached.** The energy and the node operators feeding the conic subproblem
are implemented and verified against DESC.

| module | what it does |
|---|---|
| `src/convex_mhd/energy.py` | `W(a)` of Eq. (8) over DESC's Fourier–Zernike transforms in JAX, autodiff gradient |
| `src/convex_mhd/operators.py` | node kernels: `u_q(a)` linear, `L_{k,q}(a)` affine (Eq. 11/12) |

Verified: `W_B`/`W_p` match DESC to 10 digits; `det F` matches DESC's `sqrt(g)`
to 2e-16; the affine determinant model matches autodiff to 1e-11 with measured
second-order accuracy; and `W` is stationary at DESC equilibria along
boundary-preserving variations.

Next: assemble subproblem (14) in CVXPY/Clarabel and the trust-region outer loop.

## Conventions, and one correction

[`docs/conventions.md`](docs/conventions.md) records four conventions pinned by
measurement against DESC 0.17.2. One is a correction to the source proposal:

> **The pressure term enters as `−p·det F`, not `+p·det F`.**

At a converged DESC equilibrium the null-space gradients of `W_B` and of `∫pJ`
agree to five digits and cancel *only* under the minus sign — by a factor of 45
(SOLOVEV), 2e5 (DSHAPE), 7800 (HELIOTRON) — with the residual tracking DESC's
own force residual in each case. It is also the sign the adiabatic closure
implies, since `∂T/∂V′ = −p`.

This does not disturb the strategy: the term is linear in `J` either way, so the
joint convexity in `(u, J)` and the whole conic structure survive unchanged.

## Setup

```bash
conda create -n desc python=3.12 && conda activate desc
pip install desc-opt pytest
pytest tests/ -q
```
