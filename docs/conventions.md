# Pinned conventions (Phase 0, step 3)

Measured against DESC 0.17.2 on `SOLOVEV`, `DSHAPE`, `HELIOTRON`.
Each item below is enforced by a test in `tests/`.

## 1. Jacobian — agrees with the proposal
`det F = R (R_θ Z_ρ − R_ρ Z_θ)` equals DESC's `sqrt(g)` with **sign +1**,
to 2.4e-16. Eq. (6) can be used verbatim; no absolute value needed.

## 2. Units — DESC is SI, the proposal sets µ₀ = 1
DESC's `W_B` is `∫|B|²/(2µ₀) dV`. The proposal's Eq. (4) uses µ₀ = 1.
`EnergyModel` folds `1/µ₀` into `c(ρ) = ½ψ'²/µ₀` so `W_B` matches DESC to
10 digits. Pass `mu0=1.0` to work in the proposal's units.

## 3. The λ stream function is required, not optional
Eq. (3) uses `m = (0, ι, 1)`. DESC's stock equilibria carry a nonzero λ
(`max|λ_θ| = 0.45` on SOLOVEV), and the flux representation is really

    B^ζ·√g = ψ'(1 + λ_θ)        [verified to 1.2e-16]

so the correct field-line vector is §6.6's `m(λ) = (0, ι − λ_ζ, 1 + λ_θ)`.

| form | W_B / DESC W_B |
|---|---|
| `m(λ) = (0, ι−λ_ζ, 1+λ_θ)` | 1.0000000000 |
| `m = (0, ι, 1)` | 1.0294 |

The bare form misprices the magnetic energy by **2.9%**. Consequence for the
plan: SPEC_v0's "pin the angle *or* optional λ block" is not a free choice at
validation time — comparing against DESC on stock equilibria requires λ.
Pinning the angle is a *different gauge*, valid for solving but not for
coefficient-level comparison against DESC output.

## 4. Pressure enters as −pJ, correcting Eq. (4)

**The proposal's Eq. (4) sign is wrong.** Eq. (4) writes `+ p det F`; the
correct term is `− p det F`.

Evidence — at a converged DESC equilibrium, ∇W along boundary-preserving
variations (the null space of the ρ=1 evaluation map, so A4 holds) must vanish:

| case | ‖∇W_B‖ | ‖∇∫pJ‖ | `W_B − ∫pJ` | `W_B + ∫pJ` | ratio |
|---|---|---|---|---|---|
| SOLOVEV | 9.115e0 | 9.041e0 | **4.04e-1** | 1.82e1 | 45 |
| DSHAPE | 7.7282e4 | 7.7282e4 | **7.89e-1** | 1.55e5 | 195846 |
| HELIOTRON | 1.8701e6 | 1.8701e6 | **4.80e2** | 3.74e6 | 7800 |

The two gradients agree to 5 digits and cancel *only* under the minus sign.
The surviving residual tracks DESC's own `<|F|>_vol` (6.8e-3, 2.1e-2, 2.5e1),
i.e. it is DESC's convergence error, not a systematic term.

This is also what the closure implies: with Remark 1's thermal energy
`T = (1/(γ−1)) ∫ M^γ V'^{1−γ} dρ`, we get `∂T/∂V' = −p`, so the
linear-in-J surrogate carries `−p`. DESC agrees (`W_p = −∫p dV`).

### Why this does not disturb the strategy
Proposition 1 already covers it: convexity in `(u, J)` holds "regardless of
the sign of p, since the second term is linear". The rotated-cone
reformulation, the affine model `L_k`, and the whole SOCP structure are
unchanged.

One caveat to carry forward: with `−pJ` the objective is no longer bounded
below by the pressure term alone (it rewards large J). Boundedness comes from
the fixed boundary — `∫J dρdθdζ` is the plasma volume, which A4 fixes — plus
the Jacobian floor and the prox/trust region. Worth re-checking when the
free-boundary mode of Phase 4 removes that constraint.
