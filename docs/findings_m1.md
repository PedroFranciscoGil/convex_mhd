# M1 findings: the subproblem works, the energy needs two extra conditions

Phase 1 steps 5-6 are implemented (`subproblem.py`, `lambda_block.py`,
`outer.py`) and Algorithm 1 runs. This file records what the numerics said,
including one result that revises `docs/conventions.md` finding 4.

## What works

**The affine determinant model is faithful.** The trust-region ratio
(true decrease / predicted decrease) is 0.98-1.000 across four orders of
magnitude in the damping parameter, and 1.000 for every accepted step from a
perturbed start. This is Proposal A's central claim, confirmed.

| mu | 100 | 10 | 3 | 1 | 0.3 | 0.1 |
|---|---|---|---|---|---|---|
| ratio | 0.998 | 0.998 | 0.997 | 1.000 | 0.999 | 1.000 |

**The lambda block recovers DESC's own lambda to 1.2e-7 relative**, from an
independent weighted least-squares route. Strong check on the geometry
pipeline.

**The Jacobian barrier does its job.** `min L/J_k` never went below 0.4 in any
run; no step ever drove det F toward zero. VMEC's BAD JACOBIAN mode really is
structurally excluded.

## Three things the proposal does not mention, all of which had to be fixed

### 1. The boundary constraint must be reduced to full rank
The rho=1 evaluation matrix is 265 x 91 with **rank 13** on SOLOVEV. Passing all
265 rows imposes ~250 redundant equalities, the interior-point KKT system goes
singular, and Clarabel stalls at `InsufficientProgress` from iteration 0. Using
an orthonormal basis of the row space fixes it (`Zero` cone 530 -> 25).

### 2. The prox metric is load-bearing, not a refinement
`cond(D) ~ 7e17`; `||D||_2` is set by a few stiff near-axis high-order Zernike
directions. With `S = I` the step unit collapses in *every* direction to suit
those few, `u` goes numerically frozen (`a_scale*U/|u| ~ 3e-5`), and the solver
stalls. Column-equilibrating the determinant operator -- Sec. 6.2's "spectrally
scaled prox metric" -- is what makes the subproblem solvable at all.

### 3. det F > 0 does not keep the map physical
This is the substantive one. Pure descent on W, with every det F strictly
positive and every ratio at 1.000, **inflates `int det F` by 65%** against a
pinned boundary while driving the force residual from 4e4 to 4e7. The interior
flux surfaces balloon past the boundary: `det F > 0` buys only *local*
invertibility, and nothing in the polyconvex formulation forbids global
non-injectivity. This is the known gap in Ball's framework, and the standard
remedy applies -- the **Ciarlet-Necas condition** `int det F <= vol(image)`,
which, because det F is already lifted to L, is a single *linear* constraint.

With it, the volume holds to 1.00000 and a perturbed start converges to
`W = 2.02787762e6` against DESC's `2.02791322e6` (**1.8e-5 relative**), all
ratios 1.000.

## Revision to conventions.md finding 4

Finding 4 said the pressure term must be `-p det F` rather than Eq. (4)'s
`+p det F`. **The gradient evidence stands** -- and is now corroborated
independently: Remark 1's adiabatic closure, calibrated so `p = M^g/V'^g`,
gives a stationarity measure of 4.8886e-08 at the DESC equilibrium, *identical
to five digits* to the `-int p J` value, versus 2.1975e-06 for `+int p J`.

But the reason given for why this "does not disturb the strategy" was too
glib. With `-pJ` the energy has no lower bound from the pressure term, and the
claim that A4 saves it ("`int J` is the volume, which A4 fixes") is **false as
stated**: A4 fixes the boundary, not the integral, precisely because of the
injectivity gap above. Two conditions are needed:

* the adiabatic closure of Remark 1 for a thermal term bounded below
  (`V'^{1-g} -> 0` rather than `-pJ -> -inf`); it is convex in V' and stays
  power-cone representable, so the conic structure survives; and
* Ciarlet-Necas for global injectivity.

Neither alone is sufficient -- the adiabatic closure by itself still ran away
(volume ratio 1.65), because the runaway is geometric, not thermodynamic.

## The drift was two bugs, not a landscape defect

The earlier writeup blamed the energy landscape. That was wrong. The
equilibrium is a genuine stationary point -- grad W projected onto
boundary-preserving directions is **4.9e-8** (relative), and removing the
volume-changing direction barely moves it (4.7e-8). The Hessian there is
positive definite (144 eigenvalues in [1.07e4, 2.23e9]). The drift was two
implementation bugs that a scan of `dW` vs trust radius exposed:

**Bug 1 -- boundary constraint imposed in scaled coordinates.** The physical
step is `a_scale * z / col` (column-equilibrated), but A4 was imposed as
`AbR @ z = 0`, which does *not* imply `AbR @ (z/col) = 0`. So every step leaked
into the boundary-moving subspace, where the gradient is ~1e6, and descended
*that* -- reading as faithful energy decrease (ratio ~ 1) while quietly
violating the fixed boundary. Signature: `dW ~ delta` (linear) from the
equilibrium instead of `dW ~ delta^2`. Fixed by scaling the constraint rows by
`1/col`. After the fix, starting at the equilibrium gives `|z| = 0`,
`dW = O(1e-10)`, `<|F|>` frozen at 6.8e-3 for every trust radius: the
fixed-point property of Sec. 6.5, working.

**Bug 2 -- `model_at_ak` inconsistent with the objective in adiabatic mode.**
The predicted decrease used a hardcoded `W_B - int pJ`, but the adiabatic
objective carries the thermal term, so `pred_decrease` came out negative while
W actually fell and the ratio test rejected every good step. At `a_k` the
affine model is exact, so `model_at_ak = W(a_k)` for either closure. Fixed; the
ratio is 1.000 again.

Both are guarded by the trust-region radius (a hard `||z|| <= delta` cap) and
the SCS feasibility check -- SCS returns "optimal_inaccurate" points up to 7x
past the radius, so it is now a gated fallback behind CLARABEL.

## Where M1 actually stands

* **Fixed-point property: met.** Start at the DESC equilibrium -> zero drift,
  `<|F|>` preserved (even nudged to 5.3e-3), zero steps taken.
* **Convergence from a perturbation: faithful but slow -- a conditioning
  problem, not a closure one.** Both closures behave *identically* (`<|F|>`
  4.1e4 -> ~7.2e3 over 60 steps, prescribed-p and adiabatic within 0.01%), which
  rules the closure out as the blocker. Every ratio holds at 1.000, so the model
  is faithful; the trouble is that each accepted step is tiny (`|z| ~ 0.0028`,
  far *inside* the trust radius 0.05) and the stationarity measure crawls (0.108
  -> 0.101 over 60 steps, ~7e-5 per step). The cause is the energy Hessian's
  condition number ~2e5 (eigenvalues [1.07e4, 2.23e9]): the prox-linear step is
  a regularized Newton step, and it barely moves along the soft, force-balance-
  relevant directions while the stiff perspective-term directions dominate.

This is precisely what Sec. 6.2's spectrally scaled prox metric `S` exists to
fix ("playing the role of VMEC's preconditioner"). The current `S` only
column-equilibrates the *determinant* operator `D`; it does not capture the full
energy Hessian's anisotropy, which is what governs the outer convergence rate. A
proper preconditioner -- the diagonal-of-Hessian or a low-rank Gauss-Newton
approximation -- is the missing piece, and it is squarely Phase 3 work
("diagonal preconditioning, matrix-free").

M1's gate ("DESC's default force-residual tolerance in <= 30 outer iterations")
is **not yet met**: the remaining gap is convergence *speed*, traced to the prox
preconditioner, not correctness of the conic machinery, which now behaves
exactly as the proposal advertises (faithful model, held boundary, held volume,
positive Jacobian, true fixed point at the equilibrium).
