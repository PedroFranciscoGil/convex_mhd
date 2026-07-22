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
* **Convergence from a perturbation: now works, after fixing the step control.**
  The earlier crawl (`<|F|>` stuck at ~7e3, stationarity ~7e-5/step) was *not*
  the preconditioner and *not* the closure -- both closures and both prox
  metrics gave byte-identical steps. A mu/delta scan pinned it: the step was
  throttled entirely by the prox weight `mu`, which was ~5 orders of magnitude
  too large. `|z|` was invariant to the trust radius (0.0028 at delta =
  0.02...2.0, never binding) but scaled as `1/mu`: at mu=1e-6 a *single* step
  cut `<|F|>` from 7834 to 1029.

  The proposal frames mu as the trust-region control; here that is backwards.
  With the adiabatic closure and Ciarlet-Necas supplying boundedness, mu should
  be a whisper of Tikhonov (~1e-6) and the hard radius `delta` is the real trust
  region. Reconfigured that way, from the perturbed start: **W converges to DESC
  to 2e-7** and `<|F|>` falls geometrically, 4.1e4 -> ~1.4e2 in 30 steps (a 300x
  reduction), at a per-step rate ~0.88.

* **The Hessian prox metric is a solver speedup, not a step change.** Replacing
  the determinant-column metric with `sqrt(diag energy Hessian)` (`precond=
  "hessian"`) leaves the mathematical step identical but cuts wall-clock ~9x
  (52s vs 463s for 15 steps), because the interior-point KKT system is far
  better conditioned. Kept on by default.

## Cold-start robustness (the real test)

Everything above started from the DESC answer plus a small perturbation. The
proposal's actual claim is convergence from a *naive* guess. Tested with DESC's
boundary-to-axis interpolation (`set_initial_guess()`), using **no knowledge of
the solution** -- the Ciarlet-Necas volume is taken from the cold map's own
`int det F`, which equals the boundary-enclosed volume for any bijection (the
two agree to 1.00000, confirming it). Reproduce with `benchmarks/cold_start.py`.

| case | cold `<F>` | -> final | W vs DESC | steps |
|---|---|---|---|---|
| SOLOVEV (axisym, analytic) | 1.10e4 | 1.0e1 | 6 digits | 15 |
| DSHAPE (shaped finite-beta tokamak) | 3.51e4 | 7.1e2 | 7e-6 rel | 60 (capped) |

Both converge to the DESC equilibrium energy from a cold start. The converged W
matches DESC to ~6 digits even on the shaped tokamak, where the cold linear
interpolation is far from the answer (`<F>` = 3.5e4). This is the robustness
property the whole approach is for, and it holds. The force residual on DSHAPE
is still descending at the iteration cap (711, geometric) -- W is already at the
minimum while the gradient still has distance to run, which is the expected
signature of a quadratic minimum.

## The convergence rate: traced to mu, then to solver degeneracy

The ~0.9 linear rate was chased down in three steps.

1. **The alternating lambda block slows it, but is not the main cause.**
   Freezing lambda at its correct value (perturb R,Z only from the equilibrium)
   improves the effective rate from 0.87 to 0.73. Real, but secondary, and not
   available from a cold start where lambda = 0.

2. **mu was the throttle -- again.** With lambda frozen, shrinking the prox
   weight sharply accelerates and turns the rate near-superlinear early:

   | mu | eff. rate | steps | final `<F>` |
   |---|---|---|---|
   | 1e-6 | 0.62 | 25 | 13 |
   | 1e-8 | 0.46 | 20 | 0.038 |
   | 1e-10 | 0.42 | 18 | **0.012** |

   At mu=1e-10 the residual reaches DESC's own tolerance (6.8e-3) in 18 steps,
   and the first few steps drop the stationarity four orders of magnitude --
   the near-Newton behaviour prox-linear should show, confirming the fixed prox
   was the superlinear-killer.

3. **But small mu exposes an interior-point degeneracy near the solution.** With
   lambda alternation on (the real cold-start setting), a fixed small mu stalls:
   as the step shrinks toward the solution the SOCP cones become exactly tight,
   CLARABEL's KKT system goes degenerate and fails for *every* trust radius, and
   SCS overshoots the radius (see below), so no solver delivers a step. mu=1e-6
   actually reaches a lower residual than mu=1e-8 for this reason.

   **Fix: adaptive mu.** Grow mu on any rejected/failed step (re-regularizing
   the KKT system, rescuing CLARABEL) and let it decay on clean steps. This
   pushed SOLOVEV from a stall at stationarity 1.3e-4 (step 8) down to 9e-6
   (step 34), W matching DESC exactly. It still stalls eventually, at
   stationarity ~9e-6 (`<F>` ~ 2), when the required step (~3e-6) is below what
   the interior-point backend can resolve.

**Conclusion on rate.** The residual barrier is not the algorithm -- ratios stay
1.00, the model is faithful to the end -- but the interior-point *backend* at
tiny steps. This is exactly the regime the proposal earmarks for Phase 3: a
matrix-free first-order solver (PDHG/ADMM) with closed-form cone projections and
primal-dual warm starts, which does not degenerate on tight cones and reuses the
previous step's solution. Reaching DESC's force tolerance from a cold start on
the shaped cases is expected to need that backend, not more outer iterations.

## Where M1 stands now

Machinery: **correct and converging.** Faithful model (ratio 1.000), true fixed
point at the equilibrium, boundary and volume held, `det F > 0` throughout, and
geometric convergence to the DESC equilibrium (W to 2e-7).

The one gap left to the formal gate is *rate*: ~0.88/step means ~100 outer
iterations to DESC's 6.8e-3 force tolerance, against the gate's target of <= 30.
The linear rate (not the quadratic one prox-linear can achieve) is the next
thing to chase -- candidates: the alternating lambda block (block-coordinate
descent is linearly convergent; folding lambda into the cone step would couple
them), the first-order det F linearization, or residual mu damping. That is a
convergence-acceleration task, with all the correctness pieces now in place.
