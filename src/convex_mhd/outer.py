"""Algorithm 1: prox-linear sequential conic programming (Sec. 6.4).

Each outer iteration solves the convex subproblem of Eq. (14), accepts or
rejects the step on a trust-region ratio test against the *true* energy W, and
optionally re-solves the lambda block.  Convergence of the ratio to 1 is the
signature that the affine determinant model is faithful.

Stationarity is measured as the norm of grad W projected onto the
boundary-preserving subspace (A4), relative to the raw gradient norm -- the
same measure that validated the energy in Phase 0.  `force_residual` gives the
physical <|F|>_vol via DESC for comparison at matched tolerance.
"""

import numpy as np
import jax

from .subproblem import ConicSubproblem, boundary_matrix
from .lambda_block import LambdaBlock


def force_residual(eq, model, a, grid):
    """DESC's volume-averaged force residual <|F|>_vol at the coefficients a."""
    e = eq.copy()
    aR, aZ, aL = model.unpack(np.asarray(a))
    e.R_lmn, e.Z_lmn, e.L_lmn = np.asarray(aR), np.asarray(aZ), np.asarray(aL)
    return float(e.compute("<|F|>_vol", grid=grid)["<|F|>_vol"])


class ProxLinearSolver:
    """The outer loop of Algorithm 1."""

    def __init__(self, model, ops, eq, grid, kappa=0.2, use_lambda=True, volume=None):
        self.m, self.eq, self.grid = model, eq, grid
        self.sub = ConicSubproblem(model, ops, eq, kappa=kappa, volume=volume)
        self.lam = LambdaBlock(model, eq, grid) if use_lambda else None
        self.PR = boundary_matrix(eq, eq.R_basis)
        self.PZ = boundary_matrix(eq, eq.Z_basis)

    def stationarity(self, a):
        """||P grad W|| / ||grad W||, P projecting onto boundary-preserving moves."""
        g = np.asarray(jax.grad(self.m.W)(a))
        gR, gZ = g[:self.m.nR], g[self.m.nR:self.m.nR + self.m.nZ]
        pR = gR - self.PR.T @ (self.PR @ gR)
        pZ = gZ - self.PZ.T @ (self.PZ @ gZ)
        raw = np.linalg.norm(g)
        return float(np.hypot(np.linalg.norm(pR), np.linalg.norm(pZ)) / (raw + 1e-300))

    def solve(self, a0, mu0=1.0, max_iter=30, tol=1e-8, eta=0.1,
              mu_min=1e-4, mu_max=1e8, shrink=0.5, grow=4.0,
              delta0=0.05, delta_max=0.2, eta_good=0.75,
              max_rejects=8, verbose=True, track_force=False):
        a = np.asarray(a0)
        J0 = np.asarray(self.m.geometry(a)["J"])
        if np.min(J0) <= 0:
            raise ValueError(
                f"initial map is folded: min det F = {J0.min():.3e} <= 0. "
                "Algorithm 1 preserves det F > 0 but cannot repair a start that "
                "already violates it (A1)."
            )
        if self.lam is not None:
            a = self.lam.solve(a)
        mu = mu0
        delta = delta0
        W = float(self.m.W(a))
        hist = []

        if verbose:
            print(f"{'it':>3} {'W':>15} {'dW/W':>11} {'ratio':>7} {'delta':>9} "
                  f"{'|step|':>9} {'stat':>9} {'minL/J':>7} {'slv':>8}")

        for k in range(max_iter):
            stat = self.stationarity(a)
            if verbose:
                extra = ""
                if track_force:
                    extra = f"  <|F|>={force_residual(self.eq, self.m, a, self.grid):.3e}"
                print(f"{k:>3} {W:>15.8e} {'':>11} {'':>7} {delta:>9.2e} "
                      f"{'':>9} {stat:>9.2e}{extra}")
            if stat < tol:
                break

            # --- step with trust-region ratio test on the TRUE energy ---
            # The radius delta is the primary control (Sec. 6.5); mu is held
            # fixed as a mild conditioner.  On a good ratio the radius grows, on
            # a poor one it shrinks and the step is re-solved -- the standard
            # trust-region loop, now with a real (hard) radius.
            accepted = False
            for _ in range(max_rejects):
                try:
                    a_try, info = self.sub.solve(a, mu=mu, delta=delta)
                except Exception as e:
                    hist.append({"iter": k, "event": "solver_fail", "delta": delta, "err": str(e)[:80]})
                    delta *= shrink
                    continue
                W_try = float(self.m.W(a_try))
                pred = info["pred_decrease"]
                if pred <= 0:                      # model predicts no progress
                    delta *= shrink
                    continue
                ratio = (W - W_try) / pred
                if ratio >= eta and np.isfinite(W_try):
                    a, W, accepted = a_try, W_try, True
                    # grow the radius only when the model is very accurate AND
                    # the step actually pushed against the boundary
                    if ratio >= eta_good and info["rel_step"] > 0.9 * delta:
                        delta = min(delta * grow, delta_max)
                    break
                delta *= shrink
            if not accepted:
                if verbose:
                    print(f"    stalled: no acceptable step at delta <= {delta:.2e}")
                break

            # --- optional lambda block (step 4 of Algorithm 1) ---
            if self.lam is not None:
                a_l = self.lam.solve(a)
                W_l = float(self.m.W(a_l))
                if W_l <= W:                       # keep monotone descent
                    a, W = a_l, W_l

            rec = {"iter": k, "W": W, "ratio": ratio, "delta": delta,
                   "step": info["step_norm"], "rel_step": info["rel_step"],
                   "stat": stat, "min_L_over_Jk": info["min_L_over_Jk"],
                   "solver": info["solver"]}
            hist.append(rec)
            if verbose:
                dW = (hist[-2]["W"] - W) / abs(W) if len(hist) > 1 else np.nan
                print(f"{'':>3} {W:>15.8e} {dW:>11.3e} {ratio:>7.3f} {delta:>9.2e} "
                      f"{info['step_norm']:>9.2e} {'':>9} "
                      f"{info['min_L_over_Jk']:>7.3f} {info['solver']:>8}")

        return a, hist
