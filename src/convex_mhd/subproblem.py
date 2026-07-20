"""The Proposal-A conic subproblem, Eq. (14), assembled in CVXPY.

    min_{a,s}  sum_q w_q [ c_q s_q - p_q L_{k,q}(a) ] + (mu/2)||a - a_k||_S^2
    s.t.       ||u_q(a)||^2 <= s_q L_{k,q}(a)      (rotated SOC, one per node)
               L_{k,q}(a)   >= kappa J_{k,q}       (relative Jacobian floor)
               boundary coefficients fixed          (A4)

Three design decisions, none of them forced by the proposal:

1.  **The SOCP solves for (R, Z) only; lambda gets its own block.**  u is linear
    in (R, Z) *at fixed lambda*, and det F is lambda-independent, so Sec. 6.6's
    biconvex split is the natural one.  Phase 0 found lambda is not optional
    (max|lam_t| = 0.45 even on axisymmetric SOLOVEV), so the lambda block is
    live from M1 onward, not deferred to 3D.  See `lambda_block.py`.

2.  **The prox term is the gauge fix.**  Sec. 6.6 offers spectral condensation
    constraints or a Tikhonov penalty on tangential motion; the prox term
    ||a - a_k||_S^2 already *is* that penalty, selecting the solution nearest
    the previous iterate along the gauge direction.  So no extra constraints
    are imposed, and gauge conditioning is watched empirically instead.

3.  **mu > 0 is required for boundedness, not merely for trust-region control.**
    With the corrected -p J sign (docs/conventions.md), *both* objective terms
    decrease as L grows: the cone gives s = ||u||^2/L, and -p L falls linearly.
    The true det F cannot run away (its integral is the volume, fixed by A4),
    but the *affine model* L_k carries no such constraint, so the linearized
    objective is unbounded below at mu = 0.  The quadratic prox dominates the
    linear escape and restores boundedness.  A zero-damping step is therefore
    not just aggressive here -- it is ill-posed.
"""

import numpy as np
import cvxpy as cp
from desc.grid import LinearGrid
from desc.transform import Transform


def boundary_matrix(eq, basis, n_extra=2, full_rank=True, tol=1e-10):
    """Row-space basis of the rho = 1 evaluation map; fixing it pins A4.

    The raw evaluation matrix is massively rank deficient -- on SOLOVEV it is
    265 x 91 with rank 13, because far more boundary collocation points are
    sampled than there are boundary degrees of freedom.  Passing all 265 rows
    to the solver imposes ~250 redundant equalities, which makes the
    interior-point KKT system singular and stalls it with InsufficientProgress.
    Returning an orthonormal basis of the row space imposes exactly the same
    subspace with full row rank.
    """
    bgrid = LinearGrid(rho=np.array([1.0]), M=2 * eq.M + n_extra,
                       N=2 * eq.N + n_extra, NFP=eq.NFP)
    t = Transform(bgrid, basis, derivs=0, build=True, method="direct1")
    A = np.asarray(t.matrices["direct1"][0][0][0])
    if not full_rank:
        return A
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[: int(np.sum(s > tol * s.max()))]


class ConicSubproblem:
    """Builds and solves Eq. (14) about a given iterate."""

    def __init__(self, model, ops, eq, kappa=0.2, volume=None):
        self.m, self.ops, self.eq, self.kappa = model, ops, eq, kappa
        # Ciarlet-Necas bound: int det F <= vol(image).  See `solve`.
        self.volume = volume
        self.nRZ = model.nR + model.nZ
        self.AbR = boundary_matrix(eq, eq.R_basis)
        self.AbZ = boundary_matrix(eq, eq.Z_basis)
        self.w = np.asarray(model.w)
        self.c = np.asarray(model.c)
        self.p = np.asarray(model.p)

    # CLARABEL is the accurate reference backend and respects the hard trust
    # region to solver tolerance; SCS overshoots it (returns "optimal_inaccurate"
    # points up to 7x past the radius) so it is only a fallback, gated by the
    # feasibility check in `solve`.  The trust radius also removes CLARABEL's old
    # failure mode -- with the step capped it never collapses to the degenerate
    # all-cones-tight regime that used to stall the interior-point method.
    SOLVERS = ("CLARABEL", "SCS")

    def solve(self, a_k, mu, delta=None, solver=None, verbose=False, **kw):
        """One convex step about ``a_k`` with damping ``mu``. Returns (a_new, info).

        Solved in nondimensional variables.  The raw problem mixes c ~ 1e4,
        coefficients ~ 1 and W ~ 1e6, which makes interior-point solvers stall
        at iteration 0; here the step, the slack and the Jacobian are each
        referred to their own scale so every quantity is O(1):

            a = a_k + a_scale z,   s = s_ref s_hat,   L_hat = L / J_ref

        with s_ref = u2_ref/J_ref, which makes the rotated cone scale-free:
        ||u||^2 <= s L  becomes  ||u/sqrt(u2_ref)||^2 <= s_hat L_hat.
        """
        m, ops = self.m, self.ops
        a_k = np.asarray(a_k)
        aRZ_k, aL_k = a_k[:self.nRZ], a_k[self.nRZ:]

        g = m.geometry(a_k)
        J_k = np.asarray(g["J"])
        if np.min(J_k) <= 0:
            raise ValueError(f"iterate has non-positive Jacobian: min J = {J_k.min():.3e}")

        # linear/affine node operators, restricted to the (R,Z) block
        U = ops.u_matrix(np.asarray(m.iota - g["L_z"]),
                         np.asarray(1.0 + g["L_t"]))[:, :, :self.nRZ]
        D, _ = ops.L_affine(g, a_k)
        D = D[:, :self.nRZ]

        u_k = np.einsum("cqn,n->cq", U, aRZ_k)
        u2_k = np.asarray(g["u2"])

        # Per-node normalization: refer u, s and L at each node to that node's
        # OWN current value, so the iterate sits at (||u_hat||, s_hat, L_hat) =
        # (1, 1, 1) everywhere.  A single global scale is not enough here --
        # c(rho) ~ rho^2 spans 3.4 to 4e4 across the grid and J spans 0.057 to
        # 7.4, so a globally-scaled slack block still carries a ~1e4 dynamic
        # range and the interior-point KKT system stalls.
        W_ref = abs(float(m.W(a_k)))
        u_norm = np.sqrt(u2_k)                       # per node
        Dh = D / J_k[:, None]                        # d(L/J_k)/da

        # The spectrally scaled prox metric ||.||_S of Sec. 6.2, realized as
        # column equilibration of the determinant operator.  This is not
        # cosmetic: cond(D) ~ 7e17, and ||Dh||_2 is set by a few very stiff
        # directions (near-axis, high-order Zernike).  With S = I the step unit
        # collapses in *every* direction to suit those few, leaving u
        # numerically frozen (a_scale*U/|u| ~ 3e-5) and the solver stalling.
        # Weighting each mode by how strongly it moves the Jacobian makes the
        # trust region anisotropic and the KKT system well conditioned --
        # exactly VMEC's diagonal preconditioner in convex clothing.
        col = np.linalg.norm(Dh, axis=0)
        col = np.maximum(col, 1e-12 * col.max())     # S^{1/2}, diagonal
        Dh_s = Dh / col[None, :]
        U_s = U / col[None, None, :]
        a_scale = 1.0 / float(np.linalg.norm(Dh_s, 2))

        z = cp.Variable(self.nRZ)          # step in S-scaled units
        sh = cp.Variable(self.ops.nq)      # slack in units of u2_k/J_k

        Lh = 1.0 + a_scale * (Dh_s @ z)                      # = L / J_k
        uh = [(u_k[i] + a_scale * (U_s[i] @ z)) / u_norm for i in range(3)]

        # ||u||^2 <= s L  <=>  ||u_hat||^2 <= s_hat L_hat  under this scaling
        alpha = self.w * self.c * u2_k / J_k / W_ref         # sums to W_B/W_ref

        if m.closure == "adiabatic":
            # Remark 1: V' is linear in L, and t^(1-gamma) is convex decreasing
            # for gamma > 1, so convex-nonincreasing-of-affine stays convex.
            # CVXPY canonicalizes the negative power to power cones.
            g_ = m.gamma
            Vp = cp.multiply(np.asarray(m.w_ang) * J_k, Lh) @ np.asarray(m.shell)
            thermal = (np.asarray(m.w_rad) * np.asarray(m.mass) ** g_) @ \
                cp.power(Vp, 1.0 - g_) / (g_ - 1.0)
            obj = alpha @ sh + thermal / W_ref + 0.5 * mu * cp.sum_squares(z)
        else:
            beta = self.w * self.p * J_k / W_ref
            obj = alpha @ sh - beta @ Lh + 0.5 * mu * cp.sum_squares(z)

        # The boundary constraint must act on the PHYSICAL step, not on z.  The
        # physical step is a_scale * z / col (column-equilibrated), so imposing
        # AbR @ z = 0 leaves AbR @ (z/col) != 0 -- every step then silently moves
        # the boundary and descends the huge boundary-direction gradient (~1e6),
        # which reads as faithful energy descent (ratio ~ 1) while actually
        # violating A4.  Scaling the constraint rows by 1/col fixes it.
        colR, colZ = col[:m.nR], col[m.nR:]
        cons = [
            cp.SOC(sh + Lh, cp.vstack([2 * uh[0], 2 * uh[1], 2 * uh[2], sh - Lh]), axis=0),
            Lh >= self.kappa,
            (self.AbR / colR[None, :]) @ z[:m.nR] == 0,
            (self.AbZ / colZ[None, :]) @ z[m.nR:] == 0,
        ]

        if delta is not None:
            # Hard trust-region radius in the normalized step metric.  The prox
            # term alone is a soft (Levenberg) damping: with mu halving on every
            # accepted step and no floor, the step grows unboundedly and can
            # leave the local basin even when started at the equilibrium -- the
            # ratio test cannot catch it, because the model stays faithful and W
            # genuinely does decrease out there.  A hard cap makes the
            # trust-region argument of Sec. 6.5 actually hold.
            cons.append(cp.norm(z, 2) <= delta)

        if self.volume is not None:
            # Ciarlet-Necas condition.  det F > 0 buys only *local*
            # invertibility; nothing in the polyconvex formulation forbids the
            # map from becoming globally non-injective, and the interior
            # surfaces then balloon past the pinned boundary.  Measured: pure
            # descent inflates int det F by 65% while every det F stays
            # positive, driving W down and the force residual up by 1e3.
            # int det F <= vol(image) restores global injectivity, and because
            # det F is lifted to L it is a single *linear* constraint.
            cons.append(self.w @ cp.multiply(J_k, Lh) <= self.volume)

        prob = cp.Problem(cp.Minimize(obj), cons)
        tried = [solver] if solver is not None else list(self.SOLVERS)
        errors = []
        for sv in tried:
            try:
                prob.solve(solver=sv, verbose=verbose, **kw)
            except cp.error.SolverError as e:
                errors.append(f"{sv}: {str(e)[:50]}")
                continue
            if z.value is None:
                errors.append(f"{sv}: {prob.status}")
                continue
            # Reject a solution that violates the hard trust region: SCS returns
            # "optimal_inaccurate" points that overshoot the radius by up to 7x,
            # which would let the outer loop take a step it never sanctioned.
            # CLARABEL respects it to the solver tolerance.
            if delta is not None and np.linalg.norm(z.value) > delta * 1.05:
                errors.append(f"{sv}: trust-region violated "
                              f"(|z|={np.linalg.norm(z.value):.3f} > {delta:.3f})")
                continue
            break
        else:
            raise RuntimeError(f"subproblem failed on all solvers: {'; '.join(errors)}")

        step = a_scale * z.value / col
        a_new = np.concatenate([aRZ_k + step, np.asarray(aL_k)])

        # model value in physical units, with and without the prox penalty
        model_new = (float(prob.value) - 0.5 * mu * float(np.sum(z.value**2))) * W_ref
        # At a_k the affine model is exact (L_k(a_k) = J_k), so the model energy
        # there is just the true energy -- valid for either closure.  (An earlier
        # hardcoded W_B - int pJ form was wrong in adiabatic mode and made
        # pred_decrease negative, so the ratio test rejected every good step.)
        model_at_ak = float(m.W(a_k))
        return a_new, {
            "status": prob.status,
            "solver": sv,
            "model_obj": model_new,
            "model_at_ak": model_at_ak,
            "pred_decrease": model_at_ak - model_new,
            "step_norm": float(np.linalg.norm(step)),
            "rel_step": float(np.linalg.norm(z.value)),
            "min_L_over_Jk": float(np.min(1.0 + a_scale * (Dh_s @ z.value))),
        }
