"""The lambda block of Sec. 6.6: a single linear elliptic solve.

At fixed (R, Z) the energy is a convex quadratic in lambda, because det F is
lambda-independent and u = F m(lambda) is *affine* in the lambda coefficients:

    m_t = iota - L_z,   m_z = 1 + L_t
    u   = u0 + G aL

so minimizing sum_q w_q c_q ||u_q||^2 / J_q is a weighted linear least-squares
problem -- precisely VMEC's lambda equation.  The pressure term drops out
entirely (it depends only on J), which is why this block is a linear solve and
not another cone program.

Constant lambda modes are a null direction (only derivatives of lambda enter),
so the normal equations are rank-deficient by design; `lstsq` handles it.
"""

import numpy as np

from .operators import design_matrices


class LambdaBlock:
    """Exact convex minimizer over lambda at frozen (R, Z)."""

    def __init__(self, model, eq, grid):
        self.m = model
        self.AL = design_matrices(grid, eq.L_basis)
        self.nRZ = model.nR + model.nZ

    def solve(self, a_k, rcond=1e-10):
        """Return a_k with its lambda block replaced by the exact minimizer."""
        m = self.m
        g = m.geometry(a_k)
        R, R_t, R_z = (np.asarray(g[k]) for k in ("R", "R_t", "R_z"))
        Z_t, Z_z = np.asarray(g["Z_t"]), np.asarray(g["Z_z"])
        J = np.asarray(g["J"])
        iota = np.asarray(m.iota)
        At, Az = self.AL["t"], self.AL["z"]

        # u = u0 + G aL, per component
        u0 = np.stack([iota * R_t + R_z, R, iota * Z_t + Z_z])
        G = np.stack([
            -R_t[:, None] * Az + R_z[:, None] * At,
            R[:, None] * At,
            -Z_t[:, None] * Az + Z_z[:, None] * At,
        ])

        # weighted normal equations, sqrt-weighted for conditioning
        sw = np.sqrt(np.asarray(m.w) * np.asarray(m.c) / J)
        A = np.concatenate([sw[:, None] * G[i] for i in range(3)], axis=0)
        b = -np.concatenate([sw * u0[i] for i in range(3)])
        aL, *_ = np.linalg.lstsq(A, b, rcond=rcond)

        return np.concatenate([np.asarray(a_k)[:self.nRZ], aL])
