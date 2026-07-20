"""Node-operator kernels for the Proposal-A subproblem (Eq. 14).

Two objects per outer iterate, both cheap because of the structure noted in the
proposal: u is *linear* in (R, Z) at fixed lambda, and the determinant model
L_k is *affine*.  Assembling them as explicit matrices is what turns Eq. (14)
into a sparse SOCP with one rotated cone per collocation node.

  u_q(a)     = U a                     (3 Nq x n), Eq. (7) with m(lambda)
  L_{k,q}(a) = J_k + d_k . (a - a_k)   (Eq. 12 via the cofactor form Eq. 11)

The cofactor field of Eq. (11) involves no toroidal derivatives -- the
"structural gift of the zeta-undisplaced parametrization" of Appendix B -- so
the d_k rows touch only the value, rho- and theta-derivative blocks.
"""

import numpy as np
import jax.numpy as jnp
from desc.transform import Transform

_DERIVS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def design_matrices(grid, basis):
    """Dense (nodes x modes) evaluation matrices for value, d/drho, d/dtheta, d/dzeta."""
    t = Transform(grid, basis, derivs=_DERIVS, build=True, method="direct1")
    M = t.matrices["direct1"]
    return {"val": np.asarray(M[0][0][0]), "r": np.asarray(M[1][0][0]),
            "t": np.asarray(M[0][1][0]), "z": np.asarray(M[0][0][1])}


class NodeOperators:
    """Explicit linear/affine node operators on a fixed grid and basis set."""

    def __init__(self, eq, grid):
        self.grid = grid
        self.AR = design_matrices(grid, eq.R_basis)
        self.AZ = design_matrices(grid, eq.Z_basis)
        self.AL = design_matrices(grid, eq.L_basis)
        self.nR, self.nZ, self.nL = (eq.R_basis.num_modes, eq.Z_basis.num_modes,
                                     eq.L_basis.num_modes)
        self.n = self.nR + self.nZ + self.nL
        self.nq = grid.num_nodes

    def _blocks(self, R_part, Z_part):
        """Assemble a (nq x n) operator from an R-block and a Z-block."""
        return np.hstack([R_part, Z_part, np.zeros((self.nq, self.nL))])

    # -- u operator (linear in R,Z at fixed lambda) ------------------------
    def u_matrix(self, m_t, m_z):
        """Stacked (3 nq x n) matrix U with u = U a, for the given m(lambda) fields."""
        mt, mz = m_t[:, None], m_z[:, None]
        Ux = self._blocks(mt * self.AR["t"] + mz * self.AR["z"], np.zeros((self.nq, self.nZ)))
        Uy = self._blocks(mz * self.AR["val"], np.zeros((self.nq, self.nZ)))
        Uz = self._blocks(np.zeros((self.nq, self.nR)), mt * self.AZ["t"] + mz * self.AZ["z"])
        return np.stack([Ux, Uy, Uz])          # (3, nq, n)

    # -- affine determinant model (Eq. 11/12) ------------------------------
    def detF_jacobian(self, geom):
        """Rows d_k of dJ/da at the iterate, from the cofactor form of Eq. (11).

        delta J = tau dR + R (Z_r dR_t + R_t dZ_r - Z_t dR_r - R_r dZ_t)
        """
        R, tau = np.asarray(geom["R"]), np.asarray(geom["tau"])
        R_r, R_t = np.asarray(geom["R_r"]), np.asarray(geom["R_t"])
        Z_r, Z_t = np.asarray(geom["Z_r"]), np.asarray(geom["Z_t"])
        c = R[:, None]
        Rblk = (tau[:, None] * self.AR["val"]
                + c * (Z_r[:, None] * self.AR["t"] - Z_t[:, None] * self.AR["r"]))
        Zblk = c * (R_t[:, None] * self.AZ["r"] - R_r[:, None] * self.AZ["t"])
        return self._blocks(Rblk, Zblk)        # (nq, n)

    def L_affine(self, geom, a_k):
        """Return (D, c) with L_k(a) = D a + c, the affine model of Eq. (12)."""
        D = self.detF_jacobian(geom)
        J_k = np.asarray(geom["J"])
        return D, J_k - D @ np.asarray(a_k)

    def L_eval(self, geom, a_k, a):
        D, c = self.L_affine(geom, a_k)
        return D @ np.asarray(a) + c
