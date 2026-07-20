"""The MHD energy W(a) of Eq. (8), evaluated on DESC's Fourier-Zernike transforms.

Conventions pinned against DESC 0.17.2 in Phase 0 (see `docs/conventions.md`):
  * det F = R (R_t Z_r - R_r Z_t)  equals DESC's sqrt(g) with sign +1.
  * DESC's W_B is SI: int |B|^2/(2 mu0) dV.  The proposal uses mu0 = 1.
  * DESC's stock equilibria carry a nonzero stream function lambda, so the
    field-line vector is m(lambda) = (0, iota - lam_z, 1 + lam_t) of Sec. 6.6,
    NOT the bare m = (0, iota, 1) of Eq. (3).  Using the bare form misprices
    W_B by ~3% on SOLOVEV.
  * The pressure term enters as -p*J, NOT the +p det F printed in Eq. (4).
    Measured: at a DESC equilibrium the null-space gradients of W_B and of
    int p J are equal to 5 digits and cancel only under the minus sign
    (DSHAPE: 7.7282e4 vs 7.7282e4 -> residual 0.79, a 2e5 improvement).
    This is the sign the adiabatic closure implies: with the thermal energy
    T = (1/(g-1)) int M^g V'^(1-g) drho of Remark 1, dT/dV' = -p, so the
    linear-in-J surrogate of the true energy carries -p.  Proposition 1 holds
    either way ("regardless of the sign of p, since the second term is
    linear"), so the convexification strategy is untouched.
"""

import jax
import jax.numpy as jnp
import numpy as np
from desc.transform import Transform

jax.config.update("jax_enable_x64", True)

MU_0 = 4e-7 * np.pi

# Derivative orders needed for the geometry: value, d/drho, d/dtheta, d/dzeta.
_DERIVS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


class EnergyModel:
    """W(a) for a fixed grid, basis set and prescribed profiles.

    The unknown is the flat coefficient vector ``a = concat(R_lmn, Z_lmn, L_lmn)``.
    Everything else -- quadrature weights, psi'(rho), iota(rho), p(rho) -- is
    prescribed data held on the grid (assumptions A2-A3).
    """

    def __init__(self, eq, grid, mu0=MU_0, pressure_sign=-1.0):
        self.grid = grid
        self.mu0 = mu0
        self.pressure_sign = pressure_sign

        self._tR = Transform(grid, eq.R_basis, derivs=_DERIVS, build=True)
        self._tZ = Transform(grid, eq.Z_basis, derivs=_DERIVS, build=True)
        self._tL = Transform(grid, eq.L_basis, derivs=_DERIVS, build=True)

        self.nR = eq.R_basis.num_modes
        self.nZ = eq.Z_basis.num_modes
        self.nL = eq.L_basis.num_modes

        rho = grid.nodes[:, 0]
        self.w = jnp.asarray(grid.weights)
        # psi = Psi rho^2 / (2 pi)  =>  psi' = Psi rho / pi
        self.psi_r = jnp.asarray(eq.Psi * rho / np.pi)
        self.iota = jnp.asarray(eq.iota(rho))
        self.p = jnp.asarray(eq.pressure(rho))
        # c(rho) = 1/2 psi'^2, carrying the 1/mu0 so W matches DESC's SI W_B.
        self.c = 0.5 * self.psi_r**2 / mu0

    # -- packing -----------------------------------------------------------
    def pack(self, R_lmn, Z_lmn, L_lmn):
        return jnp.concatenate([jnp.asarray(R_lmn), jnp.asarray(Z_lmn), jnp.asarray(L_lmn)])

    def unpack(self, a):
        i, j = self.nR, self.nR + self.nZ
        return a[:i], a[i:j], a[j:]

    def from_eq(self, eq):
        return self.pack(eq.R_lmn, eq.Z_lmn, eq.L_lmn)

    # -- geometry ----------------------------------------------------------
    def geometry(self, a):
        """Node-wise geometry: the vector u, the Jacobian J, and its pieces."""
        Rc, Zc, Lc = self.unpack(a)
        R = self._tR.transform(Rc, 0, 0, 0)
        R_r = self._tR.transform(Rc, 1, 0, 0)
        R_t = self._tR.transform(Rc, 0, 1, 0)
        R_z = self._tR.transform(Rc, 0, 0, 1)
        Z_r = self._tZ.transform(Zc, 1, 0, 0)
        Z_t = self._tZ.transform(Zc, 0, 1, 0)
        Z_z = self._tZ.transform(Zc, 0, 0, 1)
        L_t = self._tL.transform(Lc, 0, 1, 0)
        L_z = self._tL.transform(Lc, 0, 0, 1)

        tau = R_t * Z_r - R_r * Z_t
        J = R * tau

        # m(lambda) = (0, iota - lam_z, 1 + lam_t);  u = F m.
        m_t = self.iota - L_z
        m_z = 1.0 + L_t
        u = jnp.stack([m_t * R_t + m_z * R_z, m_z * R, m_t * Z_t + m_z * Z_z])

        return {"R": R, "R_r": R_r, "R_t": R_t, "R_z": R_z,
                "Z_r": Z_r, "Z_t": Z_t, "Z_z": Z_z,
                "L_t": L_t, "L_z": L_z,
                "tau": tau, "J": J, "u": u, "u2": jnp.sum(u**2, axis=0)}

    # -- energy ------------------------------------------------------------
    def W_parts(self, a):
        g = self.geometry(a)
        W_B = jnp.sum(self.w * self.c * g["u2"] / g["J"])
        W_p = jnp.sum(self.w * self.p * g["J"])
        return W_B, self.pressure_sign * W_p

    def W(self, a):
        W_B, W_p = self.W_parts(a)
        return W_B + W_p

    def grad(self, a):
        return jax.grad(self.W)(a)
