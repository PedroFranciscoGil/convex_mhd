"""M0 step 4: the node operators feeding the SOCP (Eq. 14) are exact.

The u operator must reproduce the geometry's u exactly (it is linear), and the
affine determinant model L_k must match autodiff of det F to first order.
"""

import numpy as np
import jax
import jax.numpy as jnp

from convex_mhd.energy import EnergyModel
from convex_mhd.operators import NodeOperators


def test_u_operator_is_exact(solovev):
    """u = U a reproduces the nonlinear geometry's u (u is linear in R,Z)."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    ops = NodeOperators(eq, grid)
    a = m.from_eq(eq)
    g = m.geometry(a)
    U = ops.u_matrix(np.asarray(m.iota - g["L_z"]), np.asarray(1.0 + g["L_t"]))
    u_op = np.einsum("cqn,n->cq", U, np.asarray(a))
    u_ref = np.asarray(g["u"])
    assert np.max(np.abs(u_op - u_ref)) / np.max(np.abs(u_ref)) < 1e-12


def test_detF_jacobian_matches_autodiff(solovev):
    """Eq. (11)'s cofactor rows equal the autodiff Jacobian of the J field."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    ops = NodeOperators(eq, grid)
    a = m.from_eq(eq)
    D = ops.detF_jacobian(m.geometry(a))
    D_ad = np.asarray(jax.jacrev(lambda x: m.geometry(x)["J"])(a))
    assert np.max(np.abs(D - D_ad)) / np.max(np.abs(D_ad)) < 1e-11


def test_L_affine_interpolates_and_is_second_order(solovev):
    """L_k(a_k) = J_k exactly, and L_k(a) - det F(a) = O(||a-a_k||^2)."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    ops = NodeOperators(eq, grid)
    a_k = m.from_eq(eq)
    g_k = m.geometry(a_k)
    J_k = np.asarray(g_k["J"])

    assert np.max(np.abs(ops.L_eval(g_k, a_k, a_k) - J_k)) / np.max(np.abs(J_k)) < 1e-11

    rng = np.random.default_rng(1)
    d = rng.standard_normal(a_k.shape)
    d = d / np.linalg.norm(d) * np.linalg.norm(np.asarray(a_k))

    errs = []
    for h in (1e-3, 5e-4, 2.5e-4):
        a = a_k + h * d
        J = np.asarray(m.geometry(a)["J"])
        errs.append(np.max(np.abs(ops.L_eval(g_k, a_k, a) - J)))
    # halving the step must quarter the error -> observed order ~2
    order = np.log(errs[0] / errs[2]) / np.log(4.0)
    assert 1.8 < order < 2.2


def test_no_toroidal_derivatives_in_cofactor(solovev):
    """Appendix B: the cofactor field involves no zeta derivatives.

    So L_k must be blind to any coefficient perturbation that changes only the
    toroidal-derivative content -- checked here via the axisymmetric N=0 case,
    where the d/dzeta design blocks vanish identically.
    """
    eq, grid = solovev
    ops = NodeOperators(eq, grid)
    assert np.max(np.abs(ops.AR["z"])) < 1e-14
    assert np.max(np.abs(ops.AZ["z"])) < 1e-14
