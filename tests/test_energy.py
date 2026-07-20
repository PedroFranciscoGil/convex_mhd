"""M0: the energy of Eq. (8) reproduces DESC, and is stationary at equilibrium."""

import numpy as np
import jax
import pytest

from convex_mhd.energy import EnergyModel


def test_jacobian_convention(solovev):
    """det F = R (R_t Z_r - R_r Z_t) equals DESC's sqrt(g), sign +1."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    g = m.geometry(m.from_eq(eq))
    sg = np.asarray(eq.compute("sqrt(g)", grid=grid)["sqrt(g)"])
    J = np.asarray(g["J"])
    assert np.max(np.abs(J - sg)) / np.max(np.abs(sg)) < 1e-13


def test_magnetic_energy_matches_desc(solovev):
    """W_B from Eq. (8) matches DESC's W_B to near machine precision."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    W_B, _ = m.W_parts(m.from_eq(eq))
    ref = float(eq.compute("W_B", grid=grid)["W_B"])
    assert abs(float(W_B) / ref - 1.0) < 1e-9


def test_pressure_energy_matches_desc(solovev):
    """The pressure block equals DESC's W_p, which is -int p dV."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    _, W_p = m.W_parts(m.from_eq(eq))
    ref = float(eq.compute("W_p", grid=grid)["W_p"])
    assert abs(float(W_p) / ref - 1.0) < 1e-9


def test_lambda_is_required_not_optional(solovev):
    """The bare m=(0,iota,1) misprices W_B by ~3%: DESC equilibria carry lambda."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    a = m.from_eq(eq)
    g = m.geometry(a)
    bare = ((m.iota * g["R_t"] + g["R_z"]) ** 2 + g["R"] ** 2
            + (m.iota * g["Z_t"] + g["Z_z"]) ** 2)
    W_bare = float(np.sum(np.asarray(m.w * m.c * bare / g["J"])))
    ref = float(eq.compute("W_B", grid=grid)["W_B"])
    assert abs(W_bare / ref - 1.0) > 0.01
    assert np.max(np.abs(np.asarray(g["L_t"]))) > 0.1


def _nullspace_grad(m, eq, fn, a0, nullspace):
    g = np.asarray(jax.grad(fn)(a0))
    tot = 0.0
    for basis, off, n, key in ((eq.R_basis, 0, m.nR, "R"), (eq.Z_basis, m.nR, m.nZ, "Z")):
        ns = nullspace(eq, basis, key)
        tot += np.linalg.norm(ns.T @ g[off:off + n]) ** 2
    return np.sqrt(tot)


def test_pressure_sign_is_negative(solovev, boundary_nullspace):
    """Stationarity at the DESC equilibrium selects -int p J over +int p J.

    Guards the correction to Eq. (4) recorded in docs/conventions.md.
    """
    eq, grid = solovev
    # Build both conventions explicitly rather than leaning on the default.
    m_minus = EnergyModel(eq, grid, pressure_sign=-1.0)
    m_plus = EnergyModel(eq, grid, pressure_sign=+1.0)
    a0 = m_minus.from_eq(eq)
    minus = _nullspace_grad(m_minus, eq, m_minus.W, a0, boundary_nullspace)
    plus = _nullspace_grad(m_plus, eq, m_plus.W, a0, boundary_nullspace)
    assert minus < plus / 20.0


def test_energy_is_stationary_at_equilibrium(solovev, boundary_nullspace):
    """delta W = 0 along boundary-preserving variations (Eq. 5) at equilibrium."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    a0 = m.from_eq(eq)
    g = _nullspace_grad(m, eq, m.W, a0, boundary_nullspace)
    raw = np.linalg.norm(np.asarray(m.grad(a0)))
    assert g / raw < 1e-6


def test_autodiff_matches_finite_difference(solovev):
    """jax.grad(W) agrees with a central difference along a random direction."""
    eq, grid = solovev
    m = EnergyModel(eq, grid)
    a0 = m.from_eq(eq)
    rng = np.random.default_rng(0)
    d = rng.standard_normal(a0.shape)
    d = d / np.linalg.norm(d)
    h = 1e-6
    fd = (float(m.W(a0 + h * d)) - float(m.W(a0 - h * d))) / (2 * h)
    ad = float(np.dot(np.asarray(m.grad(a0)), d))
    assert abs(fd - ad) / abs(ad) < 1e-6
