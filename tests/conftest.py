import sys, pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import desc.examples
from desc.grid import QuadratureGrid, LinearGrid
from desc.transform import Transform


@pytest.fixture(scope="session")
def solovev():
    eq = desc.examples.get("SOLOVEV")
    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    return eq, grid


@pytest.fixture(scope="session")
def boundary_nullspace():
    """Null space of the rho=1 evaluation map, per basis: variations obeying A4."""
    cache = {}

    def get(eq, basis, key):
        if key not in cache:
            bgrid = LinearGrid(rho=np.array([1.0]), M=2 * eq.M + 2, N=2 * eq.N + 2, NFP=eq.NFP)
            A = np.asarray(Transform(bgrid, basis, derivs=0, build=True,
                                     method="direct1").matrices["direct1"][0][0][0])
            _, s, Vt = np.linalg.svd(A)
            cache[key] = Vt[np.sum(s > 1e-10 * s.max()):].T
        return cache[key]

    return get
