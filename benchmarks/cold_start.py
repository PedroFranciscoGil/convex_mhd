"""Cold-start robustness benchmark (M1).

Converge Algorithm 1 from a genuine cold start -- DESC's boundary-to-axis
linear interpolation, which knows nothing about the solution -- and compare the
converged energy against the reference DESC equilibrium.

The Ciarlet-Necas volume bound is taken from the cold map's OWN int det F.  For
any bijective map this equals the boundary-enclosed volume, so the solver uses
no knowledge of the answer; the two volumes agreeing to 1.00000 (printed) is the
check that this holds.

Run:  python benchmarks/cold_start.py [CASE ...]      (default: SOLOVEV DSHAPE)
"""

import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import desc.examples
from desc.grid import QuadratureGrid

from convex_mhd.energy import EnergyModel
from convex_mhd.operators import NodeOperators
from convex_mhd.outer import ProxLinearSolver, force_residual


def run_case(case, max_iter=60):
    eq = desc.examples.get(case)
    grid = QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    m = EnergyModel(eq, grid)
    ops = NodeOperators(eq, grid)
    w = np.asarray(m.w)

    a_eq = np.asarray(m.from_eq(eq))
    cold = eq.copy()
    cold.set_initial_guess()                      # boundary-to-axis interpolation
    a0 = np.asarray(m.pack(cold.R_lmn, cold.Z_lmn, cold.L_lmn))

    minJ = float(np.min(m.geometry(a0)["J"]))
    V_cold = float(np.sum(w * np.asarray(m.geometry(a0)["J"])))
    V_eq = float(np.sum(w * np.asarray(m.geometry(a_eq)["J"])))
    F_cold = force_residual(eq, m, a0, grid)
    F_desc = force_residual(eq, m, a_eq, grid)

    print(f"\n=== {case} ===")
    print(f"  cold: minJ={minJ:.3e}  V_cold/V_eq={V_cold / V_eq:.5f}  "
          f"<|F|>={F_cold:.3e}   DESC <|F|>={F_desc:.3e}")
    if minJ <= 0:
        print("  cold start folded (det F <= 0); cannot start Algorithm 1 here.")
        return

    sol = ProxLinearSolver(m, ops, eq, grid, volume=V_cold)
    t = time.time()
    a, hist = sol.solve(a0, mu0=1e-6, delta0=0.3, delta_max=2.0, eta=0.05,
                        eta_good=0.5, max_iter=max_iter, tol=1e-7, verbose=False)
    stats = [x["stat"] for x in hist if "stat" in x]
    W0, W1, Weq = float(m.W(a0)), float(m.W(a)), float(m.W(a_eq))
    print(f"  {time.time() - t:.0f}s  steps={len(stats)}")
    print(f"  W:     {W0:.7e} -> {W1:.7e}   (DESC {Weq:.7e}, rel {abs(W1 - Weq) / abs(Weq):.1e})")
    print(f"  <|F|>: {F_cold:.3e} -> {force_residual(eq, m, a, grid):.3e}   (DESC {F_desc:.3e})")
    print(f"  stationarity: {stats[0]:.2e} -> {stats[-1]:.2e}")


if __name__ == "__main__":
    cases = sys.argv[1:] or ["SOLOVEV", "DSHAPE"]
    for c in cases:
        run_case(c)
