"""Heston Monte Carlo demo (distribution blocks estimated from MC).

Run:
  python examples/heston_mc_demo.py

Notes:
- The block implementation estimates CDF/PDF derivatives in (m,eta) by a combination of smoothing + finite differences.
- Increase n_paths for stability; use smooth_phi=True for lower variance.

References:
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies, 6(2), 327–343.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import exp, log

# Allow running without installing the package
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from distributional_sensitivity_engine import (
    HestonMCBlocks,
    dse_price,
    dse_first_derivative,
    dse_mixed_partial_components,
)


@dataclass(frozen=True)
class CarryTauPrimitives:
    """Primitives with eta=tau (for MC blocks configured with eta_mode='tau')."""

    S: float
    K: float
    tau: float
    r: float
    q: float
    D: int = +1

    def primitives(self):
        a1 = self.S * exp(-self.q * self.tau)
        a2 = -self.K * exp(-self.r * self.tau)
        eta = self.tau
        m = log(self.S / self.K) + (self.r - self.q) * self.tau
        return a1, a2, eta, m

    def d_primitives(self, var: str):
        a1, a2, eta, m = self.primitives()
        if var == "S":
            return exp(-self.q * self.tau), 0.0, 0.0, 1.0 / self.S
        if var == "K":
            return 0.0, -exp(-self.r * self.tau), 0.0, -1.0 / self.K
        if var == "r":
            return 0.0, -self.tau * a2, 0.0, self.tau
        if var == "q":
            return -self.tau * a1, 0.0, 0.0, -self.tau
        if var == "tau":
            return -self.q * a1, -self.r * a2, 1.0, (self.r - self.q)
        raise ValueError(f"Unsupported var={var}")

    def d2_primitives(self, var1: str, var2: str):
        if {var1, var2} == {"S"}:
            return 0.0, 0.0, 0.0, -1.0 / (self.S * self.S)
        return 0.0, 0.0, 0.0, 0.0


def main() -> None:
    model_name = "Heston (MC distribution blocks)"

    prims = CarryTauPrimitives(S=100.0, K=100.0, tau=1.00, r=0.02, q=0.00, D=+1)

    blocks = HestonMCBlocks(
        S0=prims.S,
        v0=0.04,
        r=prims.r,
        q=prims.q,
        kappa=1.5,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        eta_mode="tau",
        n_paths=15000,
        n_steps=200,
        h_m=0.03,
        smooth_phi=True,
        fd_eta=1e-3,
    )

    price = dse_price(prims, blocks)
    delta = dse_first_derivative(prims, "S", blocks)
    theta = dse_first_derivative(prims, "tau", blocks)
    gamma = dse_mixed_partial_components(prims, "S", "S", blocks)

    print(f"Model: {model_name}")
    print(f"Price: {price:.10f}")
    print(f"Delta: {delta:.10f}")
    print(f"Theta (d/dtau): {theta:.10f}")
    print(f"Gamma (d2/dS2): {gamma['total']:.10f}")
    print(f"  dot      : {gamma['dot']:.10f}")
    print(f"  P2_cross : {gamma['P2_cross']:.10f}")
    print(f"  P2_non   : {gamma['P2_non']:.10f}")


if __name__ == "__main__":
    main()
