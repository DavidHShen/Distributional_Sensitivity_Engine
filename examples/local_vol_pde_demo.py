"""Local volatility PDE demo (distribution blocks from a backward digital PDE).

Run:
  python examples/local_vol_pde_demo.py

Note: this demo uses a finite-difference PDE + finite-difference (m,eta) derivatives.
It is meant as a reference implementation, not an optimized PDE pricer.

References:
- Dupire, B. (1994). Pricing with a smile. Risk, 7(1), 18–20.
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
    LocalVolPDEBlocks,
    dse_price,
    dse_first_derivative,
    dse_mixed_partial_components,
)


@dataclass(frozen=True)
class CarryTauPrimitives:
    """Primitives consistent with LocalVolPDEBlocks(eta_mode="tau").

    a1 = S e^{-q tau}, a2 = -K e^{-r tau}
    eta = tau
    m = ln(S/K) + (r-q) tau
    """

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
        # Minimal second derivatives used for gamma-like examples
        if {var1, var2} == {"S"}:
            return 0.0, 0.0, 0.0, -1.0 / (self.S * self.S)  # d2m/dS2
        if {var1, var2} == {"K"}:
            return 0.0, 0.0, 0.0, 1.0 / (self.K * self.K)   # d2m/dK2
        return 0.0, 0.0, 0.0, 0.0


def main() -> None:
    model_name = "Local volatility (backward PDE digital blocks)"

    prims = CarryTauPrimitives(S=100.0, K=100.0, tau=0.50, r=0.03, q=0.01, D=+1)

    # Example local vol surface: sigma(t,S)=20%*(S/100)^0.1
    def sigma_local(t: float, S: float) -> float:
        return 0.20 * (S / 100.0) ** 0.10

    blocks = LocalVolPDEBlocks(
        S0=prims.S,
        sigma_local=sigma_local,
        r=prims.r,
        q=prims.q,
        eta_mode="tau",   # eta == tau
        nx=301,
        nt=200,
        x_width=6.0,
        fd_m=1e-4,
        fd_eta=1e-4,
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
