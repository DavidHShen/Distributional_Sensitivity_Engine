"""General MC demo using likelihood-ratio (score function) blocks.

This demo implements a terminal lognormal simulator that returns score-function
terms w.r.t. eta (here, eta=sigma). The engine uses those scores to construct
(m,eta)-derivatives of the distributional blocks.

Run:
  python examples/general_mc_lr_malliavin_demo.py

References:
- Glynn, P. W. (1990). Likelihood ratio gradient estimation: An overview. In Proc. Winter Simulation Conference.
- Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L., & Touzi, N. (1999). Applications of Malliavin calculus to Monte Carlo methods in finance. Finance and Stochastics, 3, 391–412.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np

# Allow running without installing the package
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from distributional_sensitivity_engine import (
    GeneralMCLRBlocks,
    GeneralMCSimulator,
    dse_price,
    dse_first_derivative,
    dse_mixed_partial_components,
)


@dataclass(frozen=True)
class SigmaPrimitives:
    """Primitives with eta=sigma for GeneralMCLRBlocks.

    a1 = S e^{-q tau}, a2 = -K e^{-r tau}
    eta = sigma
    m = ln(S/K) + (r-q) tau
    """

    S: float
    K: float
    tau: float
    r: float
    q: float
    sigma: float
    D: int = +1

    def primitives(self):
        a1 = self.S * exp(-self.q * self.tau)
        a2 = -self.K * exp(-self.r * self.tau)
        eta = self.sigma
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
            return -self.q * a1, -self.r * a2, 0.0, (self.r - self.q)
        if var == "sigma":
            return 0.0, 0.0, 1.0, 0.0
        raise ValueError(f"Unsupported var={var}")

    def d2_primitives(self, var1: str, var2: str):
        if {var1, var2} == {"S"}:
            return 0.0, 0.0, 0.0, -1.0 / (self.S * self.S)
        return 0.0, 0.0, 0.0, 0.0


class LognormalLRSimulator(GeneralMCSimulator):
    """Terminal lognormal simulator with LR scores wrt eta=sigma.

    Under Q:
      log S_T ~ N(logS0 + (r-q - 0.5 sigma^2) tau,  sigma^2 tau)
    Under Q+:
      log S_T ~ N(logS0 + (r-q + 0.5 sigma^2) tau,  sigma^2 tau)

    Returns a dict containing:
      - S_T
      - score_eta = d/dsigma log f
      - score_eta_eta = d^2/dsigma^2 log f
    """

    def __init__(self, S0: float, r: float, q: float, tau: float):
        self.S0 = S0
        self.r = r
        self.q = q
        self.tau = tau

    def simulate(self, measure: str, m: float, eta: float, n_paths: int, n_steps: int, seed: int):
        rng = np.random.default_rng(seed)
        sigma = float(eta)
        tau = float(self.tau)
        logS0 = log(self.S0)

        is_qplus = (measure.upper() in ("Q+", "QP", "QPLUS", "Q_PLUS"))
        b = +0.5 if is_qplus else -0.5
        mu = logS0 + (self.r - self.q) * tau + b * (sigma * sigma) * tau
        v = (sigma * sigma) * tau
        std = sqrt(v)

        Z = rng.standard_normal(n_paths)
        x = mu + std * Z
        ST = np.exp(x)

        y = x - mu
        s = +1.0 if is_qplus else -1.0

        score = (-1.0 / sigma) + (s * y / sigma) + (y * y) / (sigma**3 * tau)
        score2 = (-tau) + (1.0 - 3.0 * s * y) / (sigma**2) - 3.0 * (y * y) / (sigma**4 * tau)

        return {"S_T": ST, "score_eta": score, "score_eta_eta": score2}


def main() -> None:
    model_name = "General MC (likelihood-ratio / score function; Malliavin-ready hooks)"

    prims = SigmaPrimitives(S=100.0, K=100.0, tau=0.75, r=0.02, q=0.00, sigma=0.25, D=+1)
    carry = (prims.r - prims.q) * prims.tau

    sim = LognormalLRSimulator(S0=prims.S, r=prims.r, q=prims.q, tau=prims.tau)

    blocks = GeneralMCLRBlocks(
        S0=prims.S,
        carry=carry,
        simulator=sim,
        n_paths=80000,
        n_steps=1,
        seed=2026,
        h_m=0.03,
        smooth_phi=True,
        fd_eta=1e-4,
    )

    price = dse_price(prims, blocks)
    delta = dse_first_derivative(prims, "S", blocks)
    vega = dse_first_derivative(prims, "sigma", blocks)

    # Example mixed partial (vanna-style): d^2 V / (dS d sigma)
    vanna = dse_mixed_partial_components(prims, "S", "sigma", blocks)

    print(f"Model: {model_name}")
    print(f"Price: {price:.10f}")
    print(f"Delta: {delta:.10f}")
    print(f"Vega (d/dsigma): {vega:.10f}")
    print(f"d2V/(dS d sigma): {vanna['total']:.10f}")
    print(f"  dot      : {vanna['dot']:.10f}")
    print(f"  P2_cross : {vanna['P2_cross']:.10f}")
    print(f"  P2_non   : {vanna['P2_non']:.10f}")


if __name__ == "__main__":
    main()
