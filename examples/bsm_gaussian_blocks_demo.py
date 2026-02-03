"""Deterministic-vol BSM demo (Gaussian blocks).

Run from repo root:
  python examples/bsm_gaussian_blocks_demo.py

Or after editable install:
  python -m distributional_sensitivity_engine

References:
- Black, F., and Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654.
- Merton, R. C. (1973). Theory of rational option pricing. Bell Journal of Economics and Management Science, 4(1), 141–183.
"""

from __future__ import annotations

import os
import sys

# Allow running without installing the package
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from distributional_sensitivity_engine import (
    BSMGaussianBlocks,
    BSMPrimitives,
    dse_price,
    dse_first_derivative,
    dse_mixed_partial_components,
)


def main() -> None:
    model_name = "Black–Scholes–Merton (deterministic vol) — Gaussian blocks"
    prims = BSMPrimitives(S=100.0, K=100.0, tau=0.5, r=0.03, q=0.01, sigma=0.20, D=+1)
    blocks = BSMGaussianBlocks()

    price = dse_price(prims, blocks)
    delta = dse_first_derivative(prims, "S", blocks)
    vega = dse_first_derivative(prims, "sigma", blocks)
    gamma = dse_mixed_partial_components(prims, "S", "S", blocks)

    print(f"Model: {model_name}")
    print(f"  price : {price:.10f}")
    print(f"  delta : {delta:.10f}")
    print(f"  vega  : {vega:.10f}")
    print(f"  gamma : {gamma['total']:.10f}")
    print(f"    dot      : {gamma['dot']:.10f}")
    print(f"    P2_cross : {gamma['P2_cross']:.10f}")
    print(f"    P2_non   : {gamma['P2_non']:.10f}")


if __name__ == "__main__":
    main()
