"""Entry point for `python -m distributional_sensitivity_engine`.

Prints a small banner and runs a quick BSM Gaussian-block sanity check.
"""

from __future__ import annotations

from . import __version__
from .engine import BSMGaussianBlocks, BSMPrimitives, dse_price, dse_first_derivative


def main() -> None:
    model = "Black–Scholes–Merton (deterministic vol) — Gaussian blocks"
    prims = BSMPrimitives(S=100.0, K=100.0, tau=0.5, r=0.03, q=0.01, sigma=0.2, D=+1)
    blocks = BSMGaussianBlocks()

    price = dse_price(prims, blocks)
    delta = dse_first_derivative(prims, "S", blocks)
    vega = dse_first_derivative(prims, "sigma", blocks)

    print(f"distributional_sensitivity_engine v{__version__}")
    print(f"Model: {model}")
    print(f"Price: {price:.10f}")
    print(f"Delta: {delta:.10f}")
    print(f"Vega : {vega:.10f}")


if __name__ == "__main__":
    main()
