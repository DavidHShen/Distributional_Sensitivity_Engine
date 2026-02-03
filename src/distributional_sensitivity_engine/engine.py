r"""distributional_sensitivity_engine.py

Distributional Sensitivity Engine (DSE) — initial design specification (Paper DSE).

Purpose
-------
This module provides a model-agnostic sensitivity engine for European options whose pricing and
Greeks can be expressed through (i) a small set of deterministic/\mathcal F_t-measurable primitives
and (ii) conditional distribution blocks of the Doléans–Dade log variable E_{t,T} under two measures
Q and Q^+ connected by a Girsanov change of measure.

The design follows the representation in Paper DSE:
  • price as a two-weight contraction,
  • first derivatives as a four-input contraction,
  • mixed partials as a dot-product term plus an explicit correction channel P2,
    computed in a Jacobian/Hessian matrix form suitable for implementation.

Mathematical interface
----------------------
1) Primitives (engine inputs)
   The engine works with the primitive vector
       a := (a1, a2, a3, a4)^T = (a1, a2, η, m)^T,
   where:
     • a1, a2 are deterministic prefactors (e.g., discounted spot and discounted strike),
     • m is the threshold/log-moneyness shift in the event {D E_{t,T} < D m},
     • η is a scalar parameter indexing the relevant conditional law family for E_{t,T}|\mathcal F_t
       (normalization, maturity/variance index, or another model parameter as defined by the user).

   For any scalar parameter x (spot, rate node, volatility parameter, maturity, etc.), the caller must
   supply:
     • first derivatives a_{x} = (a1_x, a2_x, η_x, m_x),
     • second derivatives a_{x1 x2} when mixed partials are requested.

2) Conditional distribution blocks (model plug-in)
   The model dependence enters ONLY through conditional CDF/PDF objects for E_{t,T} under Q and Q^+.
   Using the signed call/put convention D ∈ {+1, -1}, define:
       F1(m;η) := Φ^{Q^+}_{E|\mathcal F_t}(m;η),   F2(m;η) := Φ^{Q}_{E|\mathcal F_t}(m;η),
       f1(m;η) := ϕ^{Q^+}_{E|\mathcal F_t}(m;η),   f2(m;η) := ϕ^{Q}_{E|\mathcal F_t}(m;η).
   The DistributionBlocks protocol specifies:
     • signed CDF evaluations F1, F2 at z=m (depend on D),
     • first partials in (m,η): ∂_m, ∂_η,
     • second partials in (m,η): ∂_{mm}, ∂_{ηη}, and mixed ∂_{ηm} (optionally also ∂_{mη}).

   Note: only F1/F2 depend on the sign D. Derivatives with respect to (m,η) are D-invariant because
   the D=-1 version differs from the D=+1 version by a constant shift (Paper DSE, Section 6).

Core engine formulas (implemented here)
---------------------------------------
Weights:
    w1 := Dig_T^+ = F1(m;η)+(D-1)/2 ,      w2 := Dig_T^- = F2(m;η)+(D-1)/2,
    w3 := a1 ∂_η F1(m;η) + a2 ∂_η F2(m;η),
    w4 := a1 f1(m;η)       + a2 f2(m;η).

Price:
    V_t = w1 a1 + w2 a2.

First derivative (four-input contraction):
    V_{t,x} = w1 a1_x + w2 a2_x + w3 η_x + w4 m_x.

Mixed partials (dot term + correction channel):
    V_{t,(x1,x2)} = \tilde w^T a_{x1 x2} + P2(x1,x2),
    P2(x1,x2) := Σ_{k=1}^4 w_{k,x2} a_{k,x1}.

Matrix/Jacobian form of P2 (implementation target):
    Let V(m;η) be the 2×2 Jacobian of (F1,F2) with respect to (η,m):
        V(m;η) = [[F1_η, F1_m],
                  [F2_η, F2_m]].
    Then P2 admits the decomposition (Paper DSE, Eq. (31) and Appendix B):
        P2 = a12_{x1}^T V a34_{x2} + a12_{x2}^T V a34_{x1}     (cross part)
           + a12^T V_{x2} a34_{x1}                            (non-cross part),
    where a12=(a1,a2)^T and a34=(η,m)^T. The directional derivative V_{x2} is computed via
    Hessian blocks of (F1,F2) and the chain rule through (η,m).

Software architecture
---------------------
• Core engine is model-agnostic:
    - DistributionBlocks defines the required conditional objects.
    - dse_price, dse_first_derivative, dse_mixed_partial compute the contractions.
    - P2_matrix_form computes the correction channel using only primitive first derivatives plus
      Jacobian/Hessian blocks (no explicit differentiation of the weight vector is required).

• Reference specializations:
    - BSMGaussianBlocks provides closed-form normal blocks for the deterministic-volatility BSM limit,
      intended for regression tests and sanity checks.
    - LocalVolPDEBlocks, HestonMCBlocks, and GeneralMCLRBlocks provide illustrative plug-ins for
      PDE/MC settings (NumPy required) where the distribution blocks are computed numerically.

Assumptions and scope
---------------------
• Discounting is treated as exogenous and non-random over [t,T] (\mathcal F_t-measurable discount
  factor), to isolate the distributional-sensitivity mechanism driven by the volatility specification.
• Measure-change feasibility (Novikov / exponential-martingale condition) and distributional
  regularity (existence/differentiability of conditional densities) are assumed as in Paper DSE.
  The engine does not attempt to verify these conditions; it consumes the resulting blocks.

References for included model blocks
-----------------------------------
Some distributions blocks are provided as optional reference implementations. The underlying model
families and estimator techniques are standard; the citations below indicate canonical sources.

- Deterministic-volatility BSM Gaussian blocks: Black and Scholes (1973); Merton (1973).
- Local volatility PDE (Dupire equation): Dupire (1994).
- Stochastic volatility (Heston) Monte Carlo: Heston (1993).
- General Monte Carlo Greeks (likelihood-ratio / Malliavin weights): Glynn (1990); Fournié et al. (1999); Glasserman (2004).

Extensibility
-------------
To support a new model:
  1) implement DistributionBlocks for that model (CDF/PDF blocks and required (m,η)-partials),
  2) provide a primitives object that returns (a1,a2,η,m) and their derivatives with respect to the
     parameters of interest.

This separation is deliberate: the differentiation logic remains fixed, while model-specific effort
is concentrated in the conditional distribution block estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt, pi
from typing import Protocol, Tuple, Dict


# =========================
# Normal PDF/CDF utilities
# =========================

def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ==========================================
# Distribution blocks interface (model plug)
# ==========================================

class DistributionBlocks(Protocol):
    """Conditional distribution blocks for E_{t,T} | F_t under Q and Q+.

    Required objects (Paper DSE Section 5–7):
      - F1(m;eta) := Phi^{Q+}_{E|F_t}(m;eta)  (signed CDF at z=m)
      - F2(m;eta) := Phi^{Q }_{E|F_t}(m;eta)
      - First and second derivatives in (m,eta)

    The paper's Jacobian matrix V(m;eta) (Eq. (30) in Paper DSE) is built from:
      V = [[F1_eta, F1_m],[F2_eta, F2_m]]
    and its directional derivative uses Hessian blocks.
    """

    # Signed CDF blocks at z=m
    def F1(self, m: float, eta: float, D: int) -> float:
        ...

    def F2(self, m: float, eta: float, D: int) -> float:
        ...

    # First partials
    def dF1_dm(self, m: float, eta: float) -> float:
        ...

    def dF2_dm(self, m: float, eta: float) -> float:
        ...

    def dF1_deta(self, m: float, eta: float) -> float:
        ...

    def dF2_deta(self, m: float, eta: float) -> float:
        ...

    # Second partials
    def d2F1_dm2(self, m: float, eta: float) -> float:
        ...

    def d2F2_dm2(self, m: float, eta: float) -> float:
        ...

    def d2F1_deta2(self, m: float, eta: float) -> float:
        ...

    def d2F2_deta2(self, m: float, eta: float) -> float:
        ...

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        """F_{1,eta m} = ∂_eta ∂_m F1."""
        ...

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        """F_{2,eta m} = ∂_eta ∂_m F2."""
        ...

    # If you have *non-symmetric* mixed blocks (F_{m eta} != F_{eta m}), you may optionally
    # implement methods named `d2F1_dm_deta` and `d2F2_dm_deta`. The engine will use them
    # when present; otherwise it falls back to `d2F*_deta_dm`.


# ==========================================
# BSM deterministic-volatility Normal blocks
# ==========================================

@dataclass(frozen=True)
class BSMGaussianBlocks:
    """Deterministic-volatility BSM specialization (Gaussian blocks).

    With eta = ε = sigma*sqrt(tau) (total volatility), conditionally on F_t:
      under Q : E ~ N(+0.5*eta^2, eta^2)
      under Q+: E ~ N(-0.5*eta^2, eta^2)

    Then
      d1 = (m + 0.5*eta^2)/eta = m/eta + eta/2
      d2 = (m - 0.5*eta^2)/eta = m/eta - eta/2
      F1 = D * Φ(D d1),  F2 = D * Φ(D d2)
    and all needed derivatives in (m,eta) are closed form.
    """

    def _d1(self, m: float, eta: float) -> float:
        return m / eta + 0.5 * eta

    def _d2(self, m: float, eta: float) -> float:
        return m / eta - 0.5 * eta

    def F1(self, m: float, eta: float, D: int) -> float:
        return D * _norm_cdf(D * self._d1(m, eta))

    def F2(self, m: float, eta: float, D: int) -> float:
        return D * _norm_cdf(D * self._d2(m, eta))

    # --- first partials ---
    def dF1_dm(self, m: float, eta: float) -> float:
        return _norm_pdf(self._d1(m, eta)) / eta

    def dF2_dm(self, m: float, eta: float) -> float:
        return _norm_pdf(self._d2(m, eta)) / eta

    def dF1_deta(self, m: float, eta: float) -> float:
        d1 = self._d1(m, eta)
        dd1 = -m / (eta * eta) + 0.5
        return _norm_pdf(d1) * dd1

    def dF2_deta(self, m: float, eta: float) -> float:
        d2 = self._d2(m, eta)
        dd2 = -m / (eta * eta) - 0.5
        return _norm_pdf(d2) * dd2

    # --- second partials ---
    def d2F1_dm2(self, m: float, eta: float) -> float:
        d1 = self._d1(m, eta)
        return -(d1 * _norm_pdf(d1)) / (eta * eta)

    def d2F2_dm2(self, m: float, eta: float) -> float:
        d2 = self._d2(m, eta)
        return -(d2 * _norm_pdf(d2)) / (eta * eta)

    def d2F1_deta2(self, m: float, eta: float) -> float:
        d1 = self._d1(m, eta)
        dd1 = -m / (eta * eta) + 0.5
        d2d1 = 2.0 * m / (eta**3)
        return (-d1 * _norm_pdf(d1)) * (dd1 * dd1) + _norm_pdf(d1) * d2d1

    def d2F2_deta2(self, m: float, eta: float) -> float:
        d2 = self._d2(m, eta)
        dd2 = -m / (eta * eta) - 0.5
        d2d2 = 2.0 * m / (eta**3)
        return (-d2 * _norm_pdf(d2)) * (dd2 * dd2) + _norm_pdf(d2) * d2d2

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        # ∂_eta ∂_m F1
        d1 = self._d1(m, eta)
        dd1 = -m / (eta * eta) + 0.5
        n1 = _norm_pdf(d1)
        return -(n1 / (eta * eta)) + (1.0 / eta) * (-(d1 * n1)) * dd1

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        d2 = self._d2(m, eta)
        dd2 = -m / (eta * eta) - 0.5
        n2 = _norm_pdf(d2)
        return -(n2 / (eta * eta)) + (1.0 / eta) * (-(d2 * n2)) * dd2

    # For Gaussian blocks, Clairaut symmetry holds, so dm_deta = deta_dm.
    def d2F1_dm_deta(self, m: float, eta: float) -> float:
        return self.d2F1_deta_dm(m, eta)

    def d2F2_dm_deta(self, m: float, eta: float) -> float:
        return self.d2F2_deta_dm(m, eta)


# ==========================================
# Primitive model for BSM constants
# ==========================================

@dataclass(frozen=True)
class BSMPrimitives:
    """Constant-parameter BSM primitives for the DSE engine.

    a1  = S * exp(-q*tau)
    a2  = -K * exp(-r*tau)
    m   = ln(S/K) + (r-q)*tau
    eta = ε = sigma*sqrt(tau)    (total volatility)

    D = +1 call, D = -1 put
    """

    S: float
    K: float
    tau: float
    r: float
    q: float
    sigma: float
    D: int = +1

    def primitives(self) -> Tuple[float, float, float, float]:
        a1 = self.S * exp(-self.q * self.tau)
        a2 = -self.K * exp(-self.r * self.tau)
        m = log(self.S / self.K) + (self.r - self.q) * self.tau
        eta = self.sigma * sqrt(self.tau)
        return a1, a2, eta, m

    def d_primitives(self, var: str) -> Tuple[float, float, float, float]:
        a1, a2, eta, m = self.primitives()

        if var == "S":
            return exp(-self.q * self.tau), 0.0, 0.0, 1.0 / self.S
        if var == "K":
            return 0.0, -exp(-self.r * self.tau), 0.0, -1.0 / self.K
        if var == "r":
            return 0.0, -self.tau * a2, 0.0, self.tau
        if var == "q":
            return -self.tau * a1, 0.0, 0.0, -self.tau
        if var == "sigma":
            return 0.0, 0.0, sqrt(self.tau), 0.0
        if var == "tau":
            deta = self.sigma / (2.0 * sqrt(self.tau))
            return -self.q * a1, -self.r * a2, deta, (self.r - self.q)

        raise ValueError(f"Unsupported var '{var}'. Use one of: S,K,r,q,sigma,tau")

    def d2_primitives(self, var1: str, var2: str) -> Tuple[float, float, float, float]:
        """Second derivatives for common BSM parameters.

        Enough to demo mixed partials. Extend as needed.
        """

        a1, a2, eta, m = self.primitives()
        d2a1 = d2a2 = d2eta = d2m = 0.0

        # a1 = S e^{-q tau}
        if var1 == "S" and var2 == "S":
            d2m = -1.0 / (self.S * self.S)
        elif {var1, var2} == {"S", "q"}:
            d2a1 = -self.tau * exp(-self.q * self.tau)
        elif {var1, var2} == {"S", "tau"}:
            d2a1 = -self.q * exp(-self.q * self.tau)
        elif var1 == "q" and var2 == "q":
            d2a1 = (self.tau * self.tau) * a1
        elif {var1, var2} == {"q", "tau"}:
            d2a1 = -a1 + self.q * self.tau * a1
            d2m = -1.0
        elif var1 == "tau" and var2 == "tau":
            d2a1 = (self.q * self.q) * a1

        # a2 = -K e^{-r tau}
        if var1 == "K" and var2 == "K":
            d2m += 1.0 / (self.K * self.K)
        elif {var1, var2} == {"K", "r"}:
            d2a2 = self.tau * exp(-self.r * self.tau)
        elif {var1, var2} == {"K", "tau"}:
            d2a2 = self.r * exp(-self.r * self.tau)
        elif var1 == "r" and var2 == "r":
            d2a2 = (self.tau * self.tau) * a2
        elif {var1, var2} == {"r", "tau"}:
            d2a2 = -a2 + self.r * self.tau * a2
            d2m += 1.0
        elif var1 == "tau" and var2 == "tau":
            d2a2 = (self.r * self.r) * a2

        # eta = sigma*sqrt(tau)
        if {var1, var2} == {"sigma", "tau"}:
            d2eta = 1.0 / (2.0 * sqrt(self.tau))
        elif var1 == "tau" and var2 == "tau":
            d2eta = -self.sigma / (4.0 * (self.tau ** (3.0 / 2.0)))

        # m = ln(S/K) + (r-q)tau
        if var1 == "r" and var2 == "tau":
            d2m += 1.0
        elif var1 == "q" and var2 == "tau":
            d2m += -1.0

        return d2a1, d2a2, d2eta, d2m


# ==========================================
# Matrix blocks V, V_eta, V_m
# ==========================================

def _maybe_mixed(blocks: DistributionBlocks, which: str, m: float, eta: float) -> float:
    """Return mixed second derivative.

    which in {"F1_meta", "F2_meta"} for F_{m eta} (i.e. ∂_m ∂_eta).
    Falls back to ∂_eta ∂_m if dm_deta is not implemented.
    """
    if which == "F1_meta":
        fn = getattr(blocks, "d2F1_dm_deta", None)
        if callable(fn):
            try:
                return fn(m, eta)
            except TypeError:
                pass
        return blocks.d2F1_deta_dm(m, eta)
    if which == "F2_meta":
        fn = getattr(blocks, "d2F2_dm_deta", None)
        if callable(fn):
            try:
                return fn(m, eta)
            except TypeError:
                pass
        return blocks.d2F2_deta_dm(m, eta)
    raise ValueError("Unknown mixed derivative selector")


def V_matrix(m: float, eta: float, blocks: DistributionBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Paper DSE Eq. (30): V(m;eta) Jacobian in (eta,m)."""
    return (
        (blocks.dF1_deta(m, eta), blocks.dF1_dm(m, eta)),
        (blocks.dF2_deta(m, eta), blocks.dF2_dm(m, eta)),
    )


def V_eta_matrix(m: float, eta: float, blocks: DistributionBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Paper DSE Section 8 / Appendix B: V_eta(m;eta)."""
    return (
        (blocks.d2F1_deta2(m, eta), _maybe_mixed(blocks, "F1_meta", m, eta)),
        (blocks.d2F2_deta2(m, eta), _maybe_mixed(blocks, "F2_meta", m, eta)),
    )


def V_m_matrix(m: float, eta: float, blocks: DistributionBlocks) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Paper DSE Section 8 / Appendix B: V_m(m;eta)."""
    return (
        (blocks.d2F1_deta_dm(m, eta), blocks.d2F1_dm2(m, eta)),
        (blocks.d2F2_deta_dm(m, eta), blocks.d2F2_dm2(m, eta)),
    )


def _matvec2(M: Tuple[Tuple[float, float], Tuple[float, float]], v: Tuple[float, float]) -> Tuple[float, float]:
    return (
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    )


def _dot2(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def P2_matrix_form(
    *,
    a12: Tuple[float, float],
    a12_x1: Tuple[float, float],
    a12_x2: Tuple[float, float],
    a34_x1: Tuple[float, float],
    a34_x2: Tuple[float, float],
    m: float,
    eta: float,
    eta_x2: float,
    m_x2: float,
    blocks: DistributionBlocks,
) -> Dict[str, float]:
    """Compute P2 via Paper DSE Eq. (31) / Appendix B decomposition.

    Returns a dict with keys: "cross", "non", "total".
    """

    V = V_matrix(m, eta, blocks)

    # Cross part: a12_{x1}^T V a34_{x2} + a12_{x2}^T V a34_{x1}
    cross_1 = _dot2(a12_x1, _matvec2(V, a34_x2))
    cross_2 = _dot2(a12_x2, _matvec2(V, a34_x1))
    P2_cross = cross_1 + cross_2

    # Non-cross: a12^T V_{x2} a34_{x1} with V_{x2} = V_eta*eta_{x2} + V_m*m_{x2}
    V_eta = V_eta_matrix(m, eta, blocks)
    V_m = V_m_matrix(m, eta, blocks)
    V_x2 = (
        (V_eta[0][0] * eta_x2 + V_m[0][0] * m_x2, V_eta[0][1] * eta_x2 + V_m[0][1] * m_x2),
        (V_eta[1][0] * eta_x2 + V_m[1][0] * m_x2, V_eta[1][1] * eta_x2 + V_m[1][1] * m_x2),
    )
    P2_non = _dot2(a12, _matvec2(V_x2, a34_x1))

    return {"cross": P2_cross, "non": P2_non, "total": P2_cross + P2_non}


# ==========================================
# DSE weights
# ==========================================

def dse_weights(a1: float, a2: float, eta: float, m: float, D: int, blocks: DistributionBlocks) -> Dict[str, float]:
    """Weights w1..w4 (Paper DSE Eqs. (20), (25), (26))."""
    w1 = blocks.F1(m, eta, D)
    w2 = blocks.F2(m, eta, D)

    F1_eta = blocks.dF1_deta(m, eta)
    F2_eta = blocks.dF2_deta(m, eta)
    F1_m = blocks.dF1_dm(m, eta)
    F2_m = blocks.dF2_dm(m, eta)

    w3 = a1 * F1_eta + a2 * F2_eta
    w4 = a1 * F1_m + a2 * F2_m

    return {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "F1_eta": F1_eta,
        "F2_eta": F2_eta,
        "F1_m": F1_m,
        "F2_m": F2_m,
    }


# ==========================================
# Engine: price, first derivative, mixed partial
# ==========================================

def dse_price(prims: BSMPrimitives, blocks: DistributionBlocks) -> float:
    a1, a2, eta, m = prims.primitives()
    w = dse_weights(a1, a2, eta, m, prims.D, blocks)
    return w["w1"] * a1 + w["w2"] * a2


def dse_first_derivative(prims: BSMPrimitives, var: str, blocks: DistributionBlocks) -> float:
    """Paper DSE Eq. (27): V_x = w1*a1_x + w2*a2_x + w3*eta_x + w4*m_x."""
    a1, a2, eta, m = prims.primitives()
    da1, da2, deta, dm = prims.d_primitives(var)
    w = dse_weights(a1, a2, eta, m, prims.D, blocks)
    return w["w1"] * da1 + w["w2"] * da2 + w["w3"] * deta + w["w4"] * dm


def dse_mixed_partial_components(
    prims: BSMPrimitives,
    var1: str,
    var2: str,
    blocks: DistributionBlocks,
) -> Dict[str, float]:
    """Return the mixed-partial decomposition (Paper DSE Eq. (28)–(29) + Eq. (31)).

    Output keys:
      - dot:   w~^T a_{x1x2}
      - P2_cross: cross part of P2
      - P2_non:   non-cross part of P2
      - P2:    total correction channel
      - total: dot + P2
    """

    a1, a2, eta, m = prims.primitives()
    D = prims.D

    # First derivatives of primitives
    da1_1, da2_1, deta_1, dm_1 = prims.d_primitives(var1)
    da1_2, da2_2, deta_2, dm_2 = prims.d_primitives(var2)

    # Second derivatives of primitives
    d2a1, d2a2, d2eta, d2m = prims.d2_primitives(var1, var2)

    # Weights
    w = dse_weights(a1, a2, eta, m, D, blocks)
    dot_term = w["w1"] * d2a1 + w["w2"] * d2a2 + w["w3"] * d2eta + w["w4"] * d2m

    # Matrix-form P2 (Appendix B): use only first derivatives of primitives + Jacobian/Hessian blocks.
    a12 = (a1, a2)
    a12_x1 = (da1_1, da2_1)
    a12_x2 = (da1_2, da2_2)
    a34_x1 = (deta_1, dm_1)
    a34_x2 = (deta_2, dm_2)

    P2_parts = P2_matrix_form(
        a12=a12,
        a12_x1=a12_x1,
        a12_x2=a12_x2,
        a34_x1=a34_x1,
        a34_x2=a34_x2,
        m=m,
        eta=eta,
        eta_x2=deta_2,
        m_x2=dm_2,
        blocks=blocks,
    )

    total = dot_term + P2_parts["total"]
    return {
        "dot": dot_term,
        "P2_cross": P2_parts["cross"],
        "P2_non": P2_parts["non"],
        "P2": P2_parts["total"],
        "total": total,
    }


def dse_mixed_partial(prims: BSMPrimitives, var1: str, var2: str, blocks: DistributionBlocks) -> float:
    """Convenience wrapper: returns dot + P2."""
    return dse_mixed_partial_components(prims, var1, var2, blocks)["total"]


# ==========================================
# Example usage / sanity checks (BSM case)
# ==========================================

# =============================================================================
# Model-specific DistributionBlocks implementations
#
#   1) Local vol PDE
#   2) Heston MC
#   3) General MC with Likelihood-Ratio / (Malliavin-ready hooks)
#
# Notes:
# - These blocks expose the engine-facing interface defined by DistributionBlocks above.
# - Only F1/F2 depend on D (call/put sign convention). All (m,eta)-derivatives are D-invariant
#   because the D=-1 version differs by a constant shift (so derivatives match the call case).
# - This section requires NumPy. The core engine and BSMGaussianBlocks remain NumPy-free.
# =============================================================================

from typing import Callable, Optional, Any
import math

try:
    import numpy as np  # type: ignore
except Exception as _e:  # pragma: no cover
    np = None  # type: ignore


# ------------------------ numpy availability guard ------------------------

def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise ImportError(
            "NumPy is required for LocalVolPDEBlocks / HestonMCBlocks / GeneralMCLRBlocks. "
            "Install numpy or use BSMGaussianBlocks."
        )


# ------------------------ small math helpers (no scipy) ------------------------

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_cdf_np(x):
    _require_numpy()
    # Φ(x) = 0.5*(1+erf(x/sqrt(2)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT_2))

def _norm_pdf_np(x):
    _require_numpy()
    # φ(x) = exp(-x^2/2)/sqrt(2π)
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _as_callable_rate(x: float | Callable[[float], float]) -> Callable[[float], float]:
    if callable(x):
        return x
    return lambda t: float(x)


def _trapz_integral(f: Callable[[float], float], a: float, b: float, n: int = 64) -> float:
    _require_numpy()
    if b <= a:
        return 0.0
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(float(u)) for u in xs], dtype=float)
    # NumPy 2.x removed np.trapz; use np.trapezoid when available (fallback keeps NumPy 1.x compatibility).
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz")
    return float(trapz_fn(ys, xs))

def _signed_from_call_prob(prob_call: float, D: int) -> float:
    # Call-sign event prob_call = P(S_T > K).
    # Signed CDF used in the paper:
    #   D=+1 -> +P(S_T>K)
    #   D=-1 -> -P(S_T<K) = P(S_T>K) - 1
    if D == 1:
        return prob_call
    if D == -1:
        return prob_call - 1.0
    raise ValueError("D must be +1 or -1")


# ------------------------ tridiagonal solver (Thomas) ------------------------

def _solve_tridiagonal(a, b, c, d):
    """
    Solve tridiagonal system:
        a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i], i=0..n-1
    with a[0]=0 and c[n-1]=0 convention.
    """
    _require_numpy()
    n = len(b)
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)

    # forward sweep
    denom = b[0]
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Tridiagonal solver: near-zero pivot at 0.")
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-14:
            raise ZeroDivisionError(f"Tridiagonal solver: near-zero pivot at {i}.")
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # back substitution
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# =============================================================================
# 1) Local volatility PDE blocks
# =============================================================================

@dataclass
class LocalVolPDEBlocks:
    """
    Backward PDE for digital probabilities under Q and Q+.

    sigma_local(t, S) is the local vol (annualized).
    r(t), q(t) can be floats (constants) or callables.

    Engine interface:
      F1/F2 are the signed CDF blocks (depend on D).
      All (m,eta)-derivatives are D-invariant and match the call case.
    """
    S0: float
    sigma_local: Callable[[float, float], float]
    r: float | Callable[[float], float] = 0.0
    q: float | Callable[[float], float] = 0.0
    t0: float = 0.0

    # numerical grid
    x_width: float = 6.0         # log-space half-width around ln(S0)
    nx: int = 401
    nt: int = 200
    theta: float = 0.5           # Crank–Nicolson

    # parameterization of eta
    eta_mode: str = "tau"        # "tau" or "vol_scale"
    tau_fixed: Optional[float] = None  # used if eta_mode != "tau"

    # finite difference steps
    fd_m: float = 1e-4
    fd_eta: float = 1e-4

    # caching
    cache: bool = True

    def __post_init__(self):
        _require_numpy()
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")
        self._r = _as_callable_rate(self.r)
        self._q = _as_callable_rate(self.q)
        self._cache_prob: Dict[tuple, float] = {}

    def _carry(self, tau: float) -> float:
        return _trapz_integral(lambda u: self._r(u) - self._q(u), self.t0, self.t0 + tau, n=64)

    def _tau_from_eta(self, eta: float) -> float:
        if self.eta_mode == "tau":
            return float(eta)
        if self.tau_fixed is None:
            raise ValueError("tau_fixed must be set when eta_mode != 'tau'.")
        return float(self.tau_fixed)

    def _sigma_eff(self, t: float, S: float, eta: float) -> float:
        base = float(self.sigma_local(t, S))
        if self.eta_mode == "vol_scale":
            return float(eta) * base
        return base

    def _strike_from_m(self, m: float, tau: float) -> float:
        # m = ln(S0/K) + carry  =>  K = S0 * exp(carry - m)
        return self.S0 * math.exp(self._carry(tau) - m)

    def _prob_call_pde(self, measure: str, m: float, eta: float) -> float:
        """
        Returns P^{measure}(S_T > K(m)) where measure is 'Q' or 'Q+'.
        """
        tau = self._tau_from_eta(eta)
        if tau <= 0:
            K = self._strike_from_m(m, max(tau, 0.0))
            return 1.0 if (self.S0 > K) else 0.0

        key = (measure, float(m), float(eta))
        if self.cache and key in self._cache_prob:
            return self._cache_prob[key]

        x0 = math.log(self.S0)
        x_min = x0 - self.x_width
        x_max = x0 + self.x_width
        x = np.linspace(x_min, x_max, self.nx)
        dx = float(x[1] - x[0])

        K = self._strike_from_m(m, tau)
        lnK = math.log(K)

        # terminal condition for call-digital: 1{x > lnK}
        u = (x > lnK).astype(float)
        u[0] = 0.0
        u[-1] = 1.0

        dt = float(tau / self.nt)
        th = float(self.theta)

        # Q+ drift adjustment in log-space (see paper Section 5–6):
        # mu_Q  = (r-q) - 0.5*var,  mu_Q+ = (r-q) + 0.5*var
        is_qplus = (measure.upper() in ("Q+", "QP", "QPLUS", "Q_PLUS"))

        for n in range(self.nt):
            # stepping backward from T to t0
            t = self.t0 + (tau - (n + 1) * dt)

            S_grid = np.exp(x)
            sig = np.array([self._sigma_eff(t, float(S_grid[i]), eta) for i in range(self.nx)], dtype=float)
            var = sig * sig

            r_t = float(self._r(t))
            q_t = float(self._q(t))

            mu = (r_t - q_t) + (0.5 * var if is_qplus else -0.5 * var)

            # build tri-diagonal for interior points
            a = np.zeros(self.nx - 2, dtype=float)
            b = np.zeros(self.nx - 2, dtype=float)
            c = np.zeros(self.nx - 2, dtype=float)

            aB = np.zeros(self.nx - 2, dtype=float)
            bB = np.zeros(self.nx - 2, dtype=float)
            cB = np.zeros(self.nx - 2, dtype=float)

            # coefficients for L u = mu u_x + 0.5 var u_xx (central diffs)
            for i in range(1, self.nx - 1):
                j = i - 1
                vi = var[i]
                mui = mu[i]

                lower = 0.5 * vi / (dx * dx) - mui / (2.0 * dx)
                diag  = -vi / (dx * dx)
                upper = 0.5 * vi / (dx * dx) + mui / (2.0 * dx)

                # Crank–Nicolson step in time-to-maturity s (forward):
                #   u_s = L u
                # (I - θ dt L) u^{n+1} = (I + (1-θ) dt L) u^{n}
                a[j] = -th * dt * lower
                b[j] = 1.0 - th * dt * diag
                c[j] = -th * dt * upper

                aB[j] = (1.0 - th) * dt * lower
                bB[j] = 1.0 + (1.0 - th) * dt * diag
                cB[j] = (1.0 - th) * dt * upper

            rhs = bB * u[1:-1] + cB * u[2:] + aB * u[:-2]

            # boundary contributions
            rhs[0]  -= a[0]  * u[0]
            rhs[-1] -= c[-1] * u[-1]

            u_new_inner = _solve_tridiagonal(
                a=np.r_[0.0, a[1:]],
                b=b,
                c=np.r_[c[:-1], 0.0],
                d=rhs
            )
            u[1:-1] = u_new_inner
            u[0] = 0.0
            u[-1] = 1.0

        prob = float(np.interp(x0, x, u))
        prob = min(max(prob, 0.0), 1.0)

        if self.cache:
            self._cache_prob[key] = prob
        return prob

    # ---------- engine-facing blocks ----------

    def F1(self, m: float, eta: float, D: int) -> float:
        p = self._prob_call_pde("Q+", m, eta)
        return _signed_from_call_prob(p, D)

    def F2(self, m: float, eta: float, D: int) -> float:
        p = self._prob_call_pde("Q", m, eta)
        return _signed_from_call_prob(p, D)

    # --- derivatives via FD (robust, slower) ---
    def _fd(self, measure: str, m: float, eta: float, dm: float = None, deta: float = None) -> Dict[str, float]:
        dm = self.fd_m if dm is None else float(dm)
        deta = self.fd_eta if deta is None else float(deta)

        def F(mm: float, ee: float) -> float:
            return self._prob_call_pde(measure, mm, ee)

        f00 = F(m, eta)
        fp_m = F(m + dm, eta)
        fm_m = F(m - dm, eta)

        fp_e = F(m, eta + deta)
        fm_e = F(m, eta - deta)

        fpp = F(m + dm, eta + deta)
        fpm = F(m + dm, eta - deta)
        fmp = F(m - dm, eta + deta)
        fmm = F(m - dm, eta - deta)

        dF_dm = (fp_m - fm_m) / (2.0 * dm)
        d2F_dm2 = (fp_m - 2.0 * f00 + fm_m) / (dm * dm)

        dF_deta = (fp_e - fm_e) / (2.0 * deta)
        d2F_deta2 = (fp_e - 2.0 * f00 + fm_e) / (deta * deta)

        d2F_deta_dm = (fpp - fpm - fmp + fmm) / (4.0 * dm * deta)

        return dict(
            F=f00,
            F_m=dF_dm,
            F_mm=d2F_dm2,
            F_eta=dF_deta,
            F_etaeta=d2F_deta2,
            F_eta_m=d2F_deta_dm,
        )

    # First partials
    def dF1_dm(self, m: float, eta: float) -> float:
        return self._fd("Q+", m, eta)["F_m"]

    def dF2_dm(self, m: float, eta: float) -> float:
        return self._fd("Q", m, eta)["F_m"]

    def dF1_deta(self, m: float, eta: float) -> float:
        return self._fd("Q+", m, eta)["F_eta"]

    def dF2_deta(self, m: float, eta: float) -> float:
        return self._fd("Q", m, eta)["F_eta"]

    # Second partials
    def d2F1_dm2(self, m: float, eta: float) -> float:
        return self._fd("Q+", m, eta)["F_mm"]

    def d2F2_dm2(self, m: float, eta: float) -> float:
        return self._fd("Q", m, eta)["F_mm"]

    def d2F1_deta2(self, m: float, eta: float) -> float:
        return self._fd("Q+", m, eta)["F_etaeta"]

    def d2F2_deta2(self, m: float, eta: float) -> float:
        return self._fd("Q", m, eta)["F_etaeta"]

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        return self._fd("Q+", m, eta)["F_eta_m"]

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        return self._fd("Q", m, eta)["F_eta_m"]


# =============================================================================
# 2) Heston Monte Carlo blocks
# =============================================================================

@dataclass
class HestonMCBlocks:
    """
    Heston simulation for digitals under Q and Q+.

    Under Q:
      dS/S = (r-q) dt + sqrt(v) dW1
      dv   = kappa(theta - v) dt + xi sqrt(v) dW2
      corr(W1,W2)=rho

    Under Q+ induced by the Doléans–Dade density (paper Section 5–6):
      dW1^+ = dW1 - sqrt(v) dt
      and in a correlated representation:
      dW2^+ = dW2 - rho sqrt(v) dt

    Derivatives wrt m use a normal-kernel smoothing in log-moneyness:
      I_h = Φ( log(S_T/K)/h )
      d/dm I_h   = φ(z)/h
      d2/dm2 I_h = -z φ(z) / h^2
    """
    S0: float
    v0: float
    r: float = 0.0
    q: float = 0.0
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.4
    rho: float = -0.7

    # eta parameterization
    eta_mode: str = "tau"              # "tau" or "vol_scale"
    tau_fixed: Optional[float] = None  # required if eta_mode != "tau"

    # MC controls
    n_paths: int = 20000
    n_steps: int = 200
    seed: int = 12345

    # smoothing bandwidth in log-space for m-derivatives
    h_m: float = 0.02

    # finite difference for eta derivatives
    fd_eta: float = 1e-4

    # if True, use smoothed CDF for Phi (less noisy, slightly biased)
    smooth_phi: bool = False

    def __post_init__(self):
        _require_numpy()
        if self.S0 <= 0:
            raise ValueError("S0 must be positive.")
        if self.v0 < 0:
            raise ValueError("v0 must be nonnegative.")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must be in (-1,1).")
        rng = np.random.default_rng(self.seed)
        self._Z1 = rng.standard_normal((self.n_paths, self.n_steps))
        self._Z2 = rng.standard_normal((self.n_paths, self.n_steps))

    def _tau_from_eta(self, eta: float) -> float:
        if self.eta_mode == "tau":
            return float(eta)
        if self.tau_fixed is None:
            raise ValueError("tau_fixed must be set when eta_mode != 'tau'.")
        return float(self.tau_fixed)

    def _params_from_eta(self, eta: float) -> tuple[float, float]:
        # Illustrative: if eta_mode="vol_scale", scale v0 and theta by eta^2.
        if self.eta_mode == "vol_scale":
            scale = float(eta) * float(eta)
            return self.v0 * scale, self.theta * scale
        return self.v0, self.theta

    def _strike_from_m(self, m: float, tau: float) -> float:
        carry = (self.r - self.q) * tau
        return self.S0 * math.exp(carry - m)

    def _simulate_terminal(self, measure: str, tau: float, eta: float):
        if tau <= 0:
            return np.full(self.n_paths, self.S0, dtype=float)

        v0_eff, theta_eff = self._params_from_eta(eta)
        dt = tau / self.n_steps
        sqdt = math.sqrt(dt)
        rho = self.rho
        sqrt_1mr2 = math.sqrt(1.0 - rho * rho)

        logS = np.full(self.n_paths, math.log(self.S0), dtype=float)
        v = np.full(self.n_paths, v0_eff, dtype=float)

        is_qplus = (measure.upper() in ("Q+", "QP", "QPLUS", "Q_PLUS"))

        for i in range(self.n_steps):
            z1 = self._Z1[:, i]
            z2 = self._Z2[:, i]
            vpos = np.maximum(v, 0.0)
            sqrtv = np.sqrt(vpos)

            dW2 = (rho * z1 + sqrt_1mr2 * z2) * sqdt

            if is_qplus:
                # logS drift: (r-q + 0.5 v) dt
                logS += (self.r - self.q + 0.5 * vpos) * dt + sqrtv * (z1 * sqdt)
                # v drift: kappa(theta - v) + rho*xi*v
                v += (self.kappa * (theta_eff - v) + rho * self.xi * vpos) * dt + self.xi * sqrtv * dW2
            else:
                # logS drift: (r-q - 0.5 v) dt
                logS += (self.r - self.q - 0.5 * vpos) * dt + sqrtv * (z1 * sqdt)
                v += self.kappa * (theta_eff - v) * dt + self.xi * sqrtv * dW2

            v = np.maximum(v, 0.0)  # full truncation

        return np.exp(logS)

    def _prob_and_m_derivs(self, measure: str, m: float, eta: float) -> tuple[float, float, float]:
        tau = self._tau_from_eta(eta)
        ST = self._simulate_terminal(measure, tau, eta)
        K = self._strike_from_m(m, tau)

        if self.smooth_phi:
            z = (np.log(ST / K)) / self.h_m
            prob = float(np.mean(_norm_cdf_np(z)))
        else:
            prob = float(np.mean(ST > K))

        z = (np.log(ST / K)) / self.h_m
        pdfz = _norm_pdf_np(z)
        dF_dm = float(np.mean(pdfz / self.h_m))
        d2F_dm2 = float(np.mean((-z * pdfz) / (self.h_m * self.h_m)))
        return prob, dF_dm, d2F_dm2

    def _fd_eta(self, measure: str, m: float, eta: float, deta: float) -> Dict[str, float]:
        e_plus = eta + deta
        e_minus = eta - deta

        p0, dm0, dmm0 = self._prob_and_m_derivs(measure, m, eta)
        pP, dmP, dmmP = self._prob_and_m_derivs(measure, m, e_plus)
        pM, dmM, dmmM = self._prob_and_m_derivs(measure, m, e_minus)

        dP_deta = (pP - pM) / (2.0 * deta)
        d2P_deta2 = (pP - 2.0 * p0 + pM) / (deta * deta)

        d_dm_deta = (dmP - dmM) / (2.0 * deta)         # ∂_eta ∂_m Phi

        return dict(
            P=p0, P_eta=dP_deta, P_etaeta=d2P_deta2,
            P_m=dm0, P_mm=dmm0,
            P_eta_m=d_dm_deta,
        )

    # ---------- engine-facing blocks ----------

    def F1(self, m: float, eta: float, D: int) -> float:
        p, _, _ = self._prob_and_m_derivs("Q+", m, eta)
        return _signed_from_call_prob(p, D)

    def F2(self, m: float, eta: float, D: int) -> float:
        p, _, _ = self._prob_and_m_derivs("Q", m, eta)
        return _signed_from_call_prob(p, D)

    # First partials
    def dF1_dm(self, m: float, eta: float) -> float:
        _, dm, _ = self._prob_and_m_derivs("Q+", m, eta)
        return dm

    def dF2_dm(self, m: float, eta: float) -> float:
        _, dm, _ = self._prob_and_m_derivs("Q", m, eta)
        return dm

    def dF1_deta(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q+", m, eta, self.fd_eta)
        return out["P_eta"]

    def dF2_deta(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q", m, eta, self.fd_eta)
        return out["P_eta"]

    # Second partials
    def d2F1_dm2(self, m: float, eta: float) -> float:
        _, _, dmm = self._prob_and_m_derivs("Q+", m, eta)
        return dmm

    def d2F2_dm2(self, m: float, eta: float) -> float:
        _, _, dmm = self._prob_and_m_derivs("Q", m, eta)
        return dmm

    def d2F1_deta2(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q+", m, eta, self.fd_eta)
        return out["P_etaeta"]

    def d2F2_deta2(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q", m, eta, self.fd_eta)
        return out["P_etaeta"]

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q+", m, eta, self.fd_eta)
        return out["P_eta_m"]

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        out = self._fd_eta("Q", m, eta, self.fd_eta)
        return out["P_eta_m"]


# =============================================================================
# 3) General MC blocks with LR / Malliavin-ready hooks
# =============================================================================

class GeneralMCSimulator:
    """
    User-supplied simulator interface (duck-typed):

    simulate(measure, m, eta, n_paths, n_steps, seed) -> dict with at least:
        "S_T": np.ndarray shape (n_paths,)

    Optional (Likelihood Ratio / score function):
        "score_eta": np.ndarray         = ∂_eta log p(path;eta)     (under given measure)
        "score_eta_eta": np.ndarray     = ∂_eta^2 log p(path;eta)

    If score arrays are provided:
        d/deta E[g]      = E[g * score_eta]
        d2/deta2 E[g]    = E[g * (score_eta_eta + score_eta^2)]
        d/deta d/dm E[g] = E[(∂_m g) * score_eta]   (since score doesn't depend on m)
    """
    def simulate(self, measure: str, m: float, eta: float, n_paths: int, n_steps: int, seed: int) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class GeneralMCLRBlocks:
    """
    Generic Monte Carlo DistributionBlocks with optional LR scores.
    Uses smoothing in log-space for m-derivatives.
    """
    S0: float
    carry: float                   # carry = ∫(r-q)du over [t,T] (fixed here)
    simulator: GeneralMCSimulator

    n_paths: int = 50000
    n_steps: int = 200
    seed: int = 777

    # smoothing bandwidth for m derivatives
    h_m: float = 0.02

    # FD fallback if no LR scores provided
    fd_eta: float = 1e-4

    # optional: smooth Phi instead of hard indicator
    smooth_phi: bool = False

    def __post_init__(self):
        _require_numpy()

    def _strike_from_m(self, m: float) -> float:
        return self.S0 * math.exp(self.carry - m)

    def _run(self, measure: str, m: float, eta: float) -> Dict[str, Any]:
        out = self.simulator.simulate(measure, m, eta, self.n_paths, self.n_steps, self.seed)
        if "S_T" not in out:
            raise ValueError("Simulator must return dict containing key 'S_T'.")
        return out

    def _prob_and_m_derivs(self, measure: str, m: float, eta: float):
        out = self._run(measure, m, eta)
        ST = out["S_T"]
        K = self._strike_from_m(m)

        if self.smooth_phi:
            z = (np.log(ST / K)) / self.h_m
            prob = float(np.mean(_norm_cdf_np(z)))
        else:
            prob = float(np.mean(ST > K))

        z = (np.log(ST / K)) / self.h_m
        pdfz = _norm_pdf_np(z)
        dF_dm = float(np.mean(pdfz / self.h_m))
        d2F_dm2 = float(np.mean((-z * pdfz) / (self.h_m * self.h_m)))
        return prob, dF_dm, d2F_dm2, out

    def _deta_lr_or_fd(self, measure: str, m: float, eta: float):
        """
        Returns:
          (prob, prob_eta, prob_etaeta, phi_eta)
        where prob = P(S_T > K(m)).
        """
        prob, dF_dm, _, out = self._prob_and_m_derivs(measure, m, eta)

        if "score_eta" in out and out["score_eta"] is not None:
            score = out["score_eta"]
            ST = out["S_T"]
            K = self._strike_from_m(m)
            g = (ST > K).astype(float)

            prob_eta = float(np.mean(g * score))

            if "score_eta_eta" in out and out["score_eta_eta"] is not None:
                score2 = out["score_eta_eta"]
                prob_etaeta = float(np.mean(g * (score2 + score * score)))
            else:
                prob_etaeta = float("nan")

            z = (np.log(ST / K)) / self.h_m
            dm_smooth = _norm_pdf_np(z) / self.h_m
            phi_eta = float(np.mean(dm_smooth * score))
            return prob, prob_eta, prob_etaeta, phi_eta

        # FD fallback
        deta = self.fd_eta
        pP, _, _, _ = self._prob_and_m_derivs(measure, m, eta + deta)
        pM, _, _, _ = self._prob_and_m_derivs(measure, m, eta - deta)
        prob_eta = (pP - pM) / (2.0 * deta)
        prob_etaeta = (pP - 2.0 * prob + pM) / (deta * deta)

        _, dmP, _, _ = self._prob_and_m_derivs(measure, m, eta + deta)
        _, dmM, _, _ = self._prob_and_m_derivs(measure, m, eta - deta)
        phi_eta = (dmP - dmM) / (2.0 * deta)

        return prob, prob_eta, prob_etaeta, phi_eta

    # ---------- engine-facing blocks ----------

    def F1(self, m: float, eta: float, D: int) -> float:
        p, _, _, _ = self._prob_and_m_derivs("Q+", m, eta)
        return _signed_from_call_prob(p, D)

    def F2(self, m: float, eta: float, D: int) -> float:
        p, _, _, _ = self._prob_and_m_derivs("Q", m, eta)
        return _signed_from_call_prob(p, D)

    # First partials
    def dF1_dm(self, m: float, eta: float) -> float:
        _, dm, _, _ = self._prob_and_m_derivs("Q+", m, eta)
        return dm

    def dF2_dm(self, m: float, eta: float) -> float:
        _, dm, _, _ = self._prob_and_m_derivs("Q", m, eta)
        return dm

    def dF1_deta(self, m: float, eta: float) -> float:
        _, prob_eta, _, _ = self._deta_lr_or_fd("Q+", m, eta)
        return prob_eta

    def dF2_deta(self, m: float, eta: float) -> float:
        _, prob_eta, _, _ = self._deta_lr_or_fd("Q", m, eta)
        return prob_eta

    # Second partials
    def d2F1_dm2(self, m: float, eta: float) -> float:
        _, _, dmm, _ = self._prob_and_m_derivs("Q+", m, eta)
        return dmm

    def d2F2_dm2(self, m: float, eta: float) -> float:
        _, _, dmm, _ = self._prob_and_m_derivs("Q", m, eta)
        return dmm

    def d2F1_deta2(self, m: float, eta: float) -> float:
        _, _, prob_etaeta, _ = self._deta_lr_or_fd("Q+", m, eta)
        return prob_etaeta

    def d2F2_deta2(self, m: float, eta: float) -> float:
        _, _, prob_etaeta, _ = self._deta_lr_or_fd("Q", m, eta)
        return prob_etaeta

    def d2F1_deta_dm(self, m: float, eta: float) -> float:
        _, _, _, phi_eta = self._deta_lr_or_fd("Q+", m, eta)
        return phi_eta

    def d2F2_deta_dm(self, m: float, eta: float) -> float:
        _, _, _, phi_eta = self._deta_lr_or_fd("Q", m, eta)
        return phi_eta
