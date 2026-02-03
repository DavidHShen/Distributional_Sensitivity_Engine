# Distributional Sensitivity Engine (DSE)

This repository provides a reference implementation of the Distributional Sensitivity Engine (DSE):

The DSE expresses prices and (mixed) sensitivities using distributional blocks for a log Doléans–Dade
exponential and a small set of primitive derivatives. It is designed to be model-agnostic: you plug in
conditional CDF/PDF blocks (under two measures) and the engine handles the chain rule, including the
critical weight-motion correction channel.

---

## 1. Core representation

Let

- $a_1, a_2$ be the two payoff-affine primitives (e.g., carry-adjusted $S$ and $-K$)
- $a_3, a_4\equiv (\eta, m)$ be the two distribution parameters
- $E_{t,T}$ be the log Doléans–Dade variable driving the representation
- $D\in\\{+1,-1\\}$ be the call/put sign convention used in the paper

Define the signed conditional CDF blocks at the evaluation point $z=m$:

$$
F_1(m;\eta) := \Phi^{\mathbb Q^+}_{E\mid\mathcal F_t}(m;\eta),\qquad
F_2(m;\eta) := \Phi^{\mathbb Q}_{E\mid\mathcal F_t}(m;\eta),
$$

(with the signed $D$-convention applied inside `F1/F2`).

Then the value can be written as

$$
V_t = w_1\ a_1 + w_2\ a_2,
\qquad
(w_1,w_2) = (F_1,F_2).
$$

The first derivative w.r.t. any input $x$ takes the compact DSE form

$$
\partial_x V_t
= w_1\ \partial_x a_1 + w_2\ \partial_x a_2 + w_3\ \partial_x \eta + w_4\ \partial_x m,
$$

where the additional weights are

$$
\begin{aligned}
 w_3 &= a_1\ \partial_\eta F_1 + a_2\ \partial_\eta F_2,\\
 w_4 &= a_1\ \partial_m F_1 + a_2\ \partial_m F_2.
\end{aligned}
$$

---

## 2. Mixed partials and the matrix-form correction channel

Mixed partials (e.g., $\partial_{x_1x_2}^2 V_t$) contain a dot term plus a correction channel.
The dot term is the obvious “fixed-weight” chain rule contribution:

$$
\tilde w^\top\ \partial^2_{x_1x_2} a
\quad\text{with}\quad
\tilde w=(w_1,w_2,w_3,w_4)^\top,\; a=(a_1,a_2,\eta,m)^\top.
$$

The correction channel $P_2$ accounts for weight motion induced by the state-dependence of
$(a_1,a_2,\eta,m)$. In Paper 2 this channel is written in a Jacobian/Hessian matrix form that is
implementation-ready.

Define the Jacobian matrix (paper Eq. (30))

$$
V(m;\eta) :=
\begin{pmatrix}
\partial_\eta F_1 & \partial_m F_1\\
\partial_\eta F_2 & \partial_m F_2
\end{pmatrix}.
$$

Let $a_{12}=(a_1,a_2)^\top$, $a_{34}=(\eta,m)^\top$. Then the correction channel can be computed as

$$
P_2(x_1,x_2)
= a_{12,x_1}^\top V\ a_{34,x_2} + a_{12,x_2}^\top V\ a_{34,x_1} + a_{12}^\top \Big( V_{\eta}\ \eta_{x_2} + V_m\ m_{x_2} \Big) a_{34,x_1}
$$

where $V_\eta$ and $V_m$ are built from the second derivatives of $F_1,F_2$.

### DSE ↔︎ BSE bridge
In the deterministic-volatility (Gaussian) specialization, evaluating these blocks in closed form reduces
exactly to the compact BSM Sensitivity Engine (BSE) representation. In particular:
- the **cross** part uses only first-derivative blocks and captures the symmetric bilinear coupling between
  $a_{12}$ and $a_{34}$;
- the **non-cross** part is the systematic “weight-motion” component that collects the dependence of the
  Jacobian blocks on $(m,\eta)$.

The engine exposes this split as `P2_cross` and `P2_non`.

---

## 3. Implemented backends (DistributionBlocks)

The engine requires a model backend implementing the `DistributionBlocks` protocol:

- **Deterministic-volatility BSM (Gaussian blocks)**: closed-form CDF/PDF derivatives.
- **Local volatility PDE**: Crank–Nicolson backward PDE for digital probabilities under $\mathbb Q$ and $\mathbb Q^+$,F with finite-difference derivatives in $(m,\eta)$.
- **Heston Monte Carlo**: Euler/full-truncation style simulation under $\mathbb Q$ and $\mathbb Q^+$, with
  smoothed indicator probabilities and finite-difference derivatives. In this experiment, we take $Z_t=(S_t,V_t)$ as a state component and $\sigma(t,Z_t)=\sqrt{V_t}$.
- **General Monte Carlo (Likelihood-Ratio / Malliavin hooks)**: a generic simulator interface that can
  provide score-function (LR) terms for $\partial_\eta$ and $\partial^2_{\eta\eta}$. When scores are not
  provided, it falls back to finite differences.

---

## 4. Installation

```bash
pip install -e .
```

Requirements: Python ≥ 3.9 and NumPy.

---

## 5. Quick start

### BSM Gaussian blocks
```python
from distributional_sensitivity_engine import (
    BSMGaussianBlocks, BSMPrimitives,
    dse_price, dse_first_derivative, dse_mixed_partial_components,
)

prims = BSMPrimitives(S=100, K=100, tau=0.5, r=0.03, q=0.01, sigma=0.2, D=+1)
blocks = BSMGaussianBlocks()

V = dse_price(prims, blocks)
delta = dse_first_derivative(prims, "S", blocks)
gamma = dse_mixed_partial_components(prims, "S", "S", blocks)
print(V, delta, gamma["total"])
```

### Run the examples
```bash
python examples/bsm_gaussian_blocks_demo.py
python examples/local_vol_pde_demo.py
python examples/heston_mc_demo.py
python examples/general_mc_lr_malliavin_demo.py
```

Each demo prints the model name and the computed quantities.

---

## 6. Repository layout

```
.
├─ src/distributional_sensitivity_engine/
│  ├─ __init__.py
│  ├─ __main__.py
│  └─ engine.py
├─ examples/
│  ├─ bsm_gaussian_blocks_demo.py
│  ├─ local_vol_pde_demo.py
│  ├─ heston_mc_demo.py
│  └─ general_mc_lr_malliavin_demo.py
└─ pyproject.toml
```

---

## 7. License

MIT (see `LICENSE`).
