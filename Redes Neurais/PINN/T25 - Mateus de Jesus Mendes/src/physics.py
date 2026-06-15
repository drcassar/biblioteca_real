"""
src/physics.py
==============
Física do pêndulo simples e cálculo de resíduos via diferenciação automática.

Equação governante (não-linear):
    θ''(t) + (g/L) sin(θ(t)) = 0

Aproximação linear (pequenas amplitudes):
    θ''(t) + ω² θ(t) = 0,   ω = √(g/L)

Solução analítica da aproximação linear:
    θ(t) = θ₀ cos(ωt) + (θ̇₀ / ω) sin(ωt)

O resíduo da PINN é definido sobre a equação não-linear, o que permite
capturar comportamentos que a aproximação linear não consegue reproduzir.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from typing import Tuple


# ── Constantes físicas padrão ──────────────────────────────────────────────
G_EARTH: float = 9.81   # aceleração gravitacional [m/s²]
L_DEFAULT: float = 1.0  # comprimento do fio [m]


# ══════════════════════════════════════════════════════════════════════════════
# Solução de referência (scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _pendulum_rhs(t: float, y: list[float], g: float, L: float) -> list[float]:
    """RHS do sistema de 1ª ordem equivalente ao pêndulo não-linear."""
    theta, dtheta = y
    return [dtheta, -(g / L) * np.sin(theta)]


def solve_pendulum_numerical(
    t_span: Tuple[float, float],
    theta0: float,
    dtheta0: float,
    g: float = G_EARTH,
    L: float = L_DEFAULT,
    n_points: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solução numérica de referência via Dormand-Prince RK45 (scipy).

    Utilizada como "verdade" para comparação com a PINN e para geração
    de dados sintéticos no problema inverso.

    Returns
    -------
    t : np.ndarray   — instantes de tempo
    theta : np.ndarray — ângulo θ(t) [rad]
    dtheta : np.ndarray — velocidade angular θ'(t) [rad/s]
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        _pendulum_rhs,
        t_span,
        [theta0, dtheta0],
        args=(g, L),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp falhou: {sol.message}")
    return sol.t, sol.y[0], sol.y[1]


# ══════════════════════════════════════════════════════════════════════════════
# Solução analítica (aproximação linear)
# ══════════════════════════════════════════════════════════════════════════════

def analytical_solution_linear(
    t: np.ndarray,
    theta0: float,
    dtheta0: float,
    g: float = G_EARTH,
    L: float = L_DEFAULT,
) -> np.ndarray:
    """
    Solução analítica da EDO linearizada do pêndulo.

    Derivação:
        Para |θ| << 1, sin(θ) ≈ θ, logo:
            θ'' + ω²θ = 0,  ω = √(g/L)
        A solução geral é:
            θ(t) = C₁ cos(ωt) + C₂ sin(ωt)
        Aplicando CIs θ(0) = θ₀ e θ'(0) = θ̇₀:
            C₁ = θ₀,  C₂ = θ̇₀ / ω
        Portanto:
            θ(t) = θ₀ cos(ωt) + (θ̇₀/ω) sin(ωt)

    Nota: válida apenas para θ₀ ≲ 15°. Para ângulos maiores, use a solução
    numérica como referência.
    """
    omega = np.sqrt(g / L)
    return theta0 * np.cos(omega * t) + (dtheta0 / omega) * np.sin(omega * t)


def analytical_dtheta_linear(
    t: np.ndarray,
    theta0: float,
    dtheta0: float,
    g: float = G_EARTH,
    L: float = L_DEFAULT,
) -> np.ndarray:
    """Derivada temporal da solução analítica linear."""
    omega = np.sqrt(g / L)
    return -omega * theta0 * np.sin(omega * t) + dtheta0 * np.cos(omega * t)


# ══════════════════════════════════════════════════════════════════════════════
# Resíduo da PINN e termos da função de perda
# ══════════════════════════════════════════════════════════════════════════════

def compute_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    g_param: torch.Tensor | float,
    L: float = L_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula o resíduo da EDO não-linear do pêndulo via autograd:

        R(t) = θ_NN''(t) + (g/L) sin(θ_NN(t))

    Se a rede for a solução exata, R(t) = 0 em todo o domínio.

    Parameters
    ----------
    model : PINN
        Rede neural que aproxima θ(t).
    t : torch.Tensor, shape (N, 1), requires_grad=True
        Pontos de colocação no domínio temporal.
    g_param : torch.Tensor ou float
        Parâmetro g (pode ser treinável no problema inverso).
    L : float
        Comprimento do pêndulo [m].

    Returns
    -------
    residual : torch.Tensor — R(t) nos pontos de colocação
    theta    : torch.Tensor — predição θ_NN(t)
    dtheta   : torch.Tensor — predição θ_NN'(t)
    """
    if not t.requires_grad:
        t = t.requires_grad_(True)

    theta = model(t)

    dtheta = torch.autograd.grad(
        theta, t,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2theta = torch.autograd.grad(
        dtheta, t,
        grad_outputs=torch.ones_like(dtheta),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = d2theta + (g_param / L) * torch.sin(theta)
    return residual, theta, dtheta


def loss_physics(residual: torch.Tensor) -> torch.Tensor:
    """MSE do resíduo nos pontos de colocação."""
    return torch.mean(residual ** 2)


def loss_initial_conditions(
    model: torch.nn.Module,
    theta0: float,
    dtheta0: float,
) -> torch.Tensor:
    """
    Penaliza o desvio das condições iniciais:

        L_CI = (θ_NN(0) - θ₀)² + (θ_NN'(0) - θ̇₀)²

    Note que ambas θ(0) e θ'(0) são impostas como soft constraints.
    """
    t0 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    theta_pred = model(t0)

    dtheta_pred = torch.autograd.grad(
        theta_pred, t0,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=True,
    )[0]

    l_pos = (theta_pred - theta0) ** 2
    l_vel = (dtheta_pred - dtheta0) ** 2
    return l_pos.squeeze() + l_vel.squeeze()


def loss_data(
    model: torch.nn.Module,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
) -> torch.Tensor:
    """MSE entre a predição da rede e observações (problema inverso)."""
    theta_pred = model(t_obs)
    return torch.mean((theta_pred - theta_obs) ** 2)
