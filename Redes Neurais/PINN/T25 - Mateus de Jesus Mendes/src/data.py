"""
src/data.py
===========
Geração de dados sintéticos para o problema inverso.

Os dados são obtidos a partir da solução numérica de referência (RK45),
interpolados em instantes aleatórios e perturbados com ruído gaussiano.
Esse processo simula observações experimentais reais, onde:
  - O número de medições é limitado;
  - Há incerteza instrumental (ruído);
  - Os instantes de medição não são uniformemente espaçados.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple

from .physics import solve_pendulum_numerical, G_EARTH, L_DEFAULT


def generate_noisy_observations(
    t_span: Tuple[float, float],
    theta0: float,
    dtheta0: float,
    g_true: float = G_EARTH,
    L: float = L_DEFAULT,
    n_obs: int = 60,
    noise_std: float = 0.025,
    seed: int = 42,
    uniform_spacing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gera observações sintéticas ruidosas do pêndulo não-linear.

    Parameters
    ----------
    t_span : (t_start, t_end)
        Intervalo temporal de observação.
    theta0, dtheta0 : float
        Condições iniciais do pêndulo.
    g_true : float
        Valor real de g (desconhecido para a PINN).
    L : float
        Comprimento do fio [m].
    n_obs : int
        Número de observações.
    noise_std : float
        Desvio padrão do ruído gaussiano [rad].
    seed : int
        Semente para reprodutibilidade.
    uniform_spacing : bool
        Se True, usa espaçamento uniforme (menos realista, mas mais limpo didaticamente).

    Returns
    -------
    t_obs : torch.Tensor, shape (n_obs, 1)
    theta_obs : torch.Tensor, shape (n_obs, 1)
    """
    rng = np.random.default_rng(seed)

    # Solução de alta resolução como "verdade"
    t_full, theta_full, _ = solve_pendulum_numerical(
        t_span, theta0, dtheta0, g=g_true, L=L, n_points=2000
    )

    # Instantes de observação
    if uniform_spacing:
        t_obs = np.linspace(t_span[0], t_span[1], n_obs)
    else:
        t_obs = np.sort(rng.uniform(t_span[0], t_span[1], n_obs))

    # Interpolação + ruído gaussiano
    theta_obs = np.interp(t_obs, t_full, theta_full)
    theta_obs += rng.normal(0.0, noise_std, size=n_obs)

    t_tensor = torch.tensor(t_obs, dtype=torch.float32).view(-1, 1)
    theta_tensor = torch.tensor(theta_obs, dtype=torch.float32).view(-1, 1)
    return t_tensor, theta_tensor


def collocation_points(
    t_span: Tuple[float, float],
    n_points: int = 600,
    strategy: str = "uniform",
    seed: int = 0,
) -> torch.Tensor:
    """
    Gera pontos de colocação no domínio temporal.

    Parameters
    ----------
    strategy : {"uniform", "random", "latin"}
        - "uniform": espaçamento uniforme (mais simples).
        - "random": amostras aleatórias uniformes.
        - "latin": Latin Hypercube Sampling (melhor cobertura).

    Returns
    -------
    t_colloc : torch.Tensor, shape (n_points, 1), requires_grad=True
    """
    t0, t1 = t_span

    if strategy == "uniform":
        t = np.linspace(t0, t1, n_points)

    elif strategy == "random":
        rng = np.random.default_rng(seed)
        t = rng.uniform(t0, t1, n_points)
        t.sort()

    elif strategy == "latin":
        # Latin Hypercube: divide [t0,t1] em n_points subintervalos e amostra um ponto de cada um
        rng = np.random.default_rng(seed)
        bins = np.linspace(t0, t1, n_points + 1)
        t = np.array([rng.uniform(bins[i], bins[i + 1]) for i in range(n_points)])

    else:
        raise ValueError(f"Estratégia desconhecida: {strategy!r}")

    return torch.tensor(t, dtype=torch.float32).view(-1, 1).requires_grad_(True)
