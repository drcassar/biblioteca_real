"""
src/training.py
===============
Loops de treinamento para problemas direto e inverso.

Esquema híbrido de otimização:
    1. Adam (primeira ordem) — exploração rápida do espaço de parâmetros.
    2. L-BFGS (quasi-Newton) — refinamento de alta precisão perto do mínimo.

A combinação é padrão na literatura de PINNs pois:
  - Adam converge rápido mas tem precisão limitada;
  - L-BFGS tem convergência superlinear mas é sensível à inicialização.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any

from .physics import (
    compute_residual,
    loss_physics,
    loss_initial_conditions,
    loss_data,
)


# ══════════════════════════════════════════════════════════════════════════════
# Problema Direto
# ══════════════════════════════════════════════════════════════════════════════

def train_direct(
    model: nn.Module,
    t_colloc: torch.Tensor,
    theta0: float,
    dtheta0: float,
    g: float,
    L: float = 1.0,
    n_epochs_adam: int = 12_000,
    n_epochs_lbfgs: int = 500,
    lr_adam: float = 1e-3,
    w_physics: float = 1.0,
    w_ic: float = 20.0,
    verbose: bool = True,
    log_every: int = 2000,
) -> Dict[str, List[Any]]:
    """
    Treina a PINN para o problema direto: resolver θ''+(g/L)sin(θ)=0.

    Parameters
    ----------
    model : PINN
        Rede neural (não inicializada externamente para ser reutilizável).
    t_colloc : torch.Tensor, shape (N, 1)
        Pontos de colocação (devem ter requires_grad=True).
    theta0, dtheta0 : float
        Condições iniciais θ(0) e θ'(0).
    g, L : float
        Parâmetros físicos conhecidos.
    w_physics, w_ic : float
        Pesos das perdas física e de condição inicial.
    n_epochs_adam, n_epochs_lbfgs : int
        Épocas para cada otimizador.
    verbose : bool
        Imprime progresso se True.
    log_every : int
        Frequência de impressão (épocas Adam).

    Returns
    -------
    history : dict com chaves 'epoch', 'loss', 'loss_phys', 'loss_ic'
    """
    g_tensor = torch.tensor(float(g), dtype=torch.float32)
    history: Dict[str, List] = {
        "epoch": [], "loss": [], "loss_phys": [], "loss_ic": []
    }

    # ── Adam ──────────────────────────────────────────────────────────────────
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    model.train()

    t_c = t_colloc.clone().detach().requires_grad_(True)

    for epoch in range(n_epochs_adam):
        optimizer_adam.zero_grad()

        residual, _, _ = compute_residual(model, t_c, g_tensor, L)
        l_phys = w_physics * loss_physics(residual)
        l_ic = w_ic * loss_initial_conditions(model, theta0, dtheta0)
        loss = l_phys + l_ic

        loss.backward()
        optimizer_adam.step()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["loss_phys"].append(l_phys.item())
        history["loss_ic"].append(l_ic.item())

        if verbose and (epoch % log_every == 0 or epoch == n_epochs_adam - 1):
            print(
                f"[Adam] Época {epoch:6d} | "
                f"Loss Total: {loss.item():.3e} | "
                f"Física: {l_phys.item():.3e} | "
                f"CI: {l_ic.item():.3e}"
            )

    # ── L-BFGS ────────────────────────────────────────────────────────────────
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=n_epochs_lbfgs,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure_direct():
        optimizer_lbfgs.zero_grad()
        residual, _, _ = compute_residual(model, t_c, g_tensor, L)
        l_phys = w_physics * loss_physics(residual)
        l_ic = w_ic * loss_initial_conditions(model, theta0, dtheta0)
        loss = l_phys + l_ic
        loss.backward()
        return loss

    if verbose:
        print(f"\n{'─'*60}")
        print("[L-BFGS] Iniciando refinamento quasi-Newton...")

    loss_val = optimizer_lbfgs.step(closure_direct)

    if verbose:
        print(f"[L-BFGS] Perda final: {loss_val.item():.3e}")
        print(f"{'─'*60}\n")

    model.eval()
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Problema Inverso
# ══════════════════════════════════════════════════════════════════════════════

def train_inverse(
    model: nn.Module,
    g_param: nn.Parameter,
    t_colloc: torch.Tensor,
    t_obs: torch.Tensor,
    theta_obs: torch.Tensor,
    theta0: float,
    dtheta0: float,
    L: float = 1.0,
    n_epochs_adam: int = 15_000,
    n_epochs_lbfgs: int = 500,
    lr_adam: float = 1e-3,
    lr_g: float = 5e-2,
    w_physics: float = 1.0,
    w_data: float = 15.0,
    w_ic: float = 10.0,
    verbose: bool = True,
    log_every: int = 2000,
) -> Dict[str, List[Any]]:
    """
    Treina a PINN para o problema inverso: descobrir g a partir de dados.

    g_param deve ser criado como:
        g_param = torch.nn.Parameter(torch.tensor(g_inicial))

    O parâmetro é tratado como variável treinável junto com os pesos da rede.
    A perda de dados guia a estimativa de g; a perda física garante consistência
    com a equação do pêndulo.

    Returns
    -------
    history : dict com chaves 'epoch', 'loss', 'loss_phys', 'loss_data',
              'loss_ic', 'g_value'
    """
    history: Dict[str, List] = {
        "epoch": [], "loss": [], "loss_phys": [],
        "loss_data": [], "loss_ic": [], "g_value": [],
    }

    # ── Adam ──────────────────────────────────────────────────────────────────
    optimizer_adam = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": lr_adam},
            {"params": [g_param], "lr": lr_g},
        ]
    )
    model.train()
    t_c = t_colloc.clone().detach().requires_grad_(True)

    for epoch in range(n_epochs_adam):
        optimizer_adam.zero_grad()

        residual, _, _ = compute_residual(model, t_c, g_param, L)
        l_phys = w_physics * loss_physics(residual)
        l_ic = w_ic * loss_initial_conditions(model, theta0, dtheta0)
        l_data = w_data * loss_data(model, t_obs, theta_obs)
        loss = l_phys + l_data + l_ic

        loss.backward()
        optimizer_adam.step()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["loss_phys"].append(l_phys.item())
        history["loss_data"].append(l_data.item())
        history["loss_ic"].append(l_ic.item())
        history["g_value"].append(g_param.item())

        if verbose and (epoch % log_every == 0 or epoch == n_epochs_adam - 1):
            print(
                f"[Adam] Época {epoch:6d} | "
                f"Loss: {loss.item():.3e} | "
                f"g estimado: {g_param.item():.5f} m/s²"
            )

    # ── L-BFGS ────────────────────────────────────────────────────────────────
    optimizer_lbfgs = torch.optim.LBFGS(
        list(model.parameters()) + [g_param],
        lr=1.0,
        max_iter=n_epochs_lbfgs,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure_inverse():
        optimizer_lbfgs.zero_grad()
        residual, _, _ = compute_residual(model, t_c, g_param, L)
        l_phys = w_physics * loss_physics(residual)
        l_ic = w_ic * loss_initial_conditions(model, theta0, dtheta0)
        l_data = w_data * loss_data(model, t_obs, theta_obs)
        loss = l_phys + l_data + l_ic
        loss.backward()
        return loss

    if verbose:
        print(f"\n{'─'*60}")
        print(f"[L-BFGS] Refinando... g atual: {g_param.item():.5f} m/s²")

    optimizer_lbfgs.step(closure_inverse)

    history["g_value"].append(g_param.item())

    if verbose:
        print(f"[L-BFGS] g final estimado: {g_param.item():.6f} m/s²")
        print(f"{'─'*60}\n")

    model.eval()
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Avaliação
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    t_np: "np.ndarray",  # type: ignore[name-defined]
) -> "np.ndarray":  # type: ignore[name-defined]
    """
    Avalia a PINN em um array numpy de instantes temporais.

    Returns theta_pinn : np.ndarray
    """
    t_tensor = torch.tensor(t_np, dtype=torch.float32).view(-1, 1)
    theta_pred = model(t_tensor)
    return theta_pred.numpy().flatten()


def evaluate_model_with_grad(
    model: nn.Module,
    t_np: "np.ndarray",  # type: ignore[name-defined]
) -> tuple:
    """
    Avalia a PINN e calcula θ'(t) via autograd (para o retrato de fase).

    Returns (theta_pinn, dtheta_pinn) : two np.ndarrays
    """
    t_tensor = torch.tensor(t_np, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    theta_pred = model(t_tensor)
    dtheta_pred = torch.autograd.grad(
        theta_pred, t_tensor,
        grad_outputs=torch.ones_like(theta_pred),
        create_graph=False,
    )[0]
    return (
        theta_pred.detach().numpy().flatten(),
        dtheta_pred.detach().numpy().flatten(),
    )
