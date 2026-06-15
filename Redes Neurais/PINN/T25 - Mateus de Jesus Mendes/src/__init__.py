"""
src — PINNs para o Pêndulo Simples
====================================
Módulos do projeto:

    model          → Arquitetura PINN (MLP configurável)
    physics        → Equação do pêndulo, resíduos, soluções analítica e numérica
    data           → Geração de dados sintéticos e pontos de colocação
    training       → Loops de treinamento Adam + L-BFGS (direto e inverso)
    visualization  → Figuras interativas Plotly
"""

from .model import PINN
from .physics import (
    solve_pendulum_numerical,
    analytical_solution_linear,
    analytical_dtheta_linear,
    compute_residual,
    loss_physics,
    loss_initial_conditions,
    loss_data,
    G_EARTH,
    L_DEFAULT,
)
from .data import generate_noisy_observations, collocation_points
from .training import train_direct, train_inverse, evaluate_model, evaluate_model_with_grad
from .visualization import (
    plot_solution_comparison,
    plot_phase_portrait,
    animate_pendulum,
    plot_loss_history,
    plot_noisy_data,
    plot_g_convergence,
    plot_inverse_comparison,
)

__all__ = [
    "PINN",
    "solve_pendulum_numerical",
    "analytical_solution_linear",
    "analytical_dtheta_linear",
    "compute_residual",
    "loss_physics",
    "loss_initial_conditions",
    "loss_data",
    "G_EARTH",
    "L_DEFAULT",
    "generate_noisy_observations",
    "collocation_points",
    "train_direct",
    "train_inverse",
    "evaluate_model",
    "evaluate_model_with_grad",
    "plot_solution_comparison",
    "plot_phase_portrait",
    "animate_pendulum",
    "plot_loss_history",
    "plot_noisy_data",
    "plot_g_convergence",
    "plot_inverse_comparison",
]
