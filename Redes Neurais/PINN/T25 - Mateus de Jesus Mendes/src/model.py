"""
src/model.py
============
Arquitetura da rede neural utilizada como PINN.

A ativação Tanh é a escolha padrão na literatura de PINNs pois é:
  - Suave e infinitamente diferenciável (essencial para resíduos de 2ª ordem)
  - Limitada em [-1, 1], o que estabiliza o treinamento
  - Bem condicionada para inicialização Xavier

Referência:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks. Journal of Computational Physics, 378, 686-707.
"""

import torch
import torch.nn as nn
from typing import Optional


class PINN(nn.Module):
    """
    Multi-Layer Perceptron configurável para uso como PINN.

    Parameters
    ----------
    n_input : int
        Dimensão da entrada (1 para problemas de valor inicial em 1D).
    n_output : int
        Dimensão da saída (1 para θ(t)).
    hidden_layers : int
        Número de camadas ocultas.
    neurons_per_layer : int
        Neurônios por camada oculta.
    activation : nn.Module, opcional
        Função de ativação. Padrão: Tanh.
    """

    def __init__(
        self,
        n_input: int = 1,
        n_output: int = 1,
        hidden_layers: int = 4,
        neurons_per_layer: int = 64,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        if activation is None:
            activation = nn.Tanh()

        layers: list[nn.Module] = [nn.Linear(n_input, neurons_per_layer), activation]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), activation]
        layers.append(nn.Linear(neurons_per_layer, n_output))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """
        Inicialização de Xavier (Glorot) para todos os pesos lineares.

        Essa inicialização preserva a variância dos gradientes ao longo das
        camadas, o que é especialmente importante em PINNs com muitas camadas,
        onde gradientes de segunda ordem podem explodir ou desaparecer.
        """
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    # ──────────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        """Retorna o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
