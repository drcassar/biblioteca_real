"""
Este script contém todas as funções e estruturas gerais para a implementação dos exemplos, evitando repetição desnecessária

- Nota do autor
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# Classe de construção da arquitetura da rede - tipo MLP (Multilayer Perceptron)
class PINN(nn.Module):
    """
    Classe para definir a arquitetura MLP
    Recebe como parâmetros de arquitetura:

        n_inputs: Quantidade de dados de entrada
        n_outputs: Quantidade de dados de saída
        n_hidden: Quantidade de neurônios nas camadas ocultas
        n_layers: Quantidade de camadas
        activation: Função de ativação

    Retorna: O vetor X após o passo forward
    """
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers, activation):
        super().__init__()

        layers = []

        layers.append(nn.Linear(n_inputs, n_hidden))
        layers.append(activation())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation())

        layers.append(nn.Linear(n_hidden, n_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        X = self.net(X)
        return X
    
# Função de amostragem dos pontos de colocação - internos    
def sample_collocation_rectangular(N_c, lb, ub, device):
    """
    Amostra N_c pontos aleatórios no interior de um domínio retangular

    Args:
        N_c:    número de pontos de colocação
        lb:     limite inferior de cada dimensão — lista com len = número de dimensões
                ex: [0, 0] para domínio 2D, [-1, 0] para domínio com x negativo
        ub:     limite superior de cada dimensão — lista com len = número de dimensões
                ex: [1, 1] para domínio 2D
        device: dispositivo de execução

    Retorna tensor de shape (N_c, n_dims) com requires_grad=True,
    onde n_dims = len(lb).
    """
    lb = torch.tensor(lb, dtype=torch.float32, device=device)
    ub = torch.tensor(ub, dtype=torch.float32, device=device)

    X = torch.rand(int(N_c), len(lb), device=device) * (ub - lb) + lb
    X.requires_grad_(True)

    return X

# Função para amostragem dos pontos no domínio
def sample_boundary_rectangular_stationary(N_b, lb, ub, bc_fns, device):
    """
    Amostra N_b pontos em cada face do domínio [lb, ub]^2 e calcula
    os valores de CC correspondentes.

    Args:
        N_b:    número de pontos por face
        lb:     limite inferior do domínio (float)
        ub:     limite superior do domínio (float)
        bc_fns: dicionário com as funções de CC para cada face:
                {'bottom': fn, 'top': fn, 'left': fn, 'right': fn}
                cada fn recebe um tensor 1D e retorna um tensor 1D
        device: dispositivo de execução (cpu ou cuda)

    Retorna:
        X_bc: tensor de shape (4 * N_b, 2) com as coordenadas
        U_bc: tensor de shape (4 * N_b, 1) com os valores de CC
    """

    lb = torch.tensor(lb, dtype=torch.float32, device=device)
    ub = torch.tensor(ub, dtype=torch.float32, device=device)

    lb_x, lb_y = lb[0], lb[1]
    ub_x, ub_y = ub[0], ub[1]

    t = torch.rand(N_b, device=device) * (ub_x - lb_x) + lb_x

    # face inferior: y = lb_y, x livre
    X_bottom = torch.stack([t, torch.full_like(t, lb_y)], dim=1)
    U_bottom = bc_fns['bottom'](t).view(-1, 1)

    # face superior: y = ub_y, x livre
    X_top = torch.stack([t, torch.full_like(t, ub_y)], dim=1)
    U_top = bc_fns['top'](t).view(-1, 1)

    # face esquerda: x = lb_x, y livre
    s = torch.rand(N_b, device=device) * (ub_y - lb_y) + lb_y
    X_left = torch.stack([torch.full_like(s, lb_x), s], dim=1)
    U_left = bc_fns['left'](s).view(-1, 1)

    # face direita: x = ub_x, y livre
    X_right = torch.stack([torch.full_like(s, ub_x), s], dim=1)
    U_right = bc_fns['right'](s).view(-1, 1)

    X_bc = torch.cat([X_bottom, X_top, X_left, X_right], dim=0)
    U_bc = torch.cat([U_bottom, U_top, U_left, U_right], dim=0)

    return X_bc, U_bc

def sample_boundary_rectangular_transient(
    N_ic, N_bc,
    x_lb, x_ub,
    t_lb, t_ub,
    ic_fn, bc_fns,
    device
):
    """
    Burgers 1D transiente:
      - CI:  u(x, t_lb) = ic_fn(x)
      - BCs: u(x_lb, t) e u(x_ub, t)
    """
    # condição inicial: t = t_lb, x livre
    x_ic = torch.rand(N_ic, device=device) * (x_ub - x_lb) + x_lb
    t_ic = torch.full_like(x_ic, t_lb)

    X_ic = torch.stack([x_ic, t_ic], dim=1)
    U_ic = ic_fn(x_ic).view(-1, 1)

    # contorno esquerdo: x = x_lb, t livre
    t_left = torch.rand(N_bc, device=device) * (t_ub - t_lb) + t_lb
    X_left = torch.stack([torch.full_like(t_left, x_lb), t_left], dim=1)
    U_left = bc_fns['left'](t_left).view(-1, 1)

    # contorno direito: x = x_ub, t livre
    t_right = torch.rand(N_bc, device=device) * (t_ub - t_lb) + t_lb
    X_right = torch.stack([torch.full_like(t_right, x_ub), t_right], dim=1)
    U_right = bc_fns['right'](t_right).view(-1, 1)

    X_bc = torch.cat([X_left, X_right], dim=0)
    U_bc = torch.cat([U_left, U_right], dim=0)

    return X_ic, U_ic, X_bc, U_bc

def sample_boundary_rectangular_transient_2d(N_ic, N_bc, lb_x, ub_x,
                                              lb_y, ub_y, lb_t, ub_t,
                                              ic_fn, device):
    """
    Amostra pontos de CI e CCs para problemas transientes 2D.

    A entrada da rede é (x, y, t) — shape (N, 3).

    Args:
        N_ic:   número de pontos na CI
        N_bc:   número de pontos por face espacial
        lb_x:   limite inferior em x
        ub_x:   limite superior em x
        lb_y:   limite inferior em y
        ub_y:   limite superior em y
        lb_t:   limite inferior em t
        ub_t:   limite superior em t
        ic_fn:  função da CI — recebe (x, y) tensores 1D, retorna c(x,y,0)
        device: dispositivo de execução

    Retorna:
        X_ic:  tensor (N_ic, 3)  — pontos da CI
        C_ic:  tensor (N_ic, 1)  — valores da CI
        X_bc:  tensor (4*N_bc, 3) — pontos das CCs
        C_bc:  tensor (4*N_bc, 1) — valores das CCs (zeros)
    """
    # ── CI — t=0, (x,y) livres ────────────────────────────────────────────────
    x_ic = torch.rand(N_ic, device=device) * (ub_x - lb_x) + lb_x
    y_ic = torch.rand(N_ic, device=device) * (ub_y - lb_y) + lb_y
    t_ic = torch.zeros(N_ic, device=device)

    X_ic = torch.stack([x_ic, y_ic, t_ic], dim=1)
    C_ic = ic_fn(x_ic, y_ic).view(-1, 1)

    # ── CCs — quatro faces espaciais, t livre ─────────────────────────────────
    t_bc = torch.rand(N_bc, device=device) * (ub_t - lb_t) + lb_t
    s_bc = torch.rand(N_bc, device=device)

    # face x=lb_x
    x_left = torch.full_like(t_bc, lb_x)
    y_left = s_bc * (ub_y - lb_y) + lb_y
    X_left = torch.stack([x_left, y_left, t_bc], dim=1)

    # face x=ub_x
    x_right = torch.full_like(t_bc, ub_x)
    y_right = s_bc * (ub_y - lb_y) + lb_y
    X_right = torch.stack([x_right, y_right, t_bc], dim=1)

    # face y=lb_y
    x_bot = s_bc * (ub_x - lb_x) + lb_x
    y_bot = torch.full_like(t_bc, lb_y)
    X_bot = torch.stack([x_bot, y_bot, t_bc], dim=1)

    # face y=ub_y
    x_top = s_bc * (ub_x - lb_x) + lb_x
    y_top = torch.full_like(t_bc, ub_y)
    X_top = torch.stack([x_top, y_top, t_bc], dim=1)

    X_bc = torch.cat([X_left, X_right, X_bot, X_top], dim=0)
    C_bc = torch.zeros(4 * N_bc, 1, device=device)

    return X_ic, C_ic, X_bc, C_bc