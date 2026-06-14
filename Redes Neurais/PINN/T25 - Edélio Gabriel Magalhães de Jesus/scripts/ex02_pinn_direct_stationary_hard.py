"""
Notas do autor:

Este script contém as funções necessárias para implementação de uma hard-PINN para a resolução de um problema direto estacionário do tipo da equação de Laplace, com condições de contorno retangulares.

A solução analítica é particular para o problema introduzido no notebook 01_direct_stationary_vanilla.ipynb
"""

import torch
import numpy as np
from geral_functions import sample_collocation_rectangular

# Definindo a função de tentativa, com o acréscimo das funções auxiliares à saída bruta da rede
def trial_solution(model, X):
    """
    Aplica a função de tentativa para satisfazer as CCs por construção.

    u(x, y) = A(x, y) + B(x, y) * N(x, y)

    onde:
        A(x, y) = y * sin(pi * x)        — satisfaz as CCs
        B(x, y) = x(1-x) * y(1-y)       — se anula na fronteira
        N(x, y) = saída bruta da rede

    Args:
        model: rede neural
        X:     tensor de shape (N, 2)

    Retorna:
        u: tensor de shape (N, 1)
    """
    x = X[:, 0].unsqueeze(1)
    y = X[:, 1].unsqueeze(1)

    N = model(X)
    A = y * torch.sin(torch.pi * x)
    B = x * (1 - x) * y * (1 - y)

    return A + B * N

def pde_residual_hard(model, X):
    """
    Calcula o resíduo da equação de Laplace usando a função de tentativa.

    Args:
        model: rede neural
        X:     tensor de shape (N_c, 2) com requires_grad=True

    Retorna:
        res: tensor de shape (N_c, 1)
    """
    u = trial_solution(model, X)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0].unsqueeze(1)
    u_y = grads[:, 1].unsqueeze(1)

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    u_yy = torch.autograd.grad(
        outputs=u_y,
        inputs=X,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1].unsqueeze(1)

    return u_xx + u_yy

# Função de perda a
def loss_fn_hard(model, X_col):
    """
    Função de perda da hard-PINN — apenas o resíduo da EDP.

    Args:
        model: rede neural
        X_col: pontos de colocação, shape (N_c, 2)

    Retorna:
        loss_pde: perda física
    """
    residual = pde_residual_hard(model, X_col)
    loss_pde = torch.mean(residual ** 2)

    return loss_pde

# Loop de treinamento
def train_hard(model, optimizer, N_colloc, lb, ub, n_epochs, device='cpu'):

    history = {'loss_pde': []}

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        X_col = sample_collocation_rectangular(N_colloc, lb, ub, device)

        loss_pde = loss_fn_hard(model, X_col)

        loss_pde.backward()
        optimizer.step()

        history['loss_pde'].append(loss_pde.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss PDE: {loss_pde.item():.2e}')

    return history

def evaluate_hard(model, analytical_fn, device, n_grid=100, slices=None):
    """
    Avalia o modelo treinado e retorna arrays prontos para plotagem.

    Args:
        model:         rede neural treinada
        analytical_fn: função de solução analítica
        device:        dispositivo de execução
        n_grid:        resolução da grade
        slices:        valores de y para os perfis 1D (default: [0.25, 0.5, 0.75])

    Retorna dicionário com:
        'x':             array (n_grid,)
        'y':             array (n_grid,)
        'U_pred':        array (n_grid, n_grid)
        'U_ref':         array (n_grid, n_grid)
        'U_pred_slices': lista de arrays (n_grid,)
        'U_ref_slices':  lista de arrays (n_grid,)
        'l2_error':      erro L2 relativo (escalar)
    """
    if slices is None:
        slices = [0.25, 0.5, 0.75]

    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y)

    X_flat = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        U_pred = trial_solution(model, X_flat).cpu().numpy().reshape(n_grid, n_grid)
        U_ref  = analytical_fn(X_flat).cpu().numpy().reshape(n_grid, n_grid)

    # erro L2 relativo
    l2_error = np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref)

    # perfis 1D
    U_pred_slices = []
    U_ref_slices  = []

    for y_val in slices:
        X_slice = torch.tensor(
            np.stack([x, np.full_like(x, y_val)], axis=1),
            dtype=torch.float32,
            device=device
        )
        with torch.no_grad():
            U_pred_slices.append(trial_solution(model, X_slice).cpu().numpy().ravel())
            U_ref_slices.append(analytical_fn(X_slice).cpu().numpy().ravel())

    return {
        'x':             x,
        'y':             y,
        'U_pred':        U_pred,
        'U_ref':         U_ref,
        'U_pred_slices': U_pred_slices,
        'U_ref_slices':  U_ref_slices,
        'l2_error':      l2_error,
    }