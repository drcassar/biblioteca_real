"""
Notas do autor:

Este script contém as funções necessárias para implementação de uma vanilla-PINN para a resolução de um problema direto estacionário do tipo da equação de Laplace, com condições de contorno retangulares.

A solução analítica é particular para o problema introduzido no notebook 01_direct_stationary_vanilla.ipynb
"""

import torch
import numpy as np

# Definição do local onde o código serpa executado. Por padrão, gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cálculo do resíduo da PDE
def pde_residual(model, X):
    
    u = model(X)

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

    res = u_xx + u_yy

    return res

def loss_function(model, X_colloc, X_bc, U_bc, w_data, w_pde):
    """
    Calcula a função de perda total da PINN.

    Args:
        model:  rede neural
        X_col:  pontos de colocação, shape (N_c, 2)
        X_bc:   pontos de contorno, shape (4*N_b, 2)
        U_bc:   valores prescritos nas CCs, shape (4*N_b, 1)
        w_data: peso da perda de dados
        w_pde:  peso da perda física

    Retorna:
        loss:      perda total
        loss_data: perda de dados (para monitoramento)
        loss_pde:  perda física (para monitoramento)
    """

    # loss física
    residual = pde_residual(model, X_colloc)
    loss_pde = torch.mean(residual ** 2)

    # loss dados
    U_pred = model(X_bc)
    loss_data = torch.mean((U_pred - U_bc) ** 2)

    # loss total
    loss = w_data * loss_data + w_pde * loss_pde

    return loss, loss_data, loss_pde

def train(model, optimizer, X_col, X_bc, U_bc, n_epochs, w_data=1.0, w_pde=1.0):
    """
    Loop de treinamento da PINN.

    Args:
        model:    rede neural
        optimizer: otimizador
        X_col:    pontos de colocação, shape (N_c, 2)
        X_bc:     pontos de contorno, shape (4*N_b, 2)
        U_bc:     valores prescritos nas CCs, shape (4*N_b, 1)
        n_epochs: número de épocas
        w_data:   peso da perda de dados
        w_pde:    peso da perda física

    Retorna:
        history: dicionário com o histórico de perdas
                 {'loss': [], 'loss_data': [], 'loss_pde': []}
    """

    history = {'loss': [], 'loss_data': [], 'loss_pde': []}

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_data, loss_pde = loss_function(
            model, X_col, X_bc, U_bc, w_data, w_pde
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_data'].append(loss_data.item())
        history['loss_pde'].append(loss_pde.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss data: {loss_data.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e}')

    return history

def analytical_solution(X):
    """
    Solução analítica da equação de Laplace 2D com as condições de contorno:
        u(x, 0) = 0
        u(x, 1) = sin(pi * x)
        u(0, y) = 0
        u(1, y) = 0

    Args:
        X: tensor de shape (N, 2) com as coordenadas (x, y)

    Retorna:
        u: tensor de shape (N, 1) com os valores analíticos
    """
    x = X[:, 0].unsqueeze(1)
    y = X[:, 1].unsqueeze(1)

    u = (torch.sinh(torch.pi * y) / torch.sinh(torch.tensor(torch.pi))) * torch.sin(torch.pi * x)

    return u

def evaluate(model, analytical_fn, device, n_grid=100, slices=None):
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
        U_pred = model(X_flat).cpu().numpy().reshape(n_grid, n_grid)
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
            U_pred_slices.append(model(X_slice).cpu().numpy().ravel())
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