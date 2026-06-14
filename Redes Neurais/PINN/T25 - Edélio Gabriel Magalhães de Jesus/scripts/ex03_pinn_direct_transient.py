"""
Notas do autor:

Este script contém as funções necessárias para implementar uma PINN na resolução de um problema direto transiente unidimensional
submetido à condições de contorno do tipo Dirichlet
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Definição do local onde o código serpa executado. Por padrão, gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pde_residual_burguers(model, X, nu):
    """
    Calcula o resíduo da equação de Burgers 1D.

    u_t + u * u_x - nu * u_xx = 0

    Args:
        model: rede neural
        X:     tensor (N_c, 2) com requires_grad=True
               X[:, 0] = x, X[:, 1] = t
        nu:    viscosidade cinemática

    Retorna:
        res: tensor (N_c, 1)
    """
    u = model(X)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0] 

    u_x = grads[:, 0].unsqueeze(1)
    u_t = grads[:, 1].unsqueeze(1)  

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    res = u_t + u * u_x - nu * u_xx

    return res

def loss_function_burguers(model, X_col, X_ic, U_ic, X_bc, U_bc, nu, w_ic, w_bc, w_pde):
    """
    Função de perda da PINN para a equação de Burgers.

    Args:
        model:  rede neural
        X_col:  pontos de colocação, shape (N_c, 2)
        X_ic:   pontos da CI, shape (N_ic, 2)
        U_ic:   valores da CI, shape (N_ic, 1)
        X_bc:   pontos das CCs, shape (2*N_bc, 2)
        U_bc:   valores das CCs, shape (2*N_bc, 1)
        nu:     viscosidade cinemática
        w_ic:   peso da perda de CI
        w_bc:   peso da perda de CC
        w_pde:  peso da perda física

    Retorna:
        loss:     perda total
        loss_ic:  perda da CI
        loss_bc:  perda das CCs
        loss_pde: perda física
    """
    
    # perda física 
    residual = pde_residual_burguers(model, X_col, nu)
    loss_pde = torch.mean(residual ** 2)

    # perda CI
    U_ic_pred = model(X_ic)
    loss_ic = torch.mean((U_ic_pred - U_ic) ** 2)

    # perda CC
    U_cc_pred = model(X_bc)
    loss_bc = torch.mean((U_cc_pred - U_bc) ** 2)

    # perda total 
    loss = w_ic * loss_ic + w_bc * loss_bc + w_pde * loss_pde 

    return loss, loss_ic, loss_bc, loss_pde

def train_burgers(model, optimizer, X_col, X_ic, U_ic, X_bc, U_bc,
                  nu, n_epochs, w_ic=1.0, w_bc=1.0, w_pde=1.0):
    """
    Loop de treinamento da PINN para a equação de Burgers.

    Args:
        model:     rede neural
        optimizer: otimizador
        X_col:     pontos de colocação, shape (N_c, 2)
        X_ic:      pontos da CI, shape (N_ic, 2)
        U_ic:      valores da CI, shape (N_ic, 1)
        X_bc:      pontos das CCs, shape (2*N_bc, 2)
        U_bc:      valores das CCs, shape (2*N_bc, 1)
        nu:        viscosidade cinemática
        n_epochs:  número de épocas
        w_ic:      peso da CI
        w_bc:      peso das CCs
        w_pde:     peso da perda física

    Retorna:
        history: dicionário com o histórico de perdas
    """

    history = {
        'loss':     [],
        'loss_ic':  [],
        'loss_bc':  [],
        'loss_pde': [],
    }

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_ic, loss_bc, loss_pde = loss_function_burguers(
            model, X_col, X_ic, U_ic, X_bc, U_bc,
            nu, w_ic, w_bc, w_pde
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_pde'].append(loss_pde.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss IC: {loss_ic.item():.2e} | '
                  f'Loss BC: {loss_bc.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e}')

    return history

def numerical_solution_burgers(nu=0.01/np.pi, nx=256, nt=100):
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]

    # condição inicial
    u0 = -np.sin(np.pi * x)
    u0[0] = 0.0
    u0[-1] = 0.0

    def rhs(t, u):
        u = u.copy()

        # Dirichlet homogênea
        u[0] = 0.0
        u[-1] = 0.0

        # Fluxo convectivo conservativo: f(u) = u^2/2
        f = 0.5 * u**2

        # Fluxo numérico local Lax-Friedrichs / Rusanov
        # f_hat_{i+1/2} = 0.5(f_i + f_{i+1}) - 0.5 a_{i+1/2}(u_{i+1} - u_i)
        a = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
        f_hat = 0.5 * (f[:-1] + f[1:]) - 0.5 * a * (u[1:] - u[:-1])

        dudt = np.zeros_like(u)

        # derivada espacial do termo convectivo
        dudt[1:-1] += -(f_hat[1:] - f_hat[:-1]) / dx

        # termo difusivo central
        dudt[1:-1] += nu * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2

        # mantém as bordas fixas
        dudt[0] = 0.0
        dudt[-1] = 0.0

        return dudt

    # Limita o passo interno para não perder a dinâmica espacial
    u_max = np.max(np.abs(u0)) + 1e-12
    max_step = 0.5 * dx / u_max

    sol = solve_ivp(
        rhs,
        t_span=(0.0, 1.0),
        y0=u0,
        t_eval=t,
        method="BDF",
        rtol=1e-6,
        atol=1e-8,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp falhou: {sol.message}")

    U = sol.y  # shape: (nx, nt)
    return x, t, U

def evaluate_burgers(model, x_ref, t_ref, U_ref, device, snapshots=None):
    """
    Avalia o modelo treinado e retorna arrays prontos para plotagem.

    Args:
        model:     rede neural treinada
        x_ref:     array (nx,)    — coordenadas espaciais da referência
        t_ref:     array (nt,)    — instantes de tempo da referência
        U_ref:     array (nx, nt) — solução de referência
        device:    dispositivo de execução
        snapshots: lista de instantes de tempo para os perfis 1D
                   (default: [0.25, 0.50, 0.75])

    Retorna dicionário com:
        'x':               array (nx,)
        't':               array (nt,)
        'U_pred':          array (nx, nt) — solução predita na grade de referência
        'U_ref':           array (nx, nt) — solução de referência
        'U_pred_snaps':    lista de arrays (nx,) — predição nos snapshots
        'U_ref_snaps':     lista de arrays (nx,) — referência nos snapshots
        'snap_times':      lista de floats — instantes dos snapshots
        'l2_error':        erro L2 relativo (escalar)
    """
    if snapshots is None:
        snapshots = [0.25, 0.50, 0.75]

    # grade completa (nx * nt, 2)
    X_grid, T_grid = np.meshgrid(x_ref, t_ref, indexing='ij')
    X_flat = torch.tensor(
        np.stack([X_grid.ravel(), T_grid.ravel()], axis=1),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        U_pred = model(X_flat).cpu().numpy().reshape(len(x_ref), len(t_ref))

    # erro L2 relativo
    l2_error = np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref)

    # snapshots temporais
    U_pred_snaps = []
    U_ref_snaps  = []
    snap_times   = []

    for t_snap in snapshots:
        # índice do instante mais próximo em t_ref
        idx = np.argmin(np.abs(t_ref - t_snap))
        snap_times.append(t_ref[idx])

        U_pred_snaps.append(U_pred[:, idx])
        U_ref_snaps.append(U_ref[:, idx])

    return {
        'x':            x_ref,
        't':            t_ref,
        'U_pred':       U_pred,
        'U_ref':        U_ref,
        'U_pred_snaps': U_pred_snaps,
        'U_ref_snaps':  U_ref_snaps,
        'snap_times':   snap_times,
        'l2_error':     l2_error,
    }