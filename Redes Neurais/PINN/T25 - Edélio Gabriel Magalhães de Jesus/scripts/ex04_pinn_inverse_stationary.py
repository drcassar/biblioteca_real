"""
Problema inverso estacionário — Equação de Poisson-Boltzmann planar.

Recuperar o potencial de superfície psi_0 a partir de medições esparsas
do potencial elétrico psi(x) na dupla camada elétrica.

Motivação: caracterização de superfícies carregadas em solução eletrolítica,
relevante para estabilidade de coloides, nanopartículas e interfaces
em dispositivos eletroquímicos.

Referência:
    - Gouy, G. (1910). J. Phys. Theor. Appl.
    - Chapman, D. L. (1913). Philos. Mag.
    - Wikipedia: Poisson-Boltzmann equation
"""

import torch
import numpy as np

# ── Solução analítica ──────────────────────────────────────────────────────────

def analytical_solution_pb(x, psi0):
    """
    Solução analítica de Gouy-Chapman para a equação de Poisson-Boltzmann
    planar na forma adimensional.

    psi(x) = 2 * ln((1 + C*exp(-x)) / (1 - C*exp(-x)))

    onde C = tanh(psi0 / 4)

    Args:
        x:    array (N,) — posições em unidades do comprimento de Debye
        psi0: float — potencial de superfície adimensional em x=0

    Retorna:
        psi: array (N,) — potencial adimensional
    """
    C = np.tanh(psi0 / 4)
    return 2 * np.log((1 + C * np.exp(-x)) / (1 - C * np.exp(-x)))

# ── Dados sintéticos ───────────────────────────────────────────────────────────

def generate_synthetic_data_pb(psi0_true, L, N_obs, noise_amp, device, seed=42):
    """
    Gera dados sintéticos ruidosos a partir da solução analítica de
    Gouy-Chapman.

    Args:
        psi0_true:  potencial de superfície verdadeiro
        L:          comprimento do domínio em unidades de Debye
        N_obs:      número de pontos de observação
        noise_amp:  amplitude do ruído gaussiano
        device:     dispositivo de execução
        seed:       semente para reprodutibilidade

    Retorna:
        X_obs:   tensor (N_obs, 1) — posições das observações
        Psi_obs: tensor (N_obs, 1) — potencial ruidoso
    """
    np.random.seed(seed)

    x_obs = np.random.uniform(0, L, N_obs)
    x_obs = np.sort(x_obs)

    psi_exact = analytical_solution_pb(x_obs, psi0_true)
    psi_noisy = psi_exact + np.random.normal(0, noise_amp, N_obs)

    X_obs   = torch.tensor(x_obs,    dtype=torch.float32, device=device).view(-1, 1)
    Psi_obs = torch.tensor(psi_noisy, dtype=torch.float32, device=device).view(-1, 1)

    return X_obs, Psi_obs

# ── Condições de contorno ──────────────────────────────────────────────────────

def get_boundary_points_pb(psi0, L, device):
    """
    Retorna os pontos de contorno e seus valores.

    psi(0)  = psi0  — potencial de superfície
    psi(L)  = 0     — potencial nulo longe da superfície

    Args:
        psi0:   parâmetro treinável (nn.Parameter)
        L:      comprimento do domínio
        device: dispositivo de execução

    Retorna:
        X_bc:   tensor (2, 1) — posições x=0 e x=L
        Psi_bc: tensor (2, 1) — valores psi0 e 0
    """
    X_bc   = torch.tensor([[0.0], [L]], dtype=torch.float32, device=device)
    Psi_bc = torch.cat([psi0.view(1, 1),
                        torch.zeros(1, 1, device=device)], dim=0)

    return X_bc, Psi_bc

# ── Resíduo da EDP ─────────────────────────────────────────────────────────────

def pde_residual_pb(model, X):
    """
    Calcula o resíduo da equação de Poisson-Boltzmann planar adimensional.

    d²psi/dx² - sinh(psi) = 0

    Args:
        model: rede neural
        X:     tensor (N_c, 1) com requires_grad=True

    Retorna:
        res: tensor (N_c, 1)
    """
    psi = model(X)

    psi_x = torch.autograd.grad(
        outputs=psi,
        inputs=X,
        grad_outputs=torch.ones_like(psi),
        create_graph=True
    )[0]

    psi_xx = torch.autograd.grad(
        outputs=psi_x,
        inputs=X,
        grad_outputs=torch.ones_like(psi_x),
        create_graph=True
    )[0]

    res = psi_xx - torch.sinh(psi)

    return res

# ── Função de perda ────────────────────────────────────────────────────────────

def loss_fn_pb(model, psi0, X_col, X_bc, Psi_bc, X_obs, Psi_obs,
               w_pde=1.0, w_bc=1.0, w_data=1.0):
    """
    Função de perda da PINN para o problema inverso de Poisson-Boltzmann.

    Args:
        model:    rede neural
        psi0:     parâmetro treinável (nn.Parameter)
        X_col:    pontos de colocação, shape (N_c, 1)
        X_bc:     pontos de contorno, shape (2, 1)
        Psi_bc:   valores de contorno, shape (2, 1)
        X_obs:    pontos de observação, shape (N_obs, 1)
        Psi_obs:  potencial ruidoso, shape (N_obs, 1)
        w_pde:    peso da perda física
        w_bc:     peso da perda de contorno
        w_data:   peso da perda de dados
    """
    # perda física
    residual = pde_residual_pb(model, X_col)
    loss_pde = torch.mean(residual ** 2)

    # perda de contorno
    Psi_bc_true = torch.cat([psi0.view(1, 1),
                             torch.zeros(1, 1, device=X_bc.device)], dim=0)
    Psi_bc_pred = model(X_bc)
    loss_bc     = torch.mean((Psi_bc_pred - Psi_bc_true) ** 2)

    # perda de dados
    Psi_obs_pred = model(X_obs)
    loss_data    = torch.mean((Psi_obs_pred - Psi_obs) ** 2)

    loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data

    return loss, loss_pde, loss_bc, loss_data

# ── Treinamento ────────────────────────────────────────────────────────────────

def train_pb(model, psi0, optimizer, X_col, X_bc, X_obs, Psi_obs,
             n_epochs, w_pde=1.0, w_bc=1.0, w_data=1.0):
    """
    Loop de treinamento da PINN para o problema inverso de Poisson-Boltzmann.

    Args:
        model:     rede neural
        psi0:      parâmetro treinável (nn.Parameter)
        optimizer: otimizador — deve incluir psi0 nos parâmetros
        X_col:     pontos de colocação, shape (N_c, 1)
        X_obs:     pontos de observação, shape (N_obs, 1)
        Psi_obs:   potencial ruidoso, shape (N_obs, 1)
        L:         comprimento do domínio
        device:    dispositivo de execução
        n_epochs:  número de épocas
        w_pde:     peso da perda física
        w_bc:      peso da perda de contorno
        w_data:    peso da perda de dados

    Retorna:
        history: dicionário com histórico de perdas e de psi0
    """
    history = {
        'loss':      [],
        'loss_pde':  [],
        'loss_bc':   [],
        'loss_data': [],
        'psi0':      [],
    }

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_pde, loss_bc, loss_data = loss_fn_pb(
            model, psi0, X_col, X_bc, None, X_obs, Psi_obs,
            w_pde, w_bc, w_data
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_data'].append(loss_data.item())
        history['psi0'].append(psi0.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e} | '
                  f'Loss BC: {loss_bc.item():.2e} | '
                  f'Loss data: {loss_data.item():.2e} | '
                  f'psi0: {psi0.item():.4f}')

    return history

# ── Avaliação ──────────────────────────────────────────────────────────────────

def evaluate_pb(model, psi0, psi0_true, L, device, n_grid=200):
    """
    Avalia o modelo treinado e retorna arrays prontos para plotagem.

    Args:
        model:     rede neural treinada
        psi0:      parâmetro recuperado (nn.Parameter)
        psi0_true: valor verdadeiro de psi0
        L:         comprimento do domínio
        device:    dispositivo de execução
        n_grid:    resolução da grade

    Retorna dicionário com:
        'x':           array (n_grid,)
        'psi_pred':    array (n_grid,) — solução predita pela PINN
        'psi_exact':   array (n_grid,) — solução exata com psi0 verdadeiro
        'psi_recovered'array (n_grid,) — solução exata com psi0 recuperado
        'psi0_pred':   float — valor recuperado
        'psi0_true':   float — valor verdadeiro
        'error_pct':   float — erro percentual em psi0
        'l2_error':    float — erro L2 relativo na solução
    """
    x = np.linspace(0, L, n_grid)

    X_grid = torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1)

    with torch.no_grad():
        psi_pred = model(X_grid).cpu().numpy().ravel()

    psi_exact     = analytical_solution_pb(x, psi0_true)
    psi0_pred_val = psi0.item()
    psi_recovered = analytical_solution_pb(x, psi0_pred_val)

    error_pct = np.abs(psi0_pred_val - psi0_true) / np.abs(psi0_true) * 100
    l2_error  = np.linalg.norm(psi_pred - psi_exact) / np.linalg.norm(psi_exact)

    return {
        'x':            x,
        'psi_pred':     psi_pred,
        'psi_exact':    psi_exact,
        'psi_recovered':psi_recovered,
        'psi0_pred':    psi0_pred_val,
        'psi0_true':    psi0_true,
        'error_pct':    error_pct,
        'l2_error':     l2_error,
    }