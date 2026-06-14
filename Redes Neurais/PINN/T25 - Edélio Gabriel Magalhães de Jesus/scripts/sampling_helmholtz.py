"""
Notebook de amostragem — Equação de Helmholtz 2D.

Comparação entre três estratégias de amostragem dos pontos de colocação:
    - Uniforme (grade regular)
    - Aleatória (torch.rand)
    - Latin Hypercube Sampling (LHS)

Problema: arcada magnética solar — Baty (2024)
Equação: nabla^2 u + c^2 u = 0
Domínio: x in [-L/2, L/2], z in [0, L]
Parâmetros: L=3, c=0.8, (a1, a2, a3) = (1, 0, 1)

Referências:
    - Baty, H. (2024). arXiv:2403.00599.
"""

import torch
import numpy as np
from scipy.stats import qmc

# ── Parâmetros físicos ─────────────────────────────────────────────────────────

L  = 3.0
C  = 0.8
A1 = 1.0
A2 = 0.0
A3 = 1.0

NU1 = np.sqrt((1 * np.pi / L)**2 - C**2)
NU2 = np.sqrt((2 * np.pi / L)**2 - C**2)
NU3 = np.sqrt((3 * np.pi / L)**2 - C**2)

LB_X = -L / 2
UB_X =  L / 2
LB_Z =  0.0
UB_Z =  L

# ── Solução analítica ──────────────────────────────────────────────────────────

def analytical_solution_helmholtz(X):
    """
    Solução analítica de Baty (2024) para arcada magnética solar.

    u(x,z) = sum_{k=1}^{3} a_k * exp(-nu_k * z) * cos(k*pi*x/L)

    com (a1, a2, a3) = (1, 0, 1) e L=3, c=0.8.

    Args:
        X: tensor (N, 2) — X[:,0]=x, X[:,1]=z

    Retorna:
        u: tensor (N, 1)
    """
    x = X[:, 0].unsqueeze(1)
    z = X[:, 1].unsqueeze(1)

    u = (A1 * torch.exp(-NU1 * z) * torch.cos(1 * torch.pi * x / L) +
         A2 * torch.exp(-NU2 * z) * torch.cos(2 * torch.pi * x / L) +
         A3 * torch.exp(-NU3 * z) * torch.cos(3 * torch.pi * x / L))

    return u

# ── Estratégias de amostragem ──────────────────────────────────────────────────

def sample_uniform(N, device, seed=None):
    """
    Amostragem uniforme — grade regular no domínio.

    N pontos distribuídos em grade sqrt(N) x sqrt(N).
    Se N não for quadrado perfeito, usa floor(sqrt(N))^2 pontos.

    Args:
        N:      número aproximado de pontos
        device: dispositivo de execução

    Retorna tensor (N, 2) com requires_grad=True.
    """
    n = int(np.sqrt(N))
    x = np.linspace(LB_X, UB_X, n)
    z = np.linspace(LB_Z, UB_Z, n)
    X, Z = np.meshgrid(x, z)
    pts = np.stack([X.ravel(), Z.ravel()], axis=1)

    X_col = torch.tensor(pts, dtype=torch.float32, device=device)
    X_col.requires_grad_(True)
    return X_col


def sample_random(N, device, seed=367):
    """
    Amostragem aleatória — pontos sorteados uniformemente no domínio.

    Args:
        N:      número de pontos
        device: dispositivo de execução

    Retorna tensor (N, 2) com requires_grad=True.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(LB_X, UB_X, N)
    z = rng.uniform(LB_Z, UB_Z, N)

    pts = np.stack([x, z], axis=1)

    X_col = torch.from_numpy(pts).float().to(device)

    X_col.requires_grad_(True)

    return X_col


def sample_lhs(N, device, seed=367):
    """
    Latin Hypercube Sampling — amostragem quasi-aleatória com
    melhor cobertura do domínio que a amostragem aleatória pura.

    Garante que cada subdivisão do domínio em cada dimensão
    contenha exatamente um ponto.

    Referência: McKay et al. (1979). Technometrics, 21(2), 239-245.

    Args:
        N:      número de pontos
        device: dispositivo de execução
        seed:   semente para reprodutibilidade

    Retorna tensor (N, 2) com requires_grad=True.
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    pts = sampler.random(n=N)

    # escala para o domínio físico
    lb = np.array([LB_X, LB_Z])
    ub = np.array([UB_X, UB_Z])
    pts = qmc.scale(pts, lb, ub)

    X_col = torch.tensor(pts, dtype=torch.float32, device=device)
    X_col.requires_grad_(True)
    return X_col

# ── Condições de contorno ──────────────────────────────────────────────────────

def sample_boundary_helmholtz(N_b, device):
    """
    Amostra pontos de contorno com valores dados pela solução analítica.

    As quatro faces do domínio são amostradas aleatoriamente.
    Os valores de CC são calculados diretamente da solução analítica.

    Args:
        N_b:    número de pontos por face
        device: dispositivo de execução

    Retorna:
        X_bc: tensor (4*N_b, 2) — coordenadas dos pontos de contorno
        U_bc: tensor (4*N_b, 1) — valores analíticos nas fronteiras
    """
    # face inferior: z=0, x livre
    x_bot = np.random.uniform(LB_X, UB_X, N_b)
    z_bot = np.full(N_b, LB_Z)

    # face superior: z=L, x livre
    x_top = np.random.uniform(LB_X, UB_X, N_b)
    z_top = np.full(N_b, UB_Z)

    # face esquerda: x=-L/2, z livre
    x_left = np.full(N_b, LB_X)
    z_left = np.random.uniform(LB_Z, UB_Z, N_b)

    # face direita: x=L/2, z livre
    x_right = np.full(N_b, UB_X)
    z_right = np.random.uniform(LB_Z, UB_Z, N_b)

    x_all = np.concatenate([x_bot, x_top, x_left, x_right])
    z_all = np.concatenate([z_bot, z_top, z_left, z_right])

    X_bc = torch.tensor(
        np.stack([x_all, z_all], axis=1),
        dtype=torch.float32, device=device
    )

    U_bc = analytical_solution_helmholtz(X_bc)

    return X_bc, U_bc.detach()

# ── Resíduo da EDP ─────────────────────────────────────────────────────────────

def pde_residual_helmholtz(model, X):
    """
    Calcula o resíduo da equação de Helmholtz.

    nabla^2 u + c^2 u = 0
    u_xx + u_zz + c^2 * u = 0

    Args:
        model: rede neural
        X:     tensor (N_c, 2) com requires_grad=True

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
    u_z = grads[:, 1].unsqueeze(1)

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    u_zz = torch.autograd.grad(
        outputs=u_z,
        inputs=X,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True
    )[0][:, 1].unsqueeze(1)

    res = u_xx + u_zz + C**2 * u

    return res

# ── Função de perda ────────────────────────────────────────────────────────────

def loss_fn_helmholtz(model, X_col, X_bc, U_bc, w_pde=1.0, w_bc=1.0):
    """
    Função de perda da PINN para Helmholtz.

    Args:
        model:  rede neural
        X_col:  pontos de colocação, shape (N_c, 2)
        X_bc:   pontos de contorno, shape (4*N_b, 2)
        U_bc:   valores analíticos nas fronteiras, shape (4*N_b, 1)
        w_pde:  peso da perda física
        w_bc:   peso da perda de contorno

    Retorna:
        loss, loss_pde, loss_bc
    """
    residual = pde_residual_helmholtz(model, X_col)
    loss_pde = torch.mean(residual ** 2)

    U_pred   = model(X_bc)
    loss_bc  = torch.mean((U_pred - U_bc) ** 2)

    loss = w_pde * loss_pde + w_bc * loss_bc

    return loss, loss_pde, loss_bc

# ── Treinamento ────────────────────────────────────────────────────────────────

def train_helmholtz(model, optimizer, X_col, X_bc, U_bc,
                    n_epochs, w_pde=1.0, w_bc=1.0):
    """
    Loop de treinamento da PINN para Helmholtz.

    Args:
        model:     rede neural
        optimizer: otimizador
        X_col:     pontos de colocação, shape (N_c, 2)
        X_bc:      pontos de contorno, shape (4*N_b, 2)
        U_bc:      valores analíticos nas fronteiras, shape (4*N_b, 1)
        n_epochs:  número de épocas
        w_pde:     peso da perda física
        w_bc:      peso da perda de contorno

    Retorna:
        history: dicionário com histórico de perdas
    """
    history = {
        'loss':     [],
        'loss_pde': [],
        'loss_bc':  [],
    }

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_pde, loss_bc = loss_fn_helmholtz(
            model, X_col, X_bc, U_bc, w_pde, w_bc
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_bc'].append(loss_bc.item())

        if epoch % 1000 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e} | '
                  f'Loss BC: {loss_bc.item():.2e}')

    return history

# ── Avaliação ──────────────────────────────────────────────────────────────────

def evaluate_helmholtz(model, device, n_grid=100):
    """
    Avalia o modelo treinado e retorna arrays prontos para plotagem.

    Args:
        model:   rede neural treinada
        device:  dispositivo de execução
        n_grid:  resolução da grade de avaliação

    Retorna dicionário com:
        'x':       array (n_grid,)
        'z':       array (n_grid,)
        'U_pred':  array (n_grid, n_grid)
        'U_ref':   array (n_grid, n_grid)
        'l2_error': float
    """
    x = np.linspace(LB_X, UB_X, n_grid)
    z = np.linspace(LB_Z, UB_Z, n_grid)
    X, Z = np.meshgrid(x, z, indexing='ij')

    X_flat = torch.tensor(
        np.stack([X.ravel(), Z.ravel()], axis=1),
        dtype=torch.float32, device=device
    )

    with torch.no_grad():
        U_pred = model(X_flat).cpu().numpy().reshape(n_grid, n_grid)

    U_ref = analytical_solution_helmholtz(X_flat).cpu().numpy().reshape(n_grid, n_grid)

    l2_error = np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref)

    return {
        'x':        x,
        'z':        z,
        'U_pred':   U_pred,
        'U_ref':    U_ref,
        'l2_error': l2_error,
    }
#