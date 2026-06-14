"""
arquitectures.py

Funções para o notebook 06_architectures.ipynb.
Comparação entre MLP vanilla e ResNet-PINN aplicadas
ao problema de Helmholtz 2D — arcada magnética solar,
com solução de maior complexidade (8 modos espaciais).

A motivação para ResNet é a profundidade necessária para capturar
as oscilações espaciais da solução — onde o vanishing gradient
da MLP vanilla dificulta o treinamento.

Referências:
    - Zhang et al. (2021), Water, 13(4), 423.
    - Baty, H. (2024). arXiv:2403.00599
"""

import torch
import torch.nn as nn
import numpy as np

# ── Parâmetros físicos ─────────────────────────────────────────────────────────

L     = 3.0
C     = 0.8
MODES = [1, 2, 3, 4, 5, 6, 7, 8]
AMPLS = [1.0, 0.0, 1.0, 0.8, 0.0, 0.6, 0.0, 0.4]

assert all((k * np.pi / L)**2 - C**2 > 0 for k in MODES), \
    "Algum nu_k é imaginário — reduza C ou aumente L"

NUS  = [np.sqrt((k * np.pi / L)**2 - C**2) for k in MODES]

LB_X = -L / 2
UB_X =  L / 2
LB_Z =  0.0
UB_Z =  L

# ── Solução analítica ──────────────────────────────────────────────────────────

def analytical_solution_helmholtz(X):
    """
    Solução analítica de Baty (2024) para arcada magnética solar,
    estendida para 8 modos espaciais.

    u(x,z) = sum_{k in MODES} a_k * exp(-nu_k * z) * cos(k*pi*x/L)

    com MODES = [1..8], L=3, c=0.8.

    Args:
        X: tensor (N, 2) — X[:,0]=x, X[:,1]=z

    Retorna:
        u: tensor (N, 1)
    """
    x = X[:, 0].unsqueeze(1)
    z = X[:, 1].unsqueeze(1)

    u = sum(
        a * torch.exp(-nu * z) * torch.cos(k * torch.pi * x / L)
        for k, a, nu in zip(MODES, AMPLS, NUS)
    )

    return u

# ── Bloco residual ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Bloco residual com skip connection.

    Realiza:
        out = activation(linear2(activation(linear1(x))) + x)

    A skip connection permite que o gradiente flua diretamente
    pelas camadas mais profundas, evitando o problema de
    vanishing gradient em redes profundas.

    Args:
        n_hidden:   número de neurônios
        activation: classe da função de ativação
    """
    def __init__(self, n_hidden, activation):
        super().__init__()
        self.linear1    = nn.Linear(n_hidden, n_hidden)
        self.linear2    = nn.Linear(n_hidden, n_hidden)
        self.activation = activation()

    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)

# ── ResNet-PINN ────────────────────────────────────────────────────────────────

class ResNetPINN(nn.Module):
    """
    Arquitetura residual para PINNs.
    Mesma assinatura que PINN para facilitar comparação direta.

    n_layers define o número de blocos residuais.
    Cada bloco contém 2 camadas lineares + skip connection,
    então a profundidade efetiva é 2 * n_layers + 1.

    Args:
        n_inputs:   dimensão da entrada
        n_outputs:  dimensão da saída
        n_hidden:   número de neurônios por camada
        n_layers:   número de blocos residuais
        activation: classe da função de ativação
    """
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers, activation):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            activation()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_hidden, activation) for _ in range(n_layers)]
        )

        self.output_layer = nn.Linear(n_hidden, n_outputs)

    def forward(self, X):
        X = self.input_layer(X)
        X = self.res_blocks(X)
        return self.output_layer(X)

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
    x_bot   = np.random.uniform(LB_X, UB_X, N_b)
    z_bot   = np.full(N_b, LB_Z)

    x_top   = np.random.uniform(LB_X, UB_X, N_b)
    z_top   = np.full(N_b, UB_Z)

    x_left  = np.full(N_b, LB_X)
    z_left  = np.random.uniform(LB_Z, UB_Z, N_b)

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
    Calcula o resíduo da equação de Helmholtz via diferenciação automática.

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

    U_pred  = model(X_bc)
    loss_bc = torch.mean((U_pred - U_bc) ** 2)

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
        'x':        array (n_grid,)
        'z':        array (n_grid,)
        'U_pred':   array (n_grid, n_grid)
        'U_ref':    array (n_grid, n_grid)
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