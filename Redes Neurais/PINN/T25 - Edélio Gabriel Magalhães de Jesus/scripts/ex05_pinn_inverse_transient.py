"""
Problema inverso transiente 2D — Equação de difusão.

Recuperar o coeficiente de difusão D a partir de medições esparsas
do campo de concentração c(x, y, t).

Referências:
    - Raissi et al. (2019), J. Comput. Phys., 378, 686-707.
    - Thakur et al. (2024), arXiv:2403.03970
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter

# ── Solução numérica ───────────────────────────────────────────────────────────

def numerical_solution_diffusion_2d(D, nx=50, ny=50, nt_out=25,
                                     sigma=0.1, x0=0.5, y0=0.5):
    """
    Solução numérica da equação de difusão 2D via diferenças finitas explícitas.

    dc/dt = D * (d²c/dx² + d²c/dy²)

    com CI gaussiana e CCs de Dirichlet homogêneas.

    Args:
        D:      coeficiente de difusão verdadeiro
        nx:     número de pontos em x
        ny:     número de pontos em y
        nt_out: número de instantes de saída
        sigma:  largura da gaussiana da CI
        x0:     centro da gaussiana em x
        y0:     centro da gaussiana em y

    Retorna:
        x:  array (nx,)
        y:  array (ny,)
        t:  array (nt_out,)
        C:  array (nx, ny, nt_out)
    """
    x  = np.linspace(0, 1, nx)
    y  = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # condição de estabilidade CFL
    dt = 0.4 * min(dx, dy)**2 / (4 * D)

    t_end  = 1.0
    n_steps = int(t_end / dt) + 1
    dt     = t_end / n_steps

    # instantes de saída
    t_out     = np.linspace(0, t_end, nt_out)
    out_steps = (t_out / dt).astype(int)

    # grade 2D
    X, Y = np.meshgrid(x, y, indexing='ij')

    # condição inicial gaussiana
    C = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    C[0, :]  = 0
    C[-1, :] = 0
    C[:, 0]  = 0
    C[:, -1] = 0

    C_out = np.zeros((nx, ny, nt_out))
    out_idx = 0

    for step in range(n_steps + 1):
        if out_idx < nt_out and step == out_steps[out_idx]:
            C_out[:, :, out_idx] = C.copy()
            out_idx += 1

        # diferenças finitas explícitas
        d2x = (C[2:, 1:-1] - 2*C[1:-1, 1:-1] + C[:-2, 1:-1]) / dx**2
        d2y = (C[1:-1, 2:] - 2*C[1:-1, 1:-1] + C[1:-1, :-2]) / dy**2

        C[1:-1, 1:-1] += D * dt * (d2x + d2y)

        # CCs
        C[0, :]  = 0
        C[-1, :] = 0
        C[:, 0]  = 0
        C[:, -1] = 0

    return x, y, t_out, C_out

# ── Dados sintéticos ───────────────────────────────────────────────────────────

def generate_synthetic_data_diffusion(
    D_true,
    N_obs,
    noise_amp,
    device,
    seed=42,
    nx=50,
    ny=50,
    nt_out=25,
    sigma=0.1,
    x0=0.5,
    y0=0.5
):
    """
    Gera dados sintéticos ruidosos a partir da solução numérica
    da equação de difusão 2D.

    Args:
        D_true:    coeficiente de difusão verdadeiro
        N_obs:     número de observações
        noise_amp: amplitude do ruído gaussiano
        device:    dispositivo
        seed:      semente aleatória

        Parâmetros do solver:
        nx, ny:    discretização espacial
        nt_out:    número de saídas temporais
        sigma:     largura da gaussiana inicial
        x0, y0:    centro da gaussiana

    Retorna:
        X_obs: tensor (N_obs, 3) -> (x, y, t)
        C_obs: tensor (N_obs, 1) -> concentração ruidosa
    """

    np.random.seed(seed)

    # --------------------------------------------------------
    # Gera solução numérica internamente
    # --------------------------------------------------------

    x, y, t, C_ref = numerical_solution_diffusion_2d(
        D=D_true,
        nx=nx,
        ny=ny,
        nt_out=nt_out,
        sigma=sigma,
        x0=x0,
        y0=y0
    )

    nx_ref, ny_ref, nt_ref = C_ref.shape

    # --------------------------------------------------------
    # Amostragem aleatória
    # --------------------------------------------------------

    ix = np.random.randint(1, nx_ref - 1, N_obs)
    iy = np.random.randint(1, ny_ref - 1, N_obs)
    it = np.random.randint(0, nt_ref,     N_obs)

    x_obs = x[ix]
    y_obs = y[iy]
    t_obs = t[it]

    c_exact = C_ref[ix, iy, it]

    # --------------------------------------------------------
    # Adiciona ruído
    # --------------------------------------------------------

    c_noisy = c_exact + np.random.normal(0, noise_amp, N_obs)

    # --------------------------------------------------------
    # Tensores
    # --------------------------------------------------------

    X_obs = torch.tensor(
        np.stack([x_obs, y_obs, t_obs], axis=1),
        dtype=torch.float32,
        device=device
    )

    C_obs = torch.tensor(
        c_noisy,
        dtype=torch.float32,
        device=device
    ).view(-1, 1)

    return X_obs, C_obs

# ── Resíduo da EDP ─────────────────────────────────────────────────────────────

def pde_residual_diffusion_2d(model, D, X):
    """
    Calcula o resíduo da equação de difusão 2D.

    dc/dt - D*(d²c/dx² + d²c/dy²) = 0

    Args:
        model: rede neural — entrada (x, y, t), saída c
        D:     parâmetro treinável (nn.Parameter)
        X:     tensor (N_c, 3) com requires_grad=True
               X[:,0]=x, X[:,1]=y, X[:,2]=t

    Retorna:
        res: tensor (N_c, 1)
    """
    c = model(X)

    grads = torch.autograd.grad(
        outputs=c,
        inputs=X,
        grad_outputs=torch.ones_like(c),
        create_graph=True
    )[0]

    c_x = grads[:, 0].unsqueeze(1)
    c_y = grads[:, 1].unsqueeze(1)
    c_t = grads[:, 2].unsqueeze(1)

    c_xx = torch.autograd.grad(
        outputs=c_x,
        inputs=X,
        grad_outputs=torch.ones_like(c_x),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    c_yy = torch.autograd.grad(
        outputs=c_y,
        inputs=X,
        grad_outputs=torch.ones_like(c_y),
        create_graph=True
    )[0][:, 1].unsqueeze(1)

    res = c_t - D * (c_xx + c_yy)

    return res

# ── Função de perda ────────────────────────────────────────────────────────────

def loss_fn_diffusion(model, D, X_col, X_ic, C_ic, X_bc, C_bc,
                      X_obs, C_obs, w_pde=1.0, w_ic=1.0,
                      w_bc=1.0, w_data=1.0):
    """
    Função de perda da PINN para o problema inverso de difusão 2D.

    Args:
        model:  rede neural
        D:      parâmetro treinável (nn.Parameter)
        X_col:  pontos de colocação, shape (N_c, 3)
        X_ic:   pontos da CI, shape (N_ic, 3)
        C_ic:   valores da CI, shape (N_ic, 1)
        X_bc:   pontos das CCs, shape (4*N_bc, 3)
        C_bc:   valores das CCs, shape (4*N_bc, 1)
        X_obs:  pontos de observação, shape (N_obs, 3)
        C_obs:  concentração ruidosa, shape (N_obs, 1)
        w_pde:  peso da perda física
        w_ic:   peso da perda de CI
        w_bc:   peso da perda de CC
        w_data: peso da perda de dados

    Retorna:
        loss, loss_pde, loss_ic, loss_bc, loss_data
    """
    # perda física
    residual = pde_residual_diffusion_2d(model, D, X_col)
    loss_pde = torch.mean(residual ** 2)

    # perda CI
    C_ic_pred = model(X_ic)
    loss_ic   = torch.mean((C_ic_pred - C_ic) ** 2)

    # perda CC
    C_bc_pred = model(X_bc)
    loss_bc   = torch.mean((C_bc_pred - C_bc) ** 2)

    # perda dados
    C_obs_pred = model(X_obs)
    loss_data  = torch.mean((C_obs_pred - C_obs) ** 2)

    loss = w_pde * loss_pde + w_ic * loss_ic + \
           w_bc * loss_bc + w_data * loss_data

    return loss, loss_pde, loss_ic, loss_bc, loss_data

# ── Treinamento ────────────────────────────────────────────────────────────────

def train_diffusion(model, D, optimizer, X_col, X_ic, C_ic, X_bc, C_bc,
                    X_obs, C_obs, n_epochs, w_pde=1.0, w_ic=1.0,
                    w_bc=1.0, w_data=1.0):
    """
    Loop de treinamento da PINN para o problema inverso de difusão 2D.

    Args:
        model:     rede neural
        D:         parâmetro treinável (nn.Parameter)
        optimizer: otimizador — deve incluir D nos parâmetros
        X_col:     pontos de colocação, shape (N_c, 3)
        X_ic:      pontos da CI, shape (N_ic, 3)
        C_ic:      valores da CI, shape (N_ic, 1)
        X_bc:      pontos das CCs, shape (4*N_bc, 3)
        C_bc:      valores das CCs, shape (4*N_bc, 1)
        X_obs:     pontos de observação, shape (N_obs, 3)
        C_obs:     concentração ruidosa, shape (N_obs, 1)
        n_epochs:  número de épocas
        w_pde:     peso da perda física
        w_ic:      peso da perda de CI
        w_bc:      peso da perda de CC
        w_data:    peso da perda de dados

    Retorna:
        history: dicionário com histórico de perdas e de D
    """
    history = {
        'loss':      [],
        'loss_pde':  [],
        'loss_ic':   [],
        'loss_bc':   [],
        'loss_data': [],
        'D':         [],
    }

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_pde, loss_ic, loss_bc, loss_data = loss_fn_diffusion(
            model, D, X_col, X_ic, C_ic, X_bc, C_bc,
            X_obs, C_obs, w_pde, w_ic, w_bc, w_data
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_pde'].append(loss_pde.item())
        history['loss_ic'].append(loss_ic.item())
        history['loss_bc'].append(loss_bc.item())
        history['loss_data'].append(loss_data.item())
        history['D'].append(D.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e} | '
                  f'Loss IC: {loss_ic.item():.2e} | '
                  f'Loss BC: {loss_bc.item():.2e} | '
                  f'Loss data: {loss_data.item():.2e} | '
                  f'D: {D.item():.4f}')

    return history

# ── Avaliação ──────────────────────────────────────────────────────────────────

def evaluate_diffusion(model, D, D_true, device,
                       snapshots=None,
                       nx=50, ny=50, nt_out=25,
                       sigma=0.1, x0=0.5, y0=0.5):
    """
    Avalia o modelo treinado comparando a predição da PINN com a
    solução numérica de referência da equação de difusão 2D.

    A solução numérica é gerada internamente via diferenças finitas
    explícitas.

    Args:
        model:     rede neural treinada
        D:         parâmetro recuperado (nn.Parameter)
        D_true:    valor verdadeiro do coeficiente de difusão
        device:    dispositivo de execução

        snapshots:
            lista de índices temporais usados para visualização.
            Default: [0, nt_out//2, nt_out-1]

        Parâmetros do solver numérico:
            nx:     número de pontos em x
            ny:     número de pontos em y
            nt_out: número de snapshots temporais
            sigma:  largura da gaussiana inicial
            x0:     centro da gaussiana em x
            y0:     centro da gaussiana em y

    Retorna:
        dict contendo:

        'x':
            array (nx,) com coordenadas espaciais em x

        'y':
            array (ny,) com coordenadas espaciais em y

        't':
            array (nt_out,) com instantes de tempo

        'C_pred_snaps':
            lista de arrays (nx, ny) contendo as predições da PINN

        'C_ref_snaps':
            lista de arrays (nx, ny) contendo a solução numérica
            de referência

        'snap_times':
            lista dos tempos associados aos snapshots

        'D_pred':
            valor recuperado pela PINN

        'D_true':
            valor verdadeiro do coeficiente de difusão

        'error_pct':
            erro relativo percentual de D

        'l2_error':
            erro relativo L2 entre solução prevista e referência
    """

    # ------------------------------------------------------------------
    # Gera solução numérica de referência internamente
    # ------------------------------------------------------------------

    x, y, t, C_ref = numerical_solution_diffusion_2d(
        D=D_true,
        nx=nx,
        ny=ny,
        nt_out=nt_out,
        sigma=sigma,
        x0=x0,
        y0=y0
    )

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    if snapshots is None:
        snapshots = [0, len(t)//2, len(t)-1]

    nx_ref, ny_ref = len(x), len(y)

    X_grid, Y_grid = np.meshgrid(x, y, indexing='ij')

    C_pred_snaps = []
    C_ref_snaps  = []
    snap_times   = []

    for idx in snapshots:

        t_snap = t[idx]

        T_grid = np.full_like(X_grid, t_snap)

        X_flat = torch.tensor(
            np.stack([
                X_grid.ravel(),
                Y_grid.ravel(),
                T_grid.ravel()
            ], axis=1),
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            C_pred = model(X_flat).cpu().numpy().reshape(nx_ref, ny_ref)

        C_pred_snaps.append(C_pred)
        C_ref_snaps.append(C_ref[:, :, idx])
        snap_times.append(t_snap)

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------

    D_pred = D.item()

    error_pct = np.abs(D_pred - D_true) / D_true * 100

    l2_error = np.linalg.norm(
        np.array(C_pred_snaps) - np.array(C_ref_snaps)
    ) / np.linalg.norm(np.array(C_ref_snaps))

    # ------------------------------------------------------------------
    # Retorno
    # ------------------------------------------------------------------

    return {
        'x':            x,
        'y':            y,
        't':            t,
        'C_pred_snaps': C_pred_snaps,
        'C_ref_snaps':  C_ref_snaps,
        'snap_times':   snap_times,
        'D_pred':       D_pred,
        'D_true':       D_true,
        'error_pct':    error_pct,
        'l2_error':     l2_error,
    }