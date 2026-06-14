"""
plot_utils.py — funções de visualização para notebooks de PINNs.

Todas as funções recebem arrays numpy prontos — nenhuma dependência de torch,
modelos ou funções analíticas. A conversão de tensores para arrays é
responsabilidade do script particular de cada exemplo.

Compatível com problemas estacionários (ex: Laplace) e transientes (ex: Burgers).
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Dimensões e fonte ──────────────────────────────────────────────────────────

FONT_SIZE   = 16
FONT_FAMILY = "Arial"
SQ_SIZE     = 500

AXIS_COMMON = dict(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="inside",
    ticklen=5,
    tickwidth=1.5,
    tickcolor="black",
)

LAYOUT_BASE = dict(
    template="simple_white",
    font=dict(family=FONT_FAMILY, size=FONT_SIZE),
    margin=dict(l=80, r=60, t=70, b=70),
)

COLORSCALE = "RdBu_r"
COLORS     = ["#2C3E50", "#E74C3C", "#2980B9", "#27AE60"]

# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _square_axis(title, log=False):
    """Retorna dict de eixo com borda e fonte ABNT."""
    ax = dict(**AXIS_COMMON, title=dict(text=title, font=dict(size=FONT_SIZE)))
    if log:
        ax["type"] = "log"
    return ax


# ── 1. Curvas de aprendizado ───────────────────────────────────────────────────
def plot_loss(history):
    """
    Plota as curvas de aprendizado disponíveis no history.
    Aceita qualquer combinação de: 'loss', 'loss_data', 'loss_pde', 'loss_ic'
 
    Args:
        history: dicionário com qualquer subconjunto de chaves:
                 'loss', 'loss_data', 'loss_pde', 'loss_ic'
    """
    labels = {
        'loss':      ('Total',    '#2C3E50', 'solid'),
        'loss_pde':  ('PDE',      '#2980B9', 'dot'),
        'loss_bc':   ('C. Cont.', '#E74C3C', 'dash'),      # condição de contorno
        'loss_ic':   ('C. Ini.',  '#27AE60', 'dashdot'),   # condição inicial (transiente)
        'loss_data': ('Dados',    '#8E44AD', 'longdash'),  # dados observados (inverso)
    }
 
    epochs = list(range(len(next(iter(history.values())))))
 
    fig = go.Figure()
 
    for key, (name, color, dash) in labels.items():
        if key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key],
                mode='lines', name=name,
                line=dict(color=color, width=2, dash=dash)
            ))
 
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Curvas de aprendizado", font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis("Época"),
        yaxis=_square_axis("Perda", log=True),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )
 
    fig.update_yaxes(
        type="log",
        dtick=1,
        exponentformat="power",
        showexponent="all",
    )
 
    fig.show()

# ── 2. Mapas de calor ─────────────────────────────────────────────────────────

def plot_heatmaps(U_pred, U_ref, x, y, title='Solução', xlabel='x', ylabel='y',
                  square_aspect=None):
    """
    Plota mapas de calor: solução predita, referência e erro absoluto.

    Args:
        U_pred:        array (n_grid, n_grid) — solução predita
        U_ref:         array (n_grid, n_grid) — solução de referência
        x:             array (n_grid,)        — coordenadas x (1ª dimensão do grid)
        y:             array (n_grid,)        — coordenadas y ou t (2ª dimensão)
        title:         título do plot
        xlabel:        título do eixo x
        ylabel:        título do eixo y (use 't' para problemas transientes)
        square_aspect: força aspecto quadrado nos heatmaps (default: True se
                       domínio x e y têm o mesmo tamanho, False caso contrário)
    """
    E_abs = np.abs(U_pred - U_ref)

    vmin = min(U_pred.min(), U_ref.min())
    vmax = max(U_pred.max(), U_ref.max())

    # Infere aspecto automaticamente se não fornecido
    if square_aspect is None:
        x_range = x[-1] - x[0]
        y_range = y[-1] - y[0]
        square_aspect = np.isclose(x_range, y_range, rtol=0.05)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Predição", "Referência", "Erro absoluto"],
        horizontal_spacing=0.15,
        column_widths=[0.33, 0.33, 0.33],
    )

    colorbar_configs = [
        None,
        dict(x=0.655, y=0.5, len=0.9, thickness=12),
        dict(x=1.00,  y=0.5, len=0.9, thickness=12),
    ]

    for col, (Z, zmin, zmax, showscale, cbar) in enumerate(zip(
        [U_pred, U_ref,  E_abs],
        [vmin,   vmin,   0],
        [vmax,   vmax,   E_abs.max()],
        [False,  True,   True],
        colorbar_configs
    ), start=1):

        fig.add_trace(go.Heatmap(
            z=Z, x=x, y=y,
            colorscale=COLORSCALE,
            zmin=zmin, zmax=zmax,
            showscale=showscale,
            colorbar=cbar,
        ), row=1, col=col)

    x_range_plot = [x[0], x[-1]]
    y_range_plot = [y[0], y[-1]]

    for col in [1, 2, 3]:
        x_axis_kw = dict(
            **AXIS_COMMON,
            title_text=xlabel,
            title_font=dict(size=FONT_SIZE),
            range=x_range_plot,
            constrain="domain",
            row=1, col=col,
        )
        y_axis_kw = dict(
            **AXIS_COMMON,
            title_text=ylabel if col == 1 else "",
            title_font=dict(size=FONT_SIZE),
            range=y_range_plot,
            constrain="domain",
            row=1, col=col,
        )
        # Aspecto quadrado apenas quando domínios são compatíveis
        if square_aspect:
            y_axis_kw["scaleanchor"] = f"x{'' if col == 1 else col}"
            y_axis_kw["scaleratio"]  = 1

        fig.update_xaxes(**x_axis_kw)
        fig.update_yaxes(**y_axis_kw)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=450,
        width=1000,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 3. Perfis 1D ──────────────────────────────────────────────────────────────

def plot_profiles(U_pred_slices, U_ref_slices, x, slices=None, title='Perfis 1D',
                  xlabel='x', ylabel='u(x)', slice_label='t', shared_yaxes=False):
    """
    Plota perfis 1D comparando predição vs referência.

    Args:
        U_pred_slices: lista de arrays (n,) — predição em cada corte
        U_ref_slices:  lista de arrays (n,) — referência em cada corte
        x:             array (n,)           — coordenadas x
        slices:        lista de valores do corte (ex: [0.25, 0.5, 0.75])
        title:         título do plot
        xlabel:        título do eixo x
        ylabel:        título do eixo y
        slice_label:   rótulo do parâmetro de corte ('t' ou 'y')
        shared_yaxes:  compartilha eixo y entre subplots (default: False)
                       Use True apenas para campos suaves com escalas similares.
                       Para Burgers ou choques, mantenha False.
    """
    if slices is None:
        slices = [0.25, 0.5, 0.75]

    n = len(slices)

    rows, cols = (2, 2) if n == 4 else (1, n)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{slice_label} = {val:.2f}" for val in slices],
        shared_yaxes=shared_yaxes,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    for idx, (y_val, U_pred, U_ref) in enumerate(zip(slices, U_pred_slices, U_ref_slices)):
        row = idx // cols + 1
        col = idx %  cols + 1
        color = COLORS[idx % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=x, y=U_pred,
            mode="lines",
            name="Predição",
            legendgroup="pred",
            line=dict(color=color, width=2.5),
            showlegend=(idx == 0),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=U_ref,
            mode="lines",
            name="Referência",
            legendgroup="ref",
            line=dict(color=color, width=2.5, dash="dash"),
            showlegend=(idx == 0),
        ), row=row, col=col)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                **AXIS_COMMON,
                title_text=xlabel,
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )
            fig.update_yaxes(
                **AXIS_COMMON,
                title_text=ylabel if c == 1 else "",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )

    fig_w = SQ_SIZE + 80 if n == 4 else cols * (SQ_SIZE // 2) + 80
    fig_h = SQ_SIZE + 80 if n == 4 else SQ_SIZE // 2 + 120

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        width=fig_w,
        height=fig_h,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 4. Distribuição de pontos ─────────────────────────────────────────────────

def plot_points_stationary(
    X_col, X_bc=None,
    title='Distribuição dos pontos (estacionário)',
    xlabel='x', ylabel='y',
    square_aspect=None
):
    def _to_np(arr):
        if arr is None:
            return None
        return arr.detach().cpu().numpy()

    X_col = _to_np(X_col)
    X_bc  = _to_np(X_bc)

    # Range automático
    all_pts = [X_col]
    if X_bc is not None:
        all_pts.append(X_bc)
    all_pts = np.vstack(all_pts)

    pad = 0.05
    x_range = [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad]
    y_range = [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad]

    # Aspecto automático
    if square_aspect is None:
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        square_aspect = np.isclose(x_size, y_size, rtol=0.05)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col[:, 0], y=X_col[:, 1],
        mode="markers",
        name="Colocação",
        marker=dict(size=5, opacity=0.6),
    ))

    if X_bc is not None:
        fig.add_trace(go.Scatter(
            x=X_bc[:, 0], y=X_bc[:, 1],
            mode="markers",
            name="Contorno",
            marker=dict(size=7, symbol="square"),
        ))

    yaxis_kw = dict(
        **AXIS_COMMON,
        title=dict(text=ylabel, font=dict(size=FONT_SIZE)),
        range=y_range,
    )

    if square_aspect:
        yaxis_kw["scaleanchor"] = "x"
        yaxis_kw["scaleratio"] = 1

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text=xlabel, font=dict(size=FONT_SIZE)),
            range=x_range,
        ),
        yaxis=yaxis_kw,
        height=450,
        width=700,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()

def plot_points_transient(
    X_col, X_bc=None, X_ic=None,
    title='Distribuição dos pontos (transiente)',
    xlabel='t', ylabel='x',
    square_aspect=None
):
    def _to_np(arr):
        if arr is None:
            return None
        return arr.detach().cpu().numpy()

    X_col = _to_np(X_col)
    X_bc  = _to_np(X_bc)
    X_ic  = _to_np(X_ic)

    # Range automático
    all_pts = [X_col]
    if X_bc is not None:
        all_pts.append(X_bc)
    if X_ic is not None:
        all_pts.append(X_ic)
    all_pts = np.vstack(all_pts)

    pad = 0.05
    x_range = [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad]
    y_range = [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad]

    # Aspecto automático (igual ao heatmap)
    if square_aspect is None:
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        square_aspect = np.isclose(x_size, y_size, rtol=0.05)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col[:, 1], y=X_col[:, 0],
        mode="markers",
        name="Colocação",
        marker=dict(size=5, opacity=0.6),
    ))

    if X_bc is not None:
        fig.add_trace(go.Scatter(
            x=X_bc[:, 1], y=X_bc[:, 0],
            mode="markers",
            name="Contorno",
            marker=dict(size=7, symbol="square"),
        ))

    if X_ic is not None:
        fig.add_trace(go.Scatter(
            x=X_ic[:, 1], y=X_ic[:, 0],
            mode="markers",
            name="Cond. inicial",
            marker=dict(size=7, symbol="diamond"),
        ))

    yaxis_kw = dict(
        **AXIS_COMMON,
        title=dict(text=ylabel, font=dict(size=FONT_SIZE)),
        range=y_range,
    )

    if square_aspect:
        yaxis_kw["scaleanchor"] = "x"
        yaxis_kw["scaleratio"] = 1

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text=xlabel, font=dict(size=FONT_SIZE)),
            range=x_range,
        ),
        yaxis=yaxis_kw,
        height=450,
        width=700,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()

def plot_psi0_evolution(history, psi0_true, title='Evolução de ψ₀'):
    """
    Plota a evolução do potencial de superfície durante o treinamento.

    Args:
        history:   dicionário com 'psi0' — lista de valores por época
        psi0_true: valor verdadeiro de psi0
        title:     título do plot
    """
    epochs = list(range(len(history['psi0'])))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=history['psi0'],
        mode='lines', name='ψ₀ recuperado',
        line=dict(color='#2C3E50', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]],
        y=[psi0_true, psi0_true],
        mode='lines', name='ψ₀ verdadeiro',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis('Época'),
        yaxis=_square_axis('ψ₀'),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()


def plot_pb(results, X_obs, Psi_obs, title='Equação de Poisson-Boltzmann'):
    """
    Plota a solução da PINN, a solução exata e a curva recuperada,
    junto com os dados de observação ruidosos.

    Args:
        results:  dicionário retornado por evaluate_pb
        X_obs:    tensor (N_obs, 1) — posições das observações
        Psi_obs:  tensor (N_obs, 1) — potencial ruidoso
        title:    título do plot
    """
    X_obs_np   = X_obs.detach().cpu().numpy().ravel()
    Psi_obs_np = Psi_obs.detach().cpu().numpy().ravel()

    fig = go.Figure()

    # dados ruidosos
    fig.add_trace(go.Scatter(
        x=X_obs_np, y=Psi_obs_np,
        mode='markers', name='Observações',
        marker=dict(color='#2C3E50', size=7, symbol='circle-open')
    ))

    # solução exata
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_exact'],
        mode='lines', name=f"Exata (ψ₀={results['psi0_true']:.3f})",
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    # curva recuperada
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_recovered'],
        mode='lines', name=f"Recuperada (ψ₀={results['psi0_pred']:.3f})",
        line=dict(color='#2980B9', width=2, dash='dot')
    ))

    # predição da rede
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_pred'],
        mode='lines', name='PINN',
        line=dict(color='#27AE60', width=2)
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"{title}<br><sup>Erro em ψ₀: {results['error_pct']:.2f}% | "
                 f"Erro L2: {results['l2_error']:.2e}</sup>",
            font=dict(size=FONT_SIZE + 2)
        ),
        xaxis=_square_axis('x (comprimentos de Debye)'),
        yaxis=_square_axis('ψ(x)'),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()

def plot_D_evolution(history, D_true, title='Evolução de D'):
    """
    Plota a evolução do coeficiente de difusão durante o treinamento.

    Args:
        history: dicionário com 'D' — lista de valores por época
        D_true:  valor verdadeiro de D
        title:   título do plot
    """
    epochs = list(range(len(history['D'])))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=history['D'],
        mode='lines', name='D recuperado',
        line=dict(color='#2C3E50', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]],
        y=[D_true, D_true],
        mode='lines', name='D verdadeiro',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis('Época'),
        yaxis=_square_axis('D'),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()


def plot_diffusion_snapshots(results, title='Difusão 2D'):
    """
    Plota snapshots temporais da solução predita vs referência.

    Args:
        results: dicionário retornado por evaluate_diffusion
        title:   título do plot
    """
    n_snaps = len(results['snap_times'])
    x = results['x']
    y = results['y']

    fig = make_subplots(
        rows=2, cols=n_snaps,
        subplot_titles=[f"Predição t={t:.2f}" for t in results['snap_times']] +
                       [f"Referência t={t:.2f}" for t in results['snap_times']],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    vmin = min(np.min(results['C_ref_snaps']), np.min(results['C_pred_snaps']))
    vmax = max(np.max(results['C_ref_snaps']), np.max(results['C_pred_snaps']))

    for col, (C_pred, C_ref) in enumerate(
        zip(results['C_pred_snaps'], results['C_ref_snaps']), start=1
    ):
        fig.add_trace(go.Heatmap(
            z=C_pred, x=x, y=y,
            colorscale=COLORSCALE,
            zmin=vmin, zmax=vmax,
            showscale=(col == n_snaps),
            colorbar=dict(x=1.02, len=0.45, y=0.75, thickness=12),
        ), row=1, col=col)

        fig.add_trace(go.Heatmap(
            z=C_ref, x=x, y=y,
            colorscale=COLORSCALE,
            zmin=vmin, zmax=vmax,
            showscale=(col == n_snaps),
            colorbar=dict(x=1.02, len=0.45, y=0.25, thickness=12),
        ), row=2, col=col)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"{title}<br><sup>D recuperado: {results['D_pred']:.4f} | "
                 f"D verdadeiro: {results['D_true']:.4f} | "
                 f"Erro: {results['error_pct']:.2f}%</sup>",
            font=dict(size=FONT_SIZE + 2)
        ),
        height=700,
        width=300 * n_snaps + 100,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE - 2)

    fig.show()

def plot_campo_2d(U, mask, title="Campo u(x,y)", H=64, W=64):
    """
    Plota o campo u no domínio L-shape usando heatmap do Plotly.
    Pontos fora do domínio aparecem em branco (None).
    """
    x1d = np.linspace(0, 1, W)
    y1d = np.linspace(0, 1, H)

    # substitui fora do domínio por None
    U_plot = np.where(mask.cpu().numpy(), U, np.nan)

    fig = go.Figure(go.Heatmap(
        z=U_plot,
        x=x1d,
        y=y1d,
        colorscale=COLORSCALE,
        colorbar=dict(title=dict(text="u(x,y)", font=dict(size=FONT_SIZE))
),
    ))

    fig.update_xaxes(title_text="x", **AXIS_COMMON)
    fig.update_yaxes(title_text="y", **AXIS_COMMON)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=SQ_SIZE,
        width=SQ_SIZE,
    )
    fig.show()


def plot_loss_comparacao(historico_mlp, historico_geo,
                         title="Comparação de treinamento"):
    """
    Plota curvas de loss (total e PDE) do MLP e PhyGeoNet lado a lado.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss total", "Loss PDE"),
    )

    # loss total
    fig.add_trace(go.Scatter(
        y=historico_mlp["loss"], name="MLP",
        line=dict(color=COLORS[0], width=2)),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        y=historico_geo["total"], name="PhyGeoNet",
        line=dict(color=COLORS[1], width=2)),
        row=1, col=1)

    # loss PDE
    fig.add_trace(go.Scatter(
        y=historico_mlp["loss_pde"], name="MLP",
        line=dict(color=COLORS[0], width=2, dash="dash"),
        showlegend=False),
        row=1, col=2)
    fig.add_trace(go.Scatter(
        y=historico_geo["pde"], name="PhyGeoNet",
        line=dict(color=COLORS[1], width=2, dash="dash"),
        showlegend=False),
        row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(title_text="Época", row=1, col=col, **AXIS_COMMON)
        fig.update_yaxes(title_text="Loss",  row=1, col=col,
                         type="log", **AXIS_COMMON)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=450,
        width=1000,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


def plot_residuo_scatter(X_val, residuo, title="Resíduo |−∇²u − 1|"):
    """
    Plota o resíduo pontual do MLP como scatter no domínio L-shape.
    Útil para identificar onde o MLP erra mais.
    """
    fig = go.Figure(go.Scatter(
        x=X_val[:, 0].cpu().detach().numpy(),
        y=X_val[:, 1].cpu().detach().numpy(),
        mode="markers",
        marker=dict(
            color=residuo,
            colorscale=COLORSCALE,
            size=4,
            colorbar=dict(title=dict(text="|resíduo|", font=dict(size=FONT_SIZE))),
        ),
    ))

    fig.update_xaxes(title_text="x", range=[-0.02, 1.02], **AXIS_COMMON)
    fig.update_yaxes(title_text="y", range=[-0.02, 1.02], **AXIS_COMMON)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=SQ_SIZE,
        width=SQ_SIZE,
    )
    fig.show()

def plot_sampling_points(point_sets, labels, title='Distribuição dos pontos de colocação'):
    """
    Plota a distribuição dos pontos de colocação para cada estratégia
    de amostragem em subplots lado a lado.

    Args:
        point_sets: lista de arrays (N, 2) — um por estratégia
        labels:     lista de strings — nome de cada estratégia
        title:      título do plot
    """
    n = len(point_sets)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=labels,
        horizontal_spacing=0.08,
    )

    for col, (pts, label) in enumerate(zip(point_sets, labels), start=1):
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            mode='markers',
            name=label,
            marker=dict(size=3, opacity=0.6, color=COLORS[col-1]),
            showlegend=False,
        ), row=1, col=col)

        fig.update_xaxes(
            **AXIS_COMMON,
            title_text='x',
            title_font=dict(size=FONT_SIZE),
            row=1, col=col,
        )
        fig.update_yaxes(
            **AXIS_COMMON,
            title_text='z' if col == 1 else '',
            title_font=dict(size=FONT_SIZE),
            row=1, col=col,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        width=n * (SQ_SIZE // 2) + 80,
        height=SQ_SIZE // 2 + 120,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


def plot_loss_comparison(histories, labels, title='Comparação das curvas de aprendizado'):
    """
    Plota as curvas de aprendizado de múltiplos experimentos no mesmo gráfico.

    Args:
        histories: lista de dicionários — um por estratégia
        labels:    lista de strings — nome de cada estratégia
        title:     título do plot
    """
    fig = go.Figure()

    for history, label, color in zip(histories, labels, COLORS):
        epochs = list(range(len(history['loss'])))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['loss'],
            mode='lines', name=label,
            line=dict(color=color, width=2)
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis('Época'),
        yaxis=_square_axis('Perda total', log=True),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.update_yaxes(
        type='log',
        dtick=1,
        exponentformat='power',
        showexponent='all',
    )

    fig.show()


def plot_heatmaps_comparison(results_list, labels, title='Comparação das soluções'):
    """
    Plota heatmaps comparativos para múltiplas estratégias de amostragem.
    Cada linha corresponde a uma estratégia: predição e erro absoluto.
    A referência é mostrada uma única vez.

    Args:
        results_list: lista de dicionários retornados por evaluate_helmholtz
        labels:       lista de strings — nome de cada estratégia
        title:        título do plot
    """
    n = len(results_list)
    x = results_list[0]['x']
    z = results_list[0]['z']

    # referência é a mesma para todos
    U_ref = results_list[0]['U_ref']
    vmin  = U_ref.min()
    vmax  = U_ref.max()

    subplot_titles = []
    for label in labels:
        subplot_titles += [f'Predição — {label}', f'Erro — {label}']
    subplot_titles = ['Referência', ''] + subplot_titles

    fig = make_subplots(
        rows=n + 1, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.08,
    )

    # referência na primeira linha
    fig.add_trace(go.Heatmap(
        z=U_ref, x=x, y=z,
        colorscale=COLORSCALE,
        zmin=vmin, zmax=vmax,
        showscale=True,
        colorbar=dict(x=0.46, len=1/(n+1)*0.9,
                      y=1 - 0.5/(n+1), thickness=12),
    ), row=1, col=1)

    # célula vazia na coluna 2 da linha 1
    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=2)

    # uma linha por estratégia
    for row, (results, label) in enumerate(zip(results_list, labels), start=2):
        E_abs = np.abs(results['U_pred'] - U_ref)

        fig.add_trace(go.Heatmap(
            z=results['U_pred'], x=x, y=z,
            colorscale=COLORSCALE,
            zmin=vmin, zmax=vmax,
            showscale=False,
        ), row=row, col=1)

        fig.add_trace(go.Heatmap(
            z=E_abs, x=x, y=z,
            colorscale='Reds',
            zmin=0, zmax=E_abs.max(),
            showscale=True,
            colorbar=dict(
                x=1.02,
                len=1/(n+1)*0.9,
                y=1 - (row - 0.5)/(n+1),
                thickness=12,
                title=f'L2={results["l2_error"]:.2e}',
                title_side='right',
            ),
        ), row=row, col=2)

    for r in range(1, n + 2):
        for c in [1, 2]:
            fig.update_xaxes(
                **AXIS_COMMON,
                title_text='x',
                title_font=dict(size=FONT_SIZE - 2),
                row=r, col=c,
            )
            fig.update_yaxes(
                **AXIS_COMMON,
                title_text='z' if c == 1 else '',
                title_font=dict(size=FONT_SIZE - 2),
                row=r, col=c,
            )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=300 * (n + 1) + 100,
        width=800,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE - 1)

    fig.show()


def plot_l2_comparison(results_list, labels, title='Erro L² por estratégia de amostragem'):
    """
    Plot de barras comparando o erro L2 de cada estratégia.

    Args:
        results_list: lista de dicionários retornados por evaluate_helmholtz
        labels:       lista de strings — nome de cada estratégia
        title:        título do plot
    """
    l2_errors = [r['l2_error'] for r in results_list]

    fig = go.Figure(go.Bar(
        x=labels,
        y=l2_errors,
        marker_color=COLORS[:len(labels)],
        text=[f'{e:.2e}' for e in l2_errors],
        textposition='outside',
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text='Estratégia', font=dict(size=FONT_SIZE)),
        ),
        yaxis=dict(
            **AXIS_COMMON,
            title=dict(text='Erro L²', font=dict(size=FONT_SIZE)),
            type='log',
            dtick=1,
            exponentformat='power',
            showexponent='all',
        ),
        width=SQ_SIZE,
        height=SQ_SIZE,
    )

    fig.show()

#