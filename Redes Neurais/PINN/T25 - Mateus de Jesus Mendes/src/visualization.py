"""
src/visualization.py
====================
Visualizações interativas com Plotly para o projeto de PINNs.

Figuras disponíveis:
    1. plot_solution_comparison  — θ(t): PINN vs numérica vs linear analítica
    2. plot_phase_portrait       — retrato de fase (θ, θ')
    3. animate_pendulum          — animação 2D do movimento do pêndulo
    4. plot_loss_history         — curvas de perda ao longo do treinamento
    5. plot_noisy_data           — dados sintéticos ruidosos + verdade
    6. plot_g_convergence        — convergência do parâmetro g (problema inverso)
    7. plot_inverse_comparison   — comparação final do problema inverso
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


# ── Paleta de cores consistente ───────────────────────────────────────────────
COLORS = {
    "numerical": "#1565C0",   # azul escuro — "verdade"
    "pinn":      "#E53935",   # vermelho    — PINN
    "linear":    "#2E7D32",   # verde       — aprox. linear
    "data":      "#FF6F00",   # âmbar       — dados observados
    "g_true":    "#1B5E20",   # verde escuro — valor verdadeiro
    "rod":       "#424242",   # cinza escuro — haste do pêndulo
    "bob":       "#E53935",   # vermelho     — bob
    "pivot":     "#212121",   # preto        — pino
    "trail":     "#BDBDBD",   # cinza claro  — rastro
}

TEMPLATE = "plotly_white"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Comparação de soluções
# ══════════════════════════════════════════════════════════════════════════════

def plot_solution_comparison(
    t: np.ndarray,
    theta_pinn: np.ndarray,
    theta_numerical: np.ndarray,
    theta_linear: np.ndarray,
    theta0_deg: float,
) -> go.Figure:
    """
    Compara θ(t) obtido pela PINN, pela solução numérica (RK45)
    e pela aproximação linear analítica.

    Parameters
    ----------
    t : np.ndarray
        Vetor de tempo comum às três soluções.
    theta_pinn : np.ndarray
        Predição da PINN.
    theta_numerical : np.ndarray
        Solução numérica de referência.
    theta_linear : np.ndarray
        Aproximação linear.
    theta0_deg : float
        Amplitude inicial em graus (para o título).
    """
    fig = go.Figure()

    # Solução numérica (verdade)
    fig.add_trace(go.Scatter(
        x=t, y=theta_numerical,
        name="Solução Numérica (RK45)",
        line=dict(color=COLORS["numerical"], width=3),
        hovertemplate="t = %{x:.3f} s<br>θ = %{y:.4f} rad<extra>RK45</extra>",
    ))

    # PINN
    fig.add_trace(go.Scatter(
        x=t, y=theta_pinn,
        name="PINN (não-linear)",
        line=dict(color=COLORS["pinn"], width=2.5, dash="dash"),
        hovertemplate="t = %{x:.3f} s<br>θ = %{y:.4f} rad<extra>PINN</extra>",
    ))

    # Linear analítica
    fig.add_trace(go.Scatter(
        x=t, y=theta_linear,
        name="Aprox. Linear (analítica)",
        line=dict(color=COLORS["linear"], width=2, dash="dot"),
        hovertemplate="t = %{x:.3f} s<br>θ = %{y:.4f} rad<extra>Linear</extra>",
    ))

    # Linha θ = 0
    fig.add_hline(y=0, line=dict(color="#9E9E9E", width=1, dash="solid"))

    fig.update_layout(
        title=dict(
            text=f"<b>Pêndulo Simples — Problema Direto</b>"
                 f"<br><sup>θ₀ = {theta0_deg:.0f}°  |  Comparação: PINN × Numérica × Analítica Linear</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title="Tempo (s)", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Ângulo θ (rad)", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#BDBDBD",
            borderwidth=1,
        ),
        template=TEMPLATE,
        height=450,
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Retrato de Fase
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_portrait(
    theta_pinn: np.ndarray,
    dtheta_pinn: np.ndarray,
    theta_numerical: np.ndarray,
    dtheta_numerical: np.ndarray,
    theta_linear: np.ndarray,
    dtheta_linear: np.ndarray,
) -> go.Figure:
    """
    Retrato de fase (θ vs θ') para as três soluções.

    O retrato de fase revela a estrutura geométrica da dinâmica:
      - Trajetórias fechadas = movimento periódico;
      - A curvatura da órbita indica a não-linearidade.
    """
    fig = go.Figure()

    # Marcadores de início (t=0)
    for theta, dtheta, name, color in [
        (theta_numerical, dtheta_numerical, "RK45", COLORS["numerical"]),
        (theta_pinn, dtheta_pinn, "PINN", COLORS["pinn"]),
        (theta_linear, dtheta_linear, "Linear", COLORS["linear"]),
    ]:
        fig.add_trace(go.Scatter(
            x=theta, y=dtheta,
            name=name,
            mode="lines",
            line=dict(
                color=color,
                width=2.5 if name == "RK45" else 2,
                dash="solid" if name == "RK45" else ("dash" if name == "PINN" else "dot"),
            ),
        ))
        # Ponto de início
        fig.add_trace(go.Scatter(
            x=[theta[0]], y=[dtheta[0]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="circle-open", line=dict(width=2)),
            showlegend=False,
            hovertemplate=f"t=0: θ={theta[0]:.3f}, θ'={dtheta[0]:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="<b>Retrato de Fase — Pêndulo Simples</b>"
                 "<br><sup>Órbitas fechadas indicam movimento periódico; "
                 "a diferença entre PINN e Linear mostra o efeito da não-linearidade</sup>",
            font=dict(size=16),
        ),
        xaxis=dict(title="Ângulo θ (rad)", zeroline=True, zerolinewidth=1.5,
                   zerolinecolor="#9E9E9E", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Velocidade angular θ' (rad/s)", zeroline=True,
                   zerolinewidth=1.5, zerolinecolor="#9E9E9E",
                   showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#BDBDBD", borderwidth=1),
        template=TEMPLATE,
        height=500,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Animação do Pêndulo
# ══════════════════════════════════════════════════════════════════════════════

def animate_pendulum(
    t: np.ndarray,
    theta_pinn: np.ndarray,
    theta_numerical: np.ndarray,
    L: float = 1.0,
    skip: int = 4,
    frame_duration_ms: int = 40,
) -> go.Figure:
    """
    Animação 2D interativa do pêndulo com dois painéis lado a lado:
      - Esquerda: animação da haste + bob (PINN vs Numérica)
      - Direita:  θ(t) em tempo real com cursor deslizante

    Parameters
    ----------
    t, theta_pinn, theta_numerical : np.ndarray
        Arrays de mesmo comprimento.
    L : float
        Comprimento do pêndulo [m].
    skip : int
        Tomar 1 frame a cada `skip` passos (controla velocidade).
    frame_duration_ms : int
        Duração de cada frame em ms.
    """
    idx = range(0, len(t), skip)
    t_frames = t[list(idx)]

    # Posições cartesianas
    x_pinn = L * np.sin(theta_pinn)
    y_pinn = -L * np.cos(theta_pinn)
    x_num = L * np.sin(theta_numerical)
    y_num = -L * np.cos(theta_numerical)

    pad = 0.15 * L

    # ── Construção dos frames ─────────────────────────────────────────────────
    frames = []
    trail_len = 30  # comprimento do rastro

    for frame_i, data_i in enumerate(idx):
        trail_start = max(0, data_i - trail_len * skip)

        # Rastro PINN
        trail_x_p = x_pinn[trail_start:data_i + 1]
        trail_y_p = y_pinn[trail_start:data_i + 1]

        # Rastro numérica
        trail_x_n = x_num[trail_start:data_i + 1]
        trail_y_n = y_num[trail_start:data_i + 1]

        frame_data = [
            # ── Rastros ───────────────────────────────────────────────────────
            go.Scatter(   # rastro PINN
                x=trail_x_p, y=trail_y_p,
                mode="lines",
                line=dict(color=COLORS["pinn"], width=1.5),
                opacity=0.4,
                showlegend=False,
                xaxis="x", yaxis="y",
            ),
            go.Scatter(   # rastro numérica
                x=trail_x_n, y=trail_y_n,
                mode="lines",
                line=dict(color=COLORS["numerical"], width=1.5),
                opacity=0.4,
                showlegend=False,
                xaxis="x", yaxis="y",
            ),
            # ── Hastes ────────────────────────────────────────────────────────
            go.Scatter(   # haste PINN
                x=[0, x_pinn[data_i]], y=[0, y_pinn[data_i]],
                mode="lines",
                line=dict(color=COLORS["pinn"], width=3),
                showlegend=False,
                xaxis="x", yaxis="y",
            ),
            go.Scatter(   # haste numérica
                x=[0, x_num[data_i]], y=[0, y_num[data_i]],
                mode="lines",
                line=dict(color=COLORS["numerical"], width=3),
                showlegend=False,
                xaxis="x", yaxis="y",
            ),
            # ── Bobs ──────────────────────────────────────────────────────────
            go.Scatter(   # bob PINN
                x=[x_pinn[data_i]], y=[y_pinn[data_i]],
                mode="markers",
                marker=dict(size=20, color=COLORS["pinn"],
                            line=dict(color="white", width=2)),
                name="PINN",
                xaxis="x", yaxis="y",
            ),
            go.Scatter(   # bob numérica
                x=[x_num[data_i]], y=[y_num[data_i]],
                mode="markers",
                marker=dict(size=20, color=COLORS["numerical"],
                            symbol="diamond",
                            line=dict(color="white", width=2)),
                name="RK45",
                xaxis="x", yaxis="y",
            ),
            # ── Pivot ─────────────────────────────────────────────────────────
            go.Scatter(
                x=[0], y=[0],
                mode="markers",
                marker=dict(size=12, color=COLORS["pivot"]),
                showlegend=False,
                xaxis="x", yaxis="y",
            ),
            # ── Gráfico θ(t) em tempo real ────────────────────────────────────
            go.Scatter(   # θ PINN
                x=t[:data_i + 1], y=theta_pinn[:data_i + 1],
                mode="lines",
                line=dict(color=COLORS["pinn"], width=2),
                showlegend=False,
                xaxis="x2", yaxis="y2",
            ),
            go.Scatter(   # θ numérica
                x=t[:data_i + 1], y=theta_numerical[:data_i + 1],
                mode="lines",
                line=dict(color=COLORS["numerical"], width=2, dash="dot"),
                showlegend=False,
                xaxis="x2", yaxis="y2",
            ),
            go.Scatter(   # cursor vertical
                x=[t[data_i], t[data_i]],
                y=[theta_numerical.min() * 1.1, theta_numerical.max() * 1.1],
                mode="lines",
                line=dict(color="#9E9E9E", width=1, dash="dash"),
                showlegend=False,
                xaxis="x2", yaxis="y2",
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(frame_i)))

    # ── Layout ────────────────────────────────────────────────────────────────
    lim = L + pad
    theta_max = max(abs(theta_pinn).max(), abs(theta_numerical).max()) * 1.15

    # Dados iniciais
    initial_data = [
        go.Scatter(x=[], y=[], mode="lines",
                   line=dict(color=COLORS["pinn"], width=1.5), opacity=0.4,
                   showlegend=False, xaxis="x", yaxis="y"),
        go.Scatter(x=[], y=[], mode="lines",
                   line=dict(color=COLORS["numerical"], width=1.5), opacity=0.4,
                   showlegend=False, xaxis="x", yaxis="y"),
        go.Scatter(x=[0, x_pinn[0]], y=[0, y_pinn[0]], mode="lines",
                   line=dict(color=COLORS["pinn"], width=3),
                   showlegend=False, xaxis="x", yaxis="y"),
        go.Scatter(x=[0, x_num[0]], y=[0, y_num[0]], mode="lines",
                   line=dict(color=COLORS["numerical"], width=3),
                   showlegend=False, xaxis="x", yaxis="y"),
        go.Scatter(x=[x_pinn[0]], y=[y_pinn[0]], mode="markers",
                   marker=dict(size=20, color=COLORS["pinn"],
                               line=dict(color="white", width=2)),
                   name="PINN", xaxis="x", yaxis="y"),
        go.Scatter(x=[x_num[0]], y=[y_num[0]], mode="markers",
                   marker=dict(size=20, color=COLORS["numerical"], symbol="diamond",
                               line=dict(color="white", width=2)),
                   name="RK45", xaxis="x", yaxis="y"),
        go.Scatter(x=[0], y=[0], mode="markers",
                   marker=dict(size=12, color=COLORS["pivot"]),
                   showlegend=False, xaxis="x", yaxis="y"),
        go.Scatter(x=[t[0]], y=[theta_pinn[0]], mode="lines",
                   line=dict(color=COLORS["pinn"], width=2),
                   showlegend=False, xaxis="x2", yaxis="y2"),
        go.Scatter(x=[t[0]], y=[theta_numerical[0]], mode="lines",
                   line=dict(color=COLORS["numerical"], width=2, dash="dot"),
                   showlegend=False, xaxis="x2", yaxis="y2"),
        go.Scatter(x=[t[0], t[0]], y=[-theta_max, theta_max], mode="lines",
                   line=dict(color="#9E9E9E", width=1, dash="dash"),
                   showlegend=False, xaxis="x2", yaxis="y2"),
    ]

    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="<b>Animação do Pêndulo Simples</b>"
                     "<br><sup>PINN (vermelho) vs Solução Numérica RK45 (azul)</sup>",
                font=dict(size=16),
            ),
            xaxis=dict(
                domain=[0, 0.45],
                range=[-lim, lim],
                scaleanchor="y",
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            yaxis=dict(
                range=[-lim, lim * 0.4],
                showgrid=False,
                zeroline=False,
                showticklabels=False,
            ),
            xaxis2=dict(
                domain=[0.55, 1.0],
                title="Tempo (s)",
                range=[t[0], t[-1]],
                showgrid=True, gridcolor="#EEEEEE",
            ),
            yaxis2=dict(
                title="θ (rad)",
                range=[-theta_max, theta_max],
                showgrid=True, gridcolor="#EEEEEE",
                anchor="x2",
            ),
            template=TEMPLATE,
            height=520,
            legend=dict(
                x=0.5, y=1.0,
                xanchor="center",
                orientation="h",
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.22, y=-0.12,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶  Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸  Pausar",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(i)], {"mode": "immediate",
                                         "frame": {"duration": frame_duration_ms, "redraw": True},
                                         "transition": {"duration": 0}}],
                        label=f"{t_frames[i]:.2f}s",
                    )
                    for i in range(len(t_frames))
                ],
                transition={"duration": 0},
                x=0.0, y=-0.05,
                currentvalue=dict(prefix="t = ", font=dict(size=13)),
                len=1.0,
                pad={"t": 30},
            )],
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Curvas de Perda
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss_history(
    history: dict,
    title: str = "Histórico de Treinamento (Adam)",
) -> go.Figure:
    """
    Curvas de perda em escala log para diagnóstico de treinamento.
    Plota: loss total, perda física e perda de CI.
    """
    epochs = history["epoch"]
    fig = go.Figure()

    components = [
        ("loss", "Loss Total", "#212121", "solid", 2.5),
        ("loss_phys", "Perda Física (resíduo)", COLORS["numerical"], "dash", 2),
        ("loss_ic", "Perda CI", COLORS["linear"], "dot", 2),
    ]
    if "loss_data" in history:
        components.append(("loss_data", "Perda de Dados", COLORS["data"], "dashdot", 2))

    for key, name, color, dash, width in components:
        if key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key],
                name=name,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
            ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15)),
        xaxis=dict(title="Época", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="Perda (escala log)", type="log",
                   showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.99, y=0.99, xanchor="right",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#BDBDBD", borderwidth=1),
        template=TEMPLATE,
        height=420,
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dados Sintéticos Ruidosos
# ══════════════════════════════════════════════════════════════════════════════

def plot_noisy_data(
    t_full: np.ndarray,
    theta_full: np.ndarray,
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    noise_std: float,
) -> go.Figure:
    """
    Visualiza os dados observados (ruidosos) sobre a trajetória verdadeira.
    """
    fig = go.Figure()

    # Trajetória verdadeira
    fig.add_trace(go.Scatter(
        x=t_full, y=theta_full,
        name="Trajetória verdadeira (g = 9,81 m/s²)",
        mode="lines",
        line=dict(color=COLORS["numerical"], width=2.5),
    ))

    # Bandas de incerteza (±noise_std)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_full, t_full[::-1]]),
        y=np.concatenate([theta_full + noise_std, (theta_full - noise_std)[::-1]]),
        fill="toself",
        fillcolor="rgba(21, 101, 192, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±{noise_std:.3f} rad",
        showlegend=True,
    ))

    # Observações ruidosas
    fig.add_trace(go.Scatter(
        x=t_obs.flatten(), y=theta_obs.flatten(),
        name="Observações (com ruído)",
        mode="markers",
        marker=dict(
            color=COLORS["data"], size=7,
            symbol="circle-open",
            line=dict(width=1.5),
        ),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Problema Inverso — Dados Sintéticos</b>"
                 f"<br><sup>Ruído gaussiano σ = {noise_std:.3f} rad  |  "
                 f"g verdadeiro = 9,81 m/s²  (desconhecido para a PINN)</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(title="Tempo (s)", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="θ (rad)", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#BDBDBD", borderwidth=1),
        template=TEMPLATE,
        height=430,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6. Convergência do Parâmetro g
# ══════════════════════════════════════════════════════════════════════════════

def plot_g_convergence(
    history: dict,
    g_true: float,
    g_init: float,
) -> go.Figure:
    """
    Mostra a evolução de g_estimado ao longo do treinamento Adam.
    """
    epochs = history["epoch"]
    g_vals = history["g_value"][:len(epochs)]  # apenas épocas Adam

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Convergência de g", "Erro relativo (%)"],
        horizontal_spacing=0.12,
    )

    # Painel 1: valor de g
    fig.add_trace(go.Scatter(
        x=epochs, y=g_vals,
        name="g estimado",
        mode="lines",
        line=dict(color=COLORS["pinn"], width=2.5),
    ), row=1, col=1)

    fig.add_hline(
        y=g_true,
        line=dict(color=COLORS["g_true"], width=2, dash="dash"),
        annotation_text=f"g verdadeiro = {g_true:.4f} m/s²",
        annotation_position="top right",
        row=1, col=1,
    )
    fig.add_hline(
        y=g_init,
        line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        annotation_text=f"g inicial = {g_init:.1f} m/s²",
        annotation_position="bottom right",
        row=1, col=1,
    )

    # Painel 2: erro relativo em %
    err_pct = [abs(g - g_true) / g_true * 100 for g in g_vals]
    fig.add_trace(go.Scatter(
        x=epochs, y=err_pct,
        name="Erro relativo (%)",
        mode="lines",
        line=dict(color=COLORS["data"], width=2.5),
        showlegend=True,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Época", showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="g (m/s²)", row=1, col=1,
                     showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Erro relativo (%)", type="log",
                     row=1, col=2, showgrid=True, gridcolor="#EEEEEE")

    fig.update_layout(
        title=dict(
            text=f"<b>Problema Inverso — Identificação de g</b>"
                 f"<br><sup>Inicialização: g₀ = {g_init:.1f} m/s²  →  "
                 f"Valor verdadeiro: g = {g_true:.4f} m/s²</sup>",
            font=dict(size=15),
        ),
        template=TEMPLATE,
        height=430,
        hovermode="x unified",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7. Comparação Final — Problema Inverso
# ══════════════════════════════════════════════════════════════════════════════

def plot_inverse_comparison(
    t: np.ndarray,
    theta_pinn: np.ndarray,
    theta_numerical: np.ndarray,
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    g_estimated: float,
    g_true: float,
) -> go.Figure:
    """
    Comparação final do problema inverso:
    PINN treinada com g_estimado vs solução numérica com g_true.
    """
    error_pct = abs(g_estimated - g_true) / g_true * 100

    fig = go.Figure()

    # Solução com g verdadeiro
    fig.add_trace(go.Scatter(
        x=t, y=theta_numerical,
        name=f"RK45  (g = {g_true:.4f} m/s²)",
        mode="lines",
        line=dict(color=COLORS["numerical"], width=3),
    ))

    # PINN com g estimado
    fig.add_trace(go.Scatter(
        x=t, y=theta_pinn,
        name=f"PINN  (g = {g_estimated:.4f} m/s², erro = {error_pct:.3f}%)",
        mode="lines",
        line=dict(color=COLORS["pinn"], width=2.5, dash="dash"),
    ))

    # Dados observados
    fig.add_trace(go.Scatter(
        x=t_obs.flatten(), y=theta_obs.flatten(),
        name="Observações ruidosas",
        mode="markers",
        marker=dict(color=COLORS["data"], size=6, symbol="circle-open",
                    line=dict(width=1.5)),
    ))

    fig.update_layout(
        title=dict(
            text="<b>Problema Inverso — Resultado Final</b>"
                 f"<br><sup>g estimado = {g_estimated:.5f} m/s²  |  "
                 f"g verdadeiro = {g_true:.4f} m/s²  |  "
                 f"Erro relativo = {error_pct:.3f}%</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(title="Tempo (s)", showgrid=True, gridcolor="#EEEEEE"),
        yaxis=dict(title="θ (rad)", showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#BDBDBD", borderwidth=1),
        template=TEMPLATE,
        height=450,
        hovermode="x unified",
    )
    return fig
