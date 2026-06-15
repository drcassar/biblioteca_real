# -*- coding: utf-8 -*-
"""
src_SDE.plots
=============
Visualizações interativas paper-like via Plotly — versão aprimorada.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, gaussian_kde
from scipy.stats import lognorm
from .black_scholes import bs_call, bs_vega

from .stochastic import variacao_quadratica


# ── Paleta e tema ─────────────────────────────────────────────────────────────
AZUL    = "#264653"
VERDE   = "#2a9d8f"
LARANJA = "#e76f51"
AMARELO = "#e9c46a"
CINZA   = "#8d99ae"
ROXO    = "#7b2d8b"
ROSA    = "#c77dff"
AZUL2   = "#457b9d"
VERMELHO= "#d62828"

CORES_SEQ = [AZUL, VERDE, LARANJA, ROXO, AMARELO, CINZA, ROSA, VERMELHO]

TEMPLATE = "plotly_white"
FONTE    = dict(family="Times New Roman, serif", size=13)
FONTE_SM = dict(family="Times New Roman, serif", size=11)

def _base_layout(**kw):
    base = dict(
        template=TEMPLATE, font=FONTE,
        paper_bgcolor="white", plot_bgcolor="#f8f9fa",
        legend=dict(bgcolor="rgba(255,255,255,0.90)", bordercolor="#dee2e6",
                    borderwidth=1, font=FONTE_SM),
        margin=dict(l=65, r=35, t=70, b=55),
    )
    base.update(kw)
    return base

def _axis(title="", **kw):
    return dict(title=dict(text=title, font=FONTE_SM),
                tickfont=FONTE_SM, showgrid=True,
                gridcolor="#e9ecef", gridwidth=1,
                zeroline=True, zerolinecolor="#adb5bd", zerolinewidth=1,
                **kw)

# ── 1. Processo de Wiener ─────────────────────────────────────────────────────
def fig_wiener(t, W, W_many=None):
    T = t[-1]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["(a) Trajetórias W<sub>t</sub>",
                        "(b) Distribuição de W<sub>T</sub>",
                        "(c) Variação Quadrática [W,W]<sub>t</sub>"],
        horizontal_spacing=0.09)

    # (a) Trajetórias com gradiente de cor
    n_paths = W.shape[0]
    for i, path in enumerate(W):
        frac  = i / max(n_paths - 1, 1)
        alpha = 0.85 if i == 0 else max(0.25, 0.55 - 0.3*frac)
        width = 2.2  if i == 0 else 0.9
        color = AZUL  if i == 0 else f"rgba(69,123,157,{alpha:.2f})"
        fig.add_trace(go.Scatter(
            x=t.tolist(), y=path.tolist(), mode="lines",
            line=dict(color=color, width=width),
            name="W<sub>t</sub>" if i == 0 else "",
            showlegend=(i == 0),
            hovertemplate="t=%{x:.3f}<br>W=%{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#adb5bd", line_width=1, row=1, col=1)

    # (b) Distribuição — histograma + curva
    W_T = (W_many if W_many is not None else W)[:, -1]
    x_g = np.linspace(-4, 4, 300)
    fig.add_trace(go.Histogram(
        x=W_T.tolist(), nbinsx=55, histnorm="probability density",
        marker=dict(color=AZUL, opacity=0.60,
                    line=dict(color="white", width=0.4)),
        name="W<sub>T</sub> empírico",
        hovertemplate="W=%{x:.2f}<br>Dens.=%{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_g.tolist(), y=norm.pdf(x_g, 0, np.sqrt(T)).tolist(),
        mode="lines", line=dict(color=LARANJA, width=2.8),
        name="𝒩(0,T) teórico"), row=1, col=2)

    # (c) Variação quadrática com banda de confiança
    vqs = np.array([variacao_quadratica(path) for path in W[:8]])
    vq_mean = vqs.mean(axis=0)
    vq_std  = vqs.std(axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t[1:], t[1:][::-1]]).tolist(),
        y=np.concatenate([vq_mean + vq_std, (vq_mean - vq_std)[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(42,157,143,0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="±1dp"), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=t[1:].tolist(), y=vq_mean.tolist(), mode="lines",
        line=dict(color=VERDE, width=2.0), name="VQ média", showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=t.tolist(), mode="lines",
        line=dict(color=LARANJA, width=2.5, dash="dash"),
        name="[W,W]<sub>t</sub> = t"), row=1, col=3)

    fig.update_xaxes(title_text="t", **{k: v for k,v in _axis().items() if k not in ['title']}, row=1, col=1)
    fig.update_xaxes(title_text="W<sub>T</sub>", **{k: v for k,v in _axis().items() if k not in ['title']}, row=1, col=2)
    fig.update_xaxes(title_text="t", **{k: v for k,v in _axis().items() if k not in ['title']}, row=1, col=3)
    fig.update_layout(height=400, title_text="<b>Figura 1</b> — Processo de Wiener",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig


# ── 2. GBM ────────────────────────────────────────────────────────────────────
def fig_gbm(t, S, S0, mu, sigma, T):
    n_show = min(50, S.shape[0])
    S_T   = S[:, -1]
    mu_l  = np.log(S0) + (mu - 0.5*sigma**2)*T
    sl_l  = sigma * np.sqrt(T)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["(a) Trajetórias S<sub>t</sub> (GBM)",
                        "(b) Distribuição de S<sub>T</sub>",
                        "(c) ln(S<sub>T</sub>) ∼ Normal"],
        horizontal_spacing=0.09)

    # (a) fan de trajetórias com envelope de percentis
    pct_lo = np.percentile(S, 10, axis=0)
    pct_hi = np.percentile(S, 90, axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]).tolist(),
        y=np.concatenate([pct_hi, pct_lo[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(38,70,83,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P10–P90", showlegend=True), row=1, col=1)
    for i in range(min(20, n_show)):
        fig.add_trace(go.Scatter(
            x=t.tolist(), y=S[i].tolist(), mode="lines",
            line=dict(color=f"rgba(38,70,83,{0.35 if i>0 else 0.9})", width=0.8 if i>0 else 2.0),
            showlegend=False, name=""), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=(S0*np.exp(mu*t)).tolist(), mode="lines",
        line=dict(color=LARANJA, width=2.8, dash="dash"),
        name="𝔼[S<sub>t</sub>] = S₀e<sup>μt</sup>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=(S0*np.exp((mu-0.5*sigma**2)*t)).tolist(), mode="lines",
        line=dict(color=VERDE, width=1.8, dash="dot"),
        name="e<sup>(μ−σ²/2)t</sup> (mediana)"), row=1, col=1)

    # (b) log-normal
    x_l = np.linspace(S_T.min()*0.7, S_T.max()*1.2, 300)
    fig.add_trace(go.Histogram(
        x=S_T.tolist(), nbinsx=65, histnorm="probability density",
        marker=dict(color=VERDE, opacity=0.60, line=dict(color="white", width=0.3)),
        name="S<sub>T</sub> simulado"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=x_l.tolist(), y=lognorm.pdf(x_l, s=sl_l, scale=np.exp(mu_l)).tolist(),
        mode="lines", line=dict(color=LARANJA, width=2.8),
        name="Log-Normal teórico"), row=1, col=2)

    # (c) ln(S_T) ~ Normal
    log_ST = np.log(S_T)
    x_n    = np.linspace(log_ST.min()-0.3, log_ST.max()+0.3, 300)
    fig.add_trace(go.Histogram(
        x=log_ST.tolist(), nbinsx=60, histnorm="probability density",
        marker=dict(color=ROXO, opacity=0.60, line=dict(color="white", width=0.3)),
        name="ln(S<sub>T</sub>)"), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=x_n.tolist(), y=norm.pdf(x_n, mu_l, sl_l).tolist(),
        mode="lines", line=dict(color=LARANJA, width=2.8),
        name="𝒩(μ<sub>ln</sub>, σ²<sub>ln</sub>)"), row=1, col=3)

    fig.update_layout(height=420,
                      title_text="<b>Figura 2</b> — GBM: trajetórias, distribuição de S<sub>T</sub> e normalidade log",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig


# ── 3. Black-Scholes analítico ────────────────────────────────────────────────
def fig_bs_analitico(S_grid, K, r, sigma, taus):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["(a) Preço da Call C(S, τ)",
                                        "(b) Monotonicidade em σ",
                                        "(c) Vega — máximo ATM"],
                        horizontal_spacing=0.09)
    cores = [AZUL, VERDE, LARANJA, ROXO]
    for tau, c in zip(taus, cores):
        y = bs_call(S_grid, K, r, sigma, tau).tolist()
        fig.add_trace(go.Scatter(x=S_grid.tolist(), y=y, mode="lines",
                                 line=dict(color=c, width=2.2),
                                 name=f"τ = {tau}"), row=1, col=1)
    # Linha de payoff intrínseco
    payoff = np.maximum(S_grid - K, 0)
    fig.add_trace(go.Scatter(x=S_grid.tolist(), y=payoff.tolist(), mode="lines",
                             line=dict(color=CINZA, width=1.5, dash="dot"),
                             name="Payoff (t=T)"), row=1, col=1)
    fig.add_vline(x=K, line_dash="dash", line_color=CINZA, line_width=1.2, row=1, col=1)

    sigs = np.linspace(0.05, 0.80, 200)
    for sv, c, lbl in zip([85, 100, 115], cores[:3], ["S=85 (OTM)", "S=100 (ATM)", "S=115 (ITM)"]):
        fig.add_trace(go.Scatter(x=sigs.tolist(),
                                 y=bs_call(sv, K, r, sigs, 0.5).tolist(),
                                 mode="lines", line=dict(color=c, width=2.2),
                                 name=lbl), row=1, col=2)
    # Marcador no sigma=0.20
    for sv, c in zip([85, 100, 115], cores[:3]):
        fig.add_trace(go.Scatter(
            x=[0.20], y=[bs_call(sv, K, r, 0.20, 0.5)],
            mode="markers", marker=dict(color=c, size=8, symbol="circle-open", line=dict(width=2)),
            showlegend=False), row=1, col=2)

    for tau, c in zip(taus, cores):
        vega = bs_vega(S_grid, K, r, sigma, tau).tolist()
        fig.add_trace(go.Scatter(x=S_grid.tolist(), y=vega, mode="lines",
                                 line=dict(color=c, width=2.2),
                                 name=f"τ = {tau}", showlegend=False), row=1, col=3)
    fig.add_vline(x=K, line_dash="dash", line_color=CINZA, line_width=1.2, row=1, col=3)
    # Anotação ATM
    fig.add_annotation(x=K, y=0, text="ATM", showarrow=False,
                       font=dict(size=10, color=CINZA), xref="x3", yref="y3")

    fig.update_layout(height=420,
                      title_text="<b>Figura 3</b> — Fórmula analítica de Black-Scholes: preços, σ-monotonicidade e Vega",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig


def fig_smile_iv(K_grid, sigma_true, IV_dict):
    fig = go.Figure()
    # Região sombreada do smile verdadeiro
    fig.add_trace(go.Scatter(
        x=K_grid.tolist(), y=(sigma_true*100).tolist(),
        mode="lines", line=dict(color="black", width=3, dash="dash"),
        name="σ(K) verdadeiro"))
    cores = [AZUL, VERDE, LARANJA, ROXO, VERMELHO]
    simbolos = ["circle", "square", "diamond", "cross", "x"]
    for (label, iv_arr), c, sym in zip(IV_dict.items(), cores, simbolos):
        valid = ~np.isnan(iv_arr)
        fig.add_trace(go.Scatter(
            x=K_grid[valid].tolist(), y=(iv_arr[valid]*100).tolist(),
            mode="lines+markers",
            line=dict(color=c, width=1.8),
            marker=dict(size=6, symbol=sym, color=c),
            name=label))
    # Anotação de instabilidade nas pontas
    fig.add_annotation(x=float(K_grid[2]), y=float(sigma_true[2]*100)+5,
                       text="Instabilidade<br>deep OTM", showarrow=True,
                       arrowhead=2, font=dict(size=10, color=VERMELHO))
    fig.update_layout(
        title_text="<b>Figura 4</b> — Volatility Smile: degradação da inversão pontual com ruído",
        xaxis=dict(title="Strike K", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="Volatilidade Implícita (%)", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=440, **_base_layout())
    return fig


def fig_superficie_bs_3d(S_grid, sigma_grid, K, r, tau):
    S_m, sig_m = np.meshgrid(S_grid, sigma_grid)
    Z = bs_call(S_m, K, r, sig_m, tau)
    # Linha de corte σ=0.20
    S_cut   = S_grid
    Z_cut   = bs_call(S_cut, K, r, 0.20, tau)
    sigma20_idx = np.argmin(np.abs(sigma_grid - 0.20))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=S_grid.tolist(), y=sigma_grid.tolist(), z=Z.tolist(),
        colorscale=[[0,"#264653"],[0.4,"#2a9d8f"],[0.7,"#e9c46a"],[1,"#e76f51"]],
        opacity=0.88, showscale=True,
        colorbar=dict(title=dict(text="C(S,σ)", side="right"),
                      thickness=14, len=0.7),
        hovertemplate="S=%{x:.1f}<br>σ=%{y:.3f}<br>C=%{z:.2f}<extra></extra>",
        name="C(S,σ)"))
    # Curva de corte σ=0.20
    fig.add_trace(go.Scatter3d(
        x=S_cut.tolist(), y=[0.20]*len(S_cut), z=Z_cut.tolist(),
        mode="lines", line=dict(color=LARANJA, width=6),
        name="σ = 0.20"))
    fig.update_layout(
        title_text="<b>Figura 3b</b> — Superfície 3D C(S, σ): problema inverso visualizado",
        scene=dict(
            xaxis=dict(title="S (Spot)", backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            yaxis=dict(title="σ (Volatilidade)", backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            zaxis=dict(title="C (Preço da Call)", backgroundcolor="#f8f9fa", gridcolor="#dee2e6"),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
        ),
        height=580, **_base_layout())
    return fig


# ── 4. Convergência numérica ──────────────────────────────────────────────────
def fig_convergencia(dts, err_em, err_mi):
    dts_a = np.array(dts); err_em_a = np.array(err_em); err_mi_a = np.array(err_mi)
    fig = go.Figure()
    # Regiões de referência sombreadas
    dt_r = np.array([dts[0], dts[-1]])
    fig.add_trace(go.Scatter(
        x=np.concatenate([dt_r, dt_r[::-1]]).tolist(),
        y=np.concatenate([0.4*dt_r**0.4, 0.2*dt_r**0.6]).tolist(),
        fill="toself", fillcolor="rgba(42,157,143,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False))

    fig.add_trace(go.Scatter(x=dts_a.tolist(), y=err_em_a.tolist(), mode="lines+markers",
                             line=dict(color=AZUL, width=2.5),
                             marker=dict(size=9, symbol="circle", color=AZUL,
                                         line=dict(width=2, color="white")),
                             name="Euler–Maruyama",
                             hovertemplate="Δt=%{x:.4f}<br>Erro=%{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dts_a.tolist(), y=err_mi_a.tolist(), mode="lines+markers",
                             line=dict(color=VERDE, width=2.5),
                             marker=dict(size=9, symbol="square", color=VERDE,
                                         line=dict(width=2, color="white")),
                             name="Milstein",
                             hovertemplate="Δt=%{x:.4f}<br>Erro=%{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=dt_r.tolist(), y=(0.30*dt_r**0.5).tolist(),
                             mode="lines", line=dict(color=CINZA, width=1.8, dash="dash"),
                             name="O(Δt<sup>1/2</sup>) referência"))
    fig.add_trace(go.Scatter(x=dt_r.tolist(), y=(0.30*dt_r**1.0).tolist(),
                             mode="lines", line=dict(color=LARANJA, width=1.8, dash="dot"),
                             name="O(Δt) referência"))
    fig.update_layout(
        title_text="<b>Figura 5</b> — Convergência de Euler–Maruyama e Milstein (escala log–log)",
        xaxis=dict(title="Δt", type="log", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="Erro no preço da opção", type="log", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=440, **_base_layout())
    return fig


# ── 5. Crank-Nicolson 3D ─────────────────────────────────────────────────────
def fig_crank_nicolson_3d(S_grid, tau_grid, V_grid, S_interior, V_an, V_cn0):
    mask_S = (S_grid >= 55) & (S_grid <= 165)
    S_3d   = S_grid[mask_S]
    V_3d   = V_grid[:, mask_S]
    erro   = np.abs(V_cn0 - V_an)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["(a) Superfície V(S, τ) — Crank-Nicolson",
                        "(b) t = 0: CN vs. Analítico + Erro"],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        horizontal_spacing=0.06)

    fig.add_trace(go.Surface(
        x=S_3d.tolist(), y=tau_grid.tolist(), z=V_3d.tolist(),
        colorscale=[[0,"#264653"],[0.35,"#2a9d8f"],[0.7,"#e9c46a"],[1,"#f4a261"]],
        showscale=True, opacity=0.92,
        colorbar=dict(x=0.44, len=0.85, thickness=13,
                      title=dict(text="V(S,τ)", side="right")),
        hovertemplate="S=%{x:.1f}<br>τ=%{y:.3f}<br>V=%{z:.2f}<extra></extra>"),
        row=1, col=1)
    fig.update_scenes(
        xaxis=dict(title="S", backgroundcolor="#f8f9fa"),
        yaxis=dict(title="τ = T−t", backgroundcolor="#f8f9fa"),
        zaxis=dict(title="V(S,τ)", backgroundcolor="#f8f9fa"),
        camera=dict(eye=dict(x=1.6, y=-1.9, z=1.3)))

    fig.add_trace(go.Scatter(
        x=S_interior.tolist(), y=V_an.tolist(), mode="lines",
        line=dict(color=AZUL, width=3.0), name="Analítico"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=S_interior.tolist(), y=V_cn0.tolist(), mode="lines",
        line=dict(color=LARANJA, width=2.0, dash="dash"), name="CN"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([S_interior, S_interior[::-1]]).tolist(),
        y=np.concatenate([V_an + erro*0.5, (V_an - erro*0.5)[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(231,111,81,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="|Erro| sombreado"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=S_interior.tolist(), y=erro.tolist(), mode="lines",
        line=dict(color=VERMELHO, width=1.5), yaxis="y3",
        name="|Erro| (eixo dir.)", showlegend=True), row=1, col=2)

    fig.update_layout(
        height=540,
        title_text="<b>Figura 6</b> — Crank-Nicolson: superfície 3D V(S,τ) e validação t=0",
        title_font=dict(size=15, family="Times New Roman, serif"),
        **_base_layout())
    return fig


# ── 6. Curvas de aprendizado ──────────────────────────────────────────────────
def fig_loss_curve(history, titulo="Curva de Aprendizado"):
    ep  = list(range(len(history["total"])))
    fig = go.Figure()
    pares = [("total","Total",AZUL,None,3.0),
             ("res","Residual EDP",LARANJA,None,1.8),
             ("ic","Cond. Inicial",VERDE,None,1.8),
             ("bc","Cond. Contorno",CINZA,"dot",1.5),
             ("kl","KL Divergência",ROXO,"dash",1.5),
             ("l_data","Dados",ROSA,None,1.8),
             ("l_res","Res. (B-PINN)",AMARELO,"dot",1.5)]
    for key, nome, cor, dash, lw in pares:
        if key not in history: continue
        ld = dict(color=cor, width=lw)
        if dash: ld["dash"] = dash
        fig.add_trace(go.Scatter(x=ep, y=history[key], mode="lines",
                                 line=ld, name=nome,
                                 hovertemplate="ep=%{x}<br>L=%{y:.3e}<extra></extra>"))
    fig.update_layout(
        title_text=titulo,
        xaxis=dict(title="Épocas", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="Perda (log)", type="log", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=400, **_base_layout())
    return fig


def fig_pinn_validacao(S_vals, tau_vals, V_pinn_dict, V_analitico_dict):
    cores = [AZUL, VERDE, ROXO, LARANJA]
    fig   = go.Figure()
    for (tau, V_p), (_, V_t), cor in zip(V_pinn_dict.items(),
                                           V_analitico_dict.items(), cores):
        er  = np.abs(np.array(V_p) - np.array(V_t)) / (np.array(V_t) + 1e-4) * 100
        fig.add_trace(go.Scatter(
            x=S_vals.tolist(), y=V_t if isinstance(V_t, list) else V_t.tolist(),
            mode="lines", line=dict(color=cor, width=2.8),
            name=f"Analítico τ={tau}"))
        fig.add_trace(go.Scatter(
            x=S_vals.tolist(), y=V_p if isinstance(V_p, list) else V_p.tolist(),
            mode="lines", line=dict(color=cor, width=1.6, dash="dash"),
            name=f"PINN τ={tau}"))
    fig.update_layout(
        title_text="<b>Figura 7</b> — PINN (problema direto) vs. Black-Scholes Analítico",
        xaxis=dict(title="S", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="V(S, τ)", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=440, **_base_layout())
    return fig


# ── 7. Posterior de σ ─────────────────────────────────────────────────────────
def fig_posterior_sigma(amostras, sigma_true, titulo="Posterior de σ — B-PINN"):
    kde    = gaussian_kde(amostras, bw_method='silverman')
    x_kde  = np.linspace(max(0.01, amostras.min()-0.06), amostras.max()+0.06, 500)
    y_kde  = kde(x_kde)
    ci_lo, ci_hi = np.percentile(amostras, [2.5, 97.5])
    ci_50_lo, ci_50_hi = np.percentile(amostras, [25, 75])
    media  = amostras.mean(); moda_idx = np.argmax(y_kde)

    fig = go.Figure()
    # Histograma
    fig.add_trace(go.Histogram(
        x=amostras.tolist(), nbinsx=70, histnorm="probability density",
        marker=dict(color=AZUL2, opacity=0.40, line=dict(color="white", width=0.3)),
        name="Amostras",
        hovertemplate="σ=%{x:.3f}<br>Dens.=%{y:.2f}<extra></extra>"))
    # IC 95% sombreado
    mask95 = (x_kde >= ci_lo) & (x_kde <= ci_hi)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_kde[mask95], x_kde[mask95][::-1]]).tolist(),
        y=np.concatenate([y_kde[mask95], np.zeros(mask95.sum())]).tolist(),
        fill="toself", fillcolor="rgba(38,70,83,0.22)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"IC 95%: [{ci_lo:.3f}, {ci_hi:.3f}]"))
    # IC 50% (HDI interno)
    mask50 = (x_kde >= ci_50_lo) & (x_kde <= ci_50_hi)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_kde[mask50], x_kde[mask50][::-1]]).tolist(),
        y=np.concatenate([y_kde[mask50], np.zeros(mask50.sum())]).tolist(),
        fill="toself", fillcolor="rgba(38,70,83,0.40)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 50%"))
    # KDE
    fig.add_trace(go.Scatter(
        x=x_kde.tolist(), y=y_kde.tolist(), mode="lines",
        line=dict(color=AZUL, width=3.0), name="KDE posterior"))
    # Linhas verticais
    fig.add_vline(x=float(sigma_true), line_color=LARANJA, line_width=3,
                  line_dash="dash",
                  annotation_text=f"<b>σ_true = {sigma_true}</b>",
                  annotation_position="top right",
                  annotation_font=dict(color=LARANJA, size=12))
    fig.add_vline(x=float(media), line_color=VERDE, line_width=2.5,
                  annotation_text=f"μ<sub>post</sub> = {media:.3f}",
                  annotation_position="top left",
                  annotation_font=dict(color=VERDE, size=11))
    fig.add_vline(x=float(x_kde[moda_idx]), line_color=ROXO, line_width=1.8,
                  line_dash="dot",
                  annotation_text=f"moda = {x_kde[moda_idx]:.3f}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=ROXO, size=10))
    fig.update_layout(
        title_text=f"<b>Figura 8b</b> — {titulo}",
        xaxis=dict(title="σ", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="Densidade", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=460, **_base_layout())
    return fig


def fig_bpinn_convergencia(history, sigma_true):
    ep  = list(range(len(history["sigma"])))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["(a) Trajetória de σ̂ durante o treino",
                        "(b) Componentes da perda (log)"],
        horizontal_spacing=0.10)

    # (a) convergência de sigma com banda de ruído
    sig_arr = np.array(history["sigma"])
    # Rolling std como proxy de ruído de amostragem
    win = max(1, len(sig_arr)//50)
    sig_smooth = np.convolve(sig_arr, np.ones(win)/win, mode='same')
    fig.add_trace(go.Scatter(x=ep, y=sig_arr.tolist(), mode="lines",
                             line=dict(color=AZUL2, width=0.8), opacity=0.5,
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=sig_smooth.tolist(), mode="lines",
                             line=dict(color=AZUL, width=2.5),
                             name="σ̂ (suavizado)"), row=1, col=1)
    fig.add_hline(y=sigma_true, line_dash="dash", line_color=LARANJA,
                  line_width=2.5,
                  annotation_text=f"σ_true = {sigma_true}", row=1, col=1)
    fig.add_annotation(
        x=ep[-1]*0.5, y=sigma_true+0.015,
        text=f"Erro final: {abs(sig_smooth[-1]-sigma_true):.4f}",
        showarrow=False, font=dict(size=10, color=CINZA),
        xref="x", yref="y")

    # (b) componentes de perda
    for key, nome, cor, dash in [
        ("total","Total",AZUL,None),
        ("l_data","Dados",LARANJA,None),
        ("l_res","Residual EDP",VERDE,None),
        ("kl","KL",ROXO,"dash")]:
        if key not in history: continue
        ld = dict(color=cor, width=1.8)
        if dash: ld["dash"] = dash
        fig.add_trace(go.Scatter(x=ep, y=history[key], mode="lines",
                                 line=ld, name=nome), row=1, col=2)

    fig.update_yaxes(type="log", row=1, col=2, title_text="Perda (log)")
    fig.update_xaxes(title_text="Épocas", row=1, col=1)
    fig.update_xaxes(title_text="Épocas", row=1, col=2)
    fig.update_yaxes(title_text="σ", row=1, col=1)
    fig.update_layout(height=440,
                      title_text="<b>Figura 8a</b> — Dinâmica de treinamento da B-PINN",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig


# ── 8. Experimentos de identificabilidade ────────────────────────────────────
def fig_identificabilidade_ruido(noise_vals, resultados, sigma_true):
    medias = [r["media"]          for r in resultados]
    ci_lo  = [r["ic_95"]["baixo"] for r in resultados]
    ci_hi  = [r["ic_95"]["alto"]  for r in resultados]
    vies   = [r.get("vies", 0)    for r in resultados]
    eta_pct = [e*100 for e in noise_vals]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["(a) Estimativa de σ̂ vs. Ruído",
                                        "(b) Viés absoluto vs. Ruído"],
                        horizontal_spacing=0.12)
    # Faixa de IC 95%
    fig.add_trace(go.Scatter(
        x=eta_pct + eta_pct[::-1],
        y=ci_hi + ci_lo[::-1],
        fill="toself", fillcolor="rgba(38,70,83,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 95%"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eta_pct, y=medias, mode="lines+markers",
        line=dict(color=AZUL, width=2.5),
        marker=dict(size=10, color=AZUL, line=dict(width=2, color="white")),
        name="σ̂ médio"), row=1, col=1)
    fig.add_hline(y=sigma_true, line_dash="dash", line_color=LARANJA,
                  line_width=2.5,
                  annotation_text=f"σ_true = {sigma_true}", row=1, col=1)

    # (b) viés com barras coloridas
    bar_colors = [VERDE if v < 0.01 else (AMARELO if v < 0.03 else VERMELHO) for v in vies]
    fig.add_trace(go.Bar(
        x=eta_pct, y=vies,
        marker=dict(color=bar_colors, line=dict(color="white", width=1)),
        name="Viés |σ̂ − σ_true|",
        hovertemplate="η=%{x:.0f}%<br>Viés=%{y:.4f}<extra></extra>"), row=1, col=2)
    # Linha de tolerância 1%
    fig.add_hline(y=0.01, line_dash="dot", line_color=VERDE, line_width=1.5,
                  annotation_text="tol. 1%", row=1, col=2)

    fig.update_xaxes(title_text="Nível de ruído η (%)", row=1, col=1)
    fig.update_xaxes(title_text="Nível de ruído η (%)", row=1, col=2)
    fig.update_yaxes(title_text="σ estimado", row=1, col=1)
    fig.update_yaxes(title_text="|σ̂ − σ_true|", row=1, col=2)
    fig.update_layout(height=440,
                      title_text="<b>Figura 9</b> — Identificabilidade de σ vs. Nível de Ruído",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig


def fig_mapa_identificabilidade(tabela):
    eta_lbs = [f"{e*100:.0f}%" for e in tabela["eta_grid"]]
    N_lbs   = [str(n) for n in tabela["N_grid"]]
    nivel   = tabela["nivel"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Viés |σ̂ − σ_true|",
                        f"Largura IC {nivel}%",
                        f"Cobertura IC {nivel}%"],
        horizontal_spacing=0.08)

    escalas  = ["RdYlGn_r", "YlOrRd", "RdYlGn"]
    datasets = [tabela["vies"], tabela["largura_ic"], tabela["cobertura"]]
    fmts     = [".4f", ".4f", ".2f"]
    for col_i, (data, cscale, fmt) in enumerate(zip(datasets, escalas, fmts), 1):
        txt = [[f"{v:{fmt}}" for v in row] for row in data]
        fig.add_trace(go.Heatmap(
            z=data.tolist(), x=N_lbs, y=eta_lbs,
            colorscale=cscale, showscale=True,
            text=txt, texttemplate="%{text}",
            textfont=dict(size=12, family="Times New Roman, serif"),
            colorbar=dict(x=0.01 + 0.345*(col_i-1), len=0.80, thickness=12),
            hovertemplate="N=%{x}<br>η=%{y}<br>val=%{z:.4f}<extra></extra>"),
            row=1, col=col_i)

    fig.update_layout(
        title_text=f"<b>Figura 11</b> — Mapa de Identificabilidade η × N",
        height=400, **_base_layout())
    return fig


def fig_posterior_multiplo(resultados_por_eta, sigma_true):
    fig = go.Figure()
    cores = [AZUL, VERDE, LARANJA, ROXO, VERMELHO]
    for (label, amostras), cor in zip(resultados_por_eta.items(), cores):
        kde   = gaussian_kde(amostras, bw_method='silverman')
        x_kde = np.linspace(max(0.01, amostras.min()-0.05),
                             amostras.max()+0.05, 500)
        y_kde = kde(x_kde)
        ci_lo, ci_hi = np.percentile(amostras, [2.5, 97.5])
        r_col = cor.lstrip("#")
        r,g,b = int(r_col[:2],16), int(r_col[2:4],16), int(r_col[4:],16)
        fig.add_trace(go.Scatter(
            x=x_kde.tolist(), y=y_kde.tolist(), mode="lines",
            line=dict(color=cor, width=2.5),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.10)",
            name=label))
    fig.add_vline(x=float(sigma_true), line_dash="dash", line_color="black",
                  line_width=3,
                  annotation_text=f"<b>σ_true = {sigma_true}</b>",
                  annotation_font=dict(size=12))
    fig.update_layout(
        title_text="<b>Figura 9b</b> — Posteriors de σ: efeito do nível de ruído",
        xaxis=dict(title="σ", **{k:v for k,v in _axis().items() if k != 'title'}),
        yaxis=dict(title="Densidade", **{k:v for k,v in _axis().items() if k != 'title'}),
        height=460, **_base_layout())
    return fig


def fig_comparacao_metodos(nr_samples, bpinn_samples, sigma_true):
    kde_b = gaussian_kde(bpinn_samples, bw_method='silverman')
    x_b   = np.linspace(bpinn_samples.min()-0.03, bpinn_samples.max()+0.03, 500)
    ci_lo, ci_hi = np.percentile(bpinn_samples, [2.5, 97.5])
    mean_nr = np.nanmean(nr_samples); mean_bp = bpinn_samples.mean()
    std_nr  = np.nanstd(nr_samples);  std_bp  = bpinn_samples.std()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["(a) Distribuições de σ estimado",
                                        "(b) Métricas comparativas"],
                        horizontal_spacing=0.12)
    # (a) comparação de distribuições
    fig.add_trace(go.Histogram(
        x=nr_samples.tolist(), nbinsx=35, histnorm="probability density",
        marker=dict(color=CINZA, opacity=0.55, line=dict(color="white", width=0.4)),
        name="IV pontual (NR)"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_b.tolist(), y=kde_b(x_b).tolist(), mode="lines",
        fill="tozeroy", fillcolor="rgba(38,70,83,0.20)",
        line=dict(color=AZUL, width=3.0), name="Posterior B-PINN"), row=1, col=1)
    fig.add_vline(x=float(sigma_true), line_color=LARANJA, line_width=3, line_dash="dash",
                  annotation_text=f"σ_true={sigma_true}",
                  annotation_position="top right", row=1, col=1)

    # (b) gráfico de comparação estilo dotplot
    metodos = ["Newton–Raphson", "Bayesian PINN"]
    medias_ = [mean_nr, mean_bp]
    stds_   = [std_nr,  std_bp]
    cores_  = [CINZA, AZUL]
    for j, (nome, med, std, cor) in enumerate(zip(metodos, medias_, stds_, cores_)):
        fig.add_trace(go.Scatter(
            x=[med], y=[j], mode="markers",
            marker=dict(color=cor, size=16, symbol="diamond"),
            name=nome, error_x=dict(type="data", array=[std],
                                     color=cor, thickness=3, width=12)), row=1, col=2)
    # IC 95% da B-PINN
    fig.add_trace(go.Scatter(
        x=[ci_lo, ci_hi], y=[1, 1], mode="lines",
        line=dict(color=AZUL, width=5), opacity=0.30,
        name="IC 95% B-PINN"), row=1, col=2)
    fig.add_vline(x=float(sigma_true), line_color=LARANJA, line_width=2.5, line_dash="dash",
                  annotation_text=f"σ_true", row=1, col=2)
    fig.update_yaxes(tickvals=[0,1], ticktext=metodos, row=1, col=2)
    fig.update_xaxes(title_text="σ estimado", row=1, col=2)

    fig.update_layout(height=460,
                      title_text="<b>Figura 10</b> — Newton–Raphson vs. B-PINN: estimação de volatilidade",
                      title_font=dict(size=15, family="Times New Roman, serif"),
                      **_base_layout())
    return fig
