# -*- coding: utf-8 -*-
"""
src_SDE.data
============
Geração de dados sintéticos e obtenção de dados reais de mercado.

Funções
-------
gerar_dados_opcoes  : Dataset sintético de opções com ruído controlado
obter_dados_mercado : Dados reais via yfinance (com fallback sintético)
amostrar_colocation : Pontos de colocação para PINN (retorna tensores PyTorch)
"""

import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
from .black_scholes import bs_call


def gerar_dados_opcoes(sigma_true: float, N: int = 80,
                       K_range: tuple = (80.0, 120.0),
                       tau_range: tuple = (0.1, 1.5),
                       S0: float = 100.0, r: float = 0.05,
                       eta: float = 0.02,
                       seed: int = 42) -> dict:
    """
    Gera um dataset sintético de preços de call europeia.

    Protocolo:
        1. Amostrar strikes K e maturidades τ aleatoriamente.
        2. Calcular preços exatos via fórmula de Black-Scholes.
        3. Adicionar ruído gaussiano relativo: C_obs = C_limpo·(1 + η·ε), ε~N(0,1).

    Parâmetros
    ----------
    sigma_true : volatilidade verdadeira (parâmetro a ser recuperado)
    N          : número de observações
    K_range    : intervalo de strikes (K_min, K_max)
    tau_range  : intervalo de maturidades (τ_min, τ_max)
    S0         : preço spot do ativo
    r          : taxa livre de risco
    eta        : desvio padrão relativo do ruído (SNR = 1/eta)
    seed       : semente para reprodutibilidade

    Retorna
    -------
    dict com chaves:
        S, K, tau    : arrays de entrada
        C_limpo      : preços exatos (sem ruído)
        C_obs        : preços observados (com ruído)
        sigma_true   : volatilidade verdadeira
        eta          : nível de ruído
    """
    rng   = np.random.default_rng(seed)
    K_obs = rng.uniform(*K_range, N)
    tau_o = rng.uniform(*tau_range, N)
    S_obs = np.full(N, S0)

    C_limpo = bs_call(S_obs, K_obs, r, sigma_true, tau_o)
    ruido   = eta * C_limpo * rng.standard_normal(N)
    C_obs   = np.maximum(C_limpo + ruido, 1e-4)

    return {
        "S":          S_obs,
        "K":          K_obs,
        "tau":        tau_o,
        "C_limpo":    C_limpo,
        "C_obs":      C_obs,
        "sigma_true": sigma_true,
        "eta":        eta,
    }


def obter_dados_mercado(ticker: str = "SPY",
                        dias_alvo: int = 60,
                        min_volume: int = 10,
                        moneyness_range: tuple = (0.70, 1.30)) -> dict | None:
    """
    Obtém dados reais de opções (calls) via yfinance.

    Seleciona a maturidade mais próxima de `dias_alvo` dias.
    Filtra contratos por volume mínimo e intervalo de moneyness.

    Parâmetros
    ----------
    ticker          : símbolo do ativo (ex: 'SPY', '^SPX')
    dias_alvo       : maturidade-alvo em dias corridos
    min_volume      : volume mínimo para inclusão do contrato
    moneyness_range : (K_min/S, K_max/S) — controla a região de strikes

    Retorna
    -------
    dict com 'S', 'K', 'tau', 'C_obs', 'S0', 'tau_val', 'ticker',
    ou None se os dados não estiverem disponíveis.
    """
    try:
        ativo  = yf.Ticker(ticker)
        S_mkt  = float(ativo.fast_info["last_price"])
        expiry = ativo.options
        if not expiry:
            return None

        hoje   = datetime.now()
        target = hoje + timedelta(days=dias_alvo)
        exp_dt = [datetime.strptime(e, "%Y-%m-%d") for e in expiry]
        best   = min(exp_dt, key=lambda d: abs((d - target).days))
        exp_s  = best.strftime("%Y-%m-%d")
        tau_v  = max((best - hoje).days / 365.25, 1.0 / 365.0)

        chain = ativo.option_chain(exp_s).calls
        chain = chain[chain["volume"] > min_volume].copy()
        lo, hi = moneyness_range
        chain = chain[(chain["strike"] > lo * S_mkt)
                      & (chain["strike"] < hi * S_mkt)]
        chain = chain.dropna(subset=["lastPrice"]).head(100)

        if len(chain) < 5:
            return None

        K_m   = chain["strike"].values.astype(float)
        C_m   = chain["lastPrice"].values.astype(float)
        S_arr = np.full(len(K_m), S_mkt)
        tau_a = np.full(len(K_m), tau_v)

        print(f"  [{ticker}] S={S_mkt:.2f} | exp={exp_s} "
              f"| τ={tau_v:.3f} ano | N={len(K_m)} contratos")
        return {
            "S":       S_arr,
            "K":       K_m,
            "tau":     tau_a,
            "C_obs":   C_m,
            "S0":      S_mkt,
            "tau_val": tau_v,
            "ticker":  ticker,
        }

    except Exception as e:
        print(f"  [yfinance] erro: {e}")
        return None


def _smile_mercado_tipico(S0: float = 490.0, tau: float = 60.0 / 365.0,
                          r: float = 0.05, seed: int = 7) -> dict:
    """
    Gera dados sintéticos calibrados com estrutura de smile típica de equities.
    Usado como fallback quando yfinance não está disponível.
    """
    K_s = np.linspace(int(S0 * 0.85), int(S0 * 1.15), 20).astype(float)
    mn  = np.log(K_s / S0)
    # Smile com skew negativo (típico de índices de ações)
    sig = 0.18 - 0.25 * mn + 0.50 * mn ** 2
    sig = np.clip(sig, 0.05, 0.80)
    S_s = np.full(len(K_s), S0)
    C_s = bs_call(S_s, K_s, r, sig, tau)
    rng = np.random.default_rng(seed)
    C_s = np.maximum(C_s + 0.03 * C_s * rng.standard_normal(len(C_s)), 0.01)

    return {
        "S":       S_s,
        "K":       K_s,
        "tau":     np.full(len(K_s), tau),
        "C_obs":   C_s,
        "S0":      S0,
        "tau_val": tau,
        "ticker":  "SPY (sintético calibrado)",
    }


def amostrar_colocation(N_r: int, N_bc: int, N_ic: int,
                        K: float, S_max: float, T: float,
                        r: float, device, seed: int = 0) -> dict:
    """
    Gera pontos de colocação para o treinamento da PINN.

    Inclui:
        - Pontos interiores (residual da EDP)
        - Condição terminal τ = 0: V(S, T) = max(S − K, 0)
        - Condição de contorno S = 0: V = 0
        - Condição de contorno S = S_max: V ≈ S_max − K·e^{−rτ}

    Parâmetros
    ----------
    N_r, N_bc, N_ic : número de pontos de cada tipo
    K, S_max, T, r  : parâmetros do modelo
    device          : dispositivo PyTorch (cpu/cuda)
    seed            : semente

    Retorna
    -------
    dict com tensores PyTorch prontos para uso no treinamento
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def rand(n, lo=0.0, hi=1.0):
        return torch.rand(n, 1, generator=g) * (hi - lo) + lo

    # Pontos interiores (colocação)
    S_r   = rand(N_r, 0.025 * S_max, 0.975 * S_max)
    tau_r = rand(N_r, 0.0, T)

    # Condição terminal: V(S, 0) = payoff
    S_ic  = rand(N_ic, 0.0, S_max)
    V_ic  = torch.clamp(S_ic - K, min=0.0)

    # BC inferior: V(0, τ) = 0
    tau_bc0 = rand(N_bc // 2, 0.0, T)

    # BC superior: V(S_max, τ) ≈ S_max − K·e^{−rτ}
    tau_bcS = rand(N_bc // 2, 0.0, T)
    V_bcS   = S_max - K * torch.exp(-r * tau_bcS)

    def to(x):
        return x.to(device)

    return {
        "S_r":    to(S_r),    "tau_r":   to(tau_r),
        "S_ic":   to(S_ic),   "tau_ic":  to(torch.zeros(N_ic, 1)),   "V_ic": to(V_ic),
        "S_bc0":  to(torch.zeros(N_bc // 2, 1)),
        "tau_bc0":to(tau_bc0), "V_bc0":  to(torch.zeros(N_bc // 2, 1)),
        "S_bcS":  to(torch.full((N_bc // 2, 1), S_max)),
        "tau_bcS":to(tau_bcS), "V_bcS":  to(V_bcS),
    }
