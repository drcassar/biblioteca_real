# -*- coding: utf-8 -*-
"""
src_SDE.black_scholes
=====================
Fórmula analítica de Black-Scholes, gregas e inversão de volatilidade.

Funções
-------
bs_call         : Preço de call europeia (Black-Scholes)
bs_put          : Preço de put europeia (paridade put-call)
bs_vega         : Vega = ∂C/∂σ
bs_delta        : Delta = ∂C/∂S
bs_gamma        : Gamma = ∂²C/∂S²
iv_newton       : Volatilidade implícita via Newton-Raphson
smile_sintetico : Smile de volatilidade paramétrico (skew + curvatura)
"""

import numpy as np
from scipy import stats


# ── Fórmula analítica ─────────────────────────────────────────────────────────

def _d1d2(S: np.ndarray, K: float, r: float,
          sigma: float, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula d1 e d2 da fórmula de Black-Scholes."""
    tau  = np.maximum(tau, 1e-10)
    d1   = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2   = d1 - sigma * np.sqrt(tau)
    return d1, d2


def bs_call(S: np.ndarray, K: float, r: float,
            sigma: float, tau: np.ndarray) -> np.ndarray:
    """
    Preço de call europeia via fórmula de Black-Scholes.

    C = S·Φ(d1) − K·e^{−rτ}·Φ(d2)

    Parâmetros
    ----------
    S     : preço do ativo (escalar ou array)
    K     : strike
    r     : taxa livre de risco (contínua)
    sigma : volatilidade
    tau   : tempo até o vencimento τ = T − t

    Retorna
    -------
    C : preço da call
    """
    d1, d2 = _d1d2(S, K, r, sigma, tau)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)


def bs_put(S: np.ndarray, K: float, r: float,
           sigma: float, tau: np.ndarray) -> np.ndarray:
    """Preço de put europeia via paridade put-call: P = C − S + K·e^{−rτ}."""
    return bs_call(S, K, r, sigma, tau) - S + K * np.exp(-r * np.maximum(tau, 1e-10))


def bs_vega(S: np.ndarray, K: float, r: float,
            sigma: float, tau: np.ndarray) -> np.ndarray:
    """
    Vega: sensibilidade do preço à volatilidade.

    𝒱 = S·φ(d1)·√τ > 0  (sempre positivo — garante identificabilidade local)
    """
    tau  = np.maximum(tau, 1e-10)
    d1, _ = _d1d2(S, K, r, sigma, tau)
    return S * stats.norm.pdf(d1) * np.sqrt(tau)


def bs_delta(S: np.ndarray, K: float, r: float,
             sigma: float, tau: np.ndarray) -> np.ndarray:
    """Delta: ∂C/∂S = Φ(d1)."""
    d1, _ = _d1d2(S, K, r, sigma, tau)
    return stats.norm.cdf(d1)


def bs_gamma(S: np.ndarray, K: float, r: float,
             sigma: float, tau: np.ndarray) -> np.ndarray:
    """Gamma: ∂²C/∂S² = φ(d1) / (S·σ·√τ)."""
    tau  = np.maximum(tau, 1e-10)
    d1, _ = _d1d2(S, K, r, sigma, tau)
    return stats.norm.pdf(d1) / (S * sigma * np.sqrt(tau))


# ── Inversão de volatilidade ──────────────────────────────────────────────────

def iv_newton(C_mkt: float, S: float, K: float, r: float, tau: float,
              sigma0: float = 0.20, tol: float = 1e-8,
              max_iter: int = 200) -> float:
    """
    Volatilidade implícita via Newton-Raphson.

    Resolve:  C_BS(σ*) = C_mkt
    Via:      σ_{n+1} = σ_n − [C_BS(σ_n) − C_mkt] / 𝒱(σ_n)

    Parâmetros
    ----------
    C_mkt   : preço de mercado da opção
    S, K    : spot e strike
    r       : taxa livre de risco
    tau     : tempo até o vencimento
    sigma0  : estimativa inicial (default 20%)
    tol     : tolerância de convergência
    max_iter: máximo de iterações

    Retorna
    -------
    sigma : volatilidade implícita, ou np.nan se não convergir
    """
    if tau <= 0 or C_mkt <= 0:
        return np.nan
    sigma = sigma0
    for _ in range(max_iter):
        C_bs = bs_call(S, K, r, sigma, tau)
        V    = bs_vega(S, K, r, sigma, tau)
        if abs(V) < 1e-12:
            return np.nan
        sigma -= (C_bs - C_mkt) / V
        sigma  = max(sigma, 1e-6)
        if abs(C_bs - C_mkt) < tol:
            return float(sigma)
    return np.nan


def smile_sintetico(S0: float, K_grid: np.ndarray, tau: float,
                    r: float = 0.05, sigma_atm: float = 0.20,
                    skew: float = -0.30, curvatura: float = 0.80,
                    seed: int = 42, eta: float = 0.0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera um smile de volatilidade paramétrico com skew e curvatura.

    σ(K) = σ_ATM + skew·ln(K/S0) + curvatura·[ln(K/S0)]²

    Parâmetros
    ----------
    S0         : spot
    K_grid     : array de strikes
    tau        : maturidade
    r          : taxa livre de risco
    sigma_atm  : volatilidade at-the-money
    skew       : inclinação (negativo → smirk típico de equities)
    curvatura  : convexidade do smile
    seed       : semente para ruído
    eta        : nível de ruído relativo nos preços (0 = sem ruído)

    Retorna
    -------
    sigma_K : superfície de vol por strike
    C_limpo : preços sem ruído
    C_obs   : preços observados (com ruído)
    """
    rng = np.random.default_rng(seed)
    mn  = np.log(K_grid / S0)
    sigma_K = np.clip(sigma_atm + skew * mn + curvatura * mn ** 2, 0.05, 1.5)
    S_arr   = np.full_like(K_grid, S0)
    C_limpo = bs_call(S_arr, K_grid, r, sigma_K, tau)
    noise   = eta * C_limpo * rng.standard_normal(len(K_grid))
    C_obs   = np.maximum(C_limpo + noise, 1e-4)
    return sigma_K, C_limpo, C_obs
