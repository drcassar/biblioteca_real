# -*- coding: utf-8 -*-
"""
src_SDE.stochastic
==================
Simulação de processos estocásticos de tempo contínuo.

Funções
-------
simular_wiener      : Trajetórias do Processo de Wiener
variacao_quadratica : Variação quadrática empírica
simular_gbm_exato   : GBM via solução exata (log-normal)
euler_maruyama      : Discretização de Euler-Maruyama
milstein            : Discretização de Milstein (ordem forte 1)
preco_mc            : Preço de opção por Monte Carlo
"""

import numpy as np


def simular_wiener(T: float = 1.0, N: int = 1000,
                   n_paths: int = 8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula n_paths trajetórias do Processo de Wiener em [0, T] com N passos.

    Parâmetros
    ----------
    T       : horizonte de tempo
    N       : número de passos de tempo
    n_paths : número de trajetórias
    seed    : semente para reprodutibilidade

    Retorna
    -------
    t : array de tempos (N+1,)
    W : matriz de trajetórias (n_paths, N+1)
    """
    rng = np.random.default_rng(seed)
    dt  = T / N
    dW  = rng.normal(0.0, np.sqrt(dt), size=(n_paths, N))
    W   = np.concatenate([np.zeros((n_paths, 1)),
                          np.cumsum(dW, axis=1)], axis=1)
    t   = np.linspace(0.0, T, N + 1)
    return t, W


def variacao_quadratica(W: np.ndarray) -> np.ndarray:
    """
    Calcula a variação quadrática empírica acumulada de uma trajetória 1-D.

    Parâmetros
    ----------
    W : trajetória 1-D (N+1,)

    Retorna
    -------
    vq : variação quadrática acumulada (N,)
    """
    return np.cumsum(np.diff(W) ** 2)


def simular_gbm_exato(S0: float, mu: float, sigma: float,
                      T: float, N: int, n_paths: int,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula GBM via integração exata (solução log-normal).

    S_t = S_0 * exp[(mu - sigma²/2)*t + sigma*W_t]

    Parâmetros
    ----------
    S0      : preço inicial
    mu      : drift
    sigma   : volatilidade
    T       : horizonte
    N       : passos de tempo
    n_paths : número de trajetórias
    seed    : semente

    Retorna
    -------
    t : array de tempos (N+1,)
    S : matriz de trajetórias (n_paths, N+1)
    """
    rng   = np.random.default_rng(seed)
    dt    = T / N
    Z     = rng.normal(0.0, 1.0, size=(n_paths, N))
    log_r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.concatenate([np.zeros((n_paths, 1)),
                             np.cumsum(log_r, axis=1)], axis=1)
    t     = np.linspace(0.0, T, N + 1)
    return t, S0 * np.exp(log_S)


def euler_maruyama(S0: float, mu: float, sigma: float,
                   T: float, N: int, n_paths: int,
                   seed: int = 42) -> np.ndarray:
    """
    Simula GBM via esquema de Euler-Maruyama (ordem forte 1/2).

    S_{n+1} = S_n * (1 + mu*dt + sigma*dW_n)

    Retorna
    -------
    S : matriz de trajetórias (n_paths, N+1)
    """
    rng = np.random.default_rng(seed)
    dt  = T / N
    S   = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.normal(0.0, np.sqrt(dt), n_paths)
        S[:, i + 1] = np.maximum(S[:, i] * (1.0 + mu * dt + sigma * dW), 0.0)
    return S


def milstein(S0: float, mu: float, sigma: float,
             T: float, N: int, n_paths: int,
             seed: int = 42) -> np.ndarray:
    """
    Simula GBM via esquema de Milstein (ordem forte 1).

    Adiciona a correção de Itô-Taylor:
    S_{n+1} = S_n + mu*S_n*dt + sigma*S_n*dW + (1/2)*sigma²*S_n*(dW² - dt)

    Retorna
    -------
    S : matriz de trajetórias (n_paths, N+1)
    """
    rng = np.random.default_rng(seed)
    dt  = T / N
    S   = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    for i in range(N):
        dW = rng.normal(0.0, np.sqrt(dt), n_paths)
        S[:, i + 1] = np.maximum(
            S[:, i]
            + mu * S[:, i] * dt
            + sigma * S[:, i] * dW
            + 0.5 * sigma ** 2 * S[:, i] * (dW ** 2 - dt),
            0.0,
        )
    return S


def preco_mc(S_T: np.ndarray, K: float, r: float, T: float,
             opcao: str = "call") -> float:
    """
    Preço de opção europeia por Monte Carlo (desconto risk-neutral).

    Parâmetros
    ----------
    S_T   : amostras do preço terminal S_T
    K     : strike
    r     : taxa livre de risco
    T     : maturidade
    opcao : 'call' ou 'put'

    Retorna
    -------
    preco : estimativa Monte Carlo do preço
    """
    if opcao == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    return float(np.mean(payoff) * np.exp(-r * T))
