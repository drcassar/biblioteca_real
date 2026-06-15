# -*- coding: utf-8 -*-
"""
src_SDE.numerics
================
Métodos numéricos para a EDP de Black-Scholes.

Funções
-------
crank_nicolson_bs : Solver Crank-Nicolson para call europeia
convergencia_cn   : Análise de convergência do método CN
"""

import numpy as np
from scipy.linalg import solve_banded
from .black_scholes import bs_call


def crank_nicolson_bs(S_max: float, K: float, r: float, sigma: float,
                      T: float, N_S: int = 300,
                      N_t: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a EDP de Black-Scholes para uma call europeia via Crank-Nicolson.

    EDP (em τ = T − t):
        ∂V/∂τ = (1/2)σ²S²∂²V/∂S² + rS·∂V/∂S − rV

    Condições:
        V(S, τ=0) = max(S − K, 0)   [condição terminal / payoff]
        V(0, τ)   = 0
        V(S_max, τ) ≈ S_max − K·e^{−rτ}

    Estabilidade: incondicional (critério de von Neumann).
    Ordem de acurácia: O(Δτ², ΔS²).

    Parâmetros
    ----------
    S_max : domínio superior em S
    K     : strike
    r     : taxa livre de risco
    sigma : volatilidade
    T     : maturidade
    N_S   : pontos na grade de S
    N_t   : passos de tempo

    Retorna
    -------
    S_grid  : grade de preços do ativo (N_S+1,)
    t_grid  : grade de tempo (N_t+1,)  — τ = T − t_grid
    V_grid  : preços da opção V(S, τ)  — shape (N_t+1, N_S+1)
              V_grid[0] corresponde a t=0 (τ=T)
              V_grid[-1] corresponde a t=T (τ=0, payoff)
    """
    S    = np.linspace(0.0, S_max, N_S + 1)
    t    = np.linspace(0.0, T, N_t + 1)          # t ∈ [0, T]
    dt   = T / N_t

    # Condição terminal (τ = 0): V = payoff
    V     = np.maximum(S - K, 0.0)
    V_all = np.zeros((N_t + 1, N_S + 1))
    V_all[-1] = V.copy()

    idx  = np.arange(1, N_S)          # nós interiores

    # Coeficientes do esquema CN
    alpha = 0.25 * dt * (sigma ** 2 * idx ** 2 - r * idx)
    beta  = -0.5 * dt * (sigma ** 2 * idx ** 2 + r)
    gamma = 0.25 * dt * (sigma ** 2 * idx ** 2 + r * idx)

    # Diagonais das matrizes A (implícita) e B (explícita)
    lo_A = -alpha[1:]
    di_A =  1.0 - beta
    up_A = -gamma[:-1]

    lo_B =  alpha[1:]
    di_B =  1.0 + beta
    up_B =  gamma[:-1]

    def matvec(lo, di, up, v):
        """Multiplica matriz tridiagonal por vetor."""
        r_      = di * v
        r_[:-1] += up * v[1:]
        r_[1:]  += lo * v[:-1]
        return r_

    # Marcha retrógrada: de τ=0 (t=T) até τ=T (t=0)
    for j in range(N_t, 0, -1):
        V_lo = 0.0
        V_hi = S_max - K * np.exp(-r * t[j - 1])   # BC assintótica

        rhs = matvec(lo_B, di_B, up_B, V[1:-1])

        # Contribuições das condições de contorno
        rhs[0]  += alpha[0]  * (V[0]  + V_lo)
        rhs[-1] += gamma[-1] * (V[-1] + V_hi)

        # Sistema tridiagonal A·V^{j+1} = rhs
        ab = np.zeros((3, N_S - 1))
        ab[0, 1:]  = up_A
        ab[1, :]   = di_A
        ab[2, :-1] = lo_A

        V[1:-1] = solve_banded((1, 1), ab, rhs)
        V[0]    = V_lo
        V[-1]   = V_hi

        V_all[j - 1] = V.copy()

    return S, t, V_all


def convergencia_cn(S0: float, K: float, r: float, sigma: float, T: float,
                    N_vals: list[int] | None = None,
                    n_paths_mc: int = 5000,
                    seed: int = 42) -> dict:
    """
    Analisa a convergência do método CN vs. preço analítico.

    Parâmetros
    ----------
    S0, K, r, sigma, T : parâmetros do modelo
    N_vals             : lista de N_t a testar (passos de tempo)
    n_paths_mc         : não utilizado (reservado)
    seed               : semente

    Retorna
    -------
    dict com 'N_vals', 'dts', 'erros_cn' e 'preco_analitico'
    """

    if N_vals is None:
        N_vals = [2 ** k for k in range(4, 10)]

    C_ref = bs_call(S0, K, r, sigma, T)
    erros = []
    for N_t in N_vals:
        _, _, V_grid = crank_nicolson_bs(3 * S0, K, r, sigma, T, N_S=200, N_t=N_t)
        C_cn = float(np.interp(S0, np.linspace(0, 3 * S0, 201), V_grid[0]))
        erros.append(abs(C_cn - C_ref))

    return {
        "N_vals":          N_vals,
        "dts":             [T / n for n in N_vals],
        "erros_cn":        erros,
        "preco_analitico": float(C_ref),
    }
