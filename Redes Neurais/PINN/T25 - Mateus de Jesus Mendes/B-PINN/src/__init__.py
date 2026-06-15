# -*- coding: utf-8 -*-
"""
src_SDE — Módulo auxiliar para o projeto
Bayesian PINNs aplicadas a SDEs / Black-Scholes.

Submódulos:
    stochastic  : Simulação de processos estocásticos (Wiener, GBM, EM, Milstein)
    black_scholes: Fórmula analítica, gregas e inversão de volatilidade
    numerics    : Solver Crank-Nicolson para a EDP de Black-Scholes
    data        : Geração de dados sintéticos e obtenção de dados reais
    plots       : Visualizações paper-like via Plotly
    metrics     : Métricas de avaliação do posterior Bayesiano
"""

from src import stochastic, black_scholes, numerics, data, plots, metrics

__all__ = ["stochastic", "black_scholes", "numerics", "data", "plots", "metrics"]
