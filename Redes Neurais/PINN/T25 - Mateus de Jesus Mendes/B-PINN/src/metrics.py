# -*- coding: utf-8 -*-
"""
src_SDE.metrics
===============
Métricas de avaliação do posterior Bayesiano para o problema inverso.

Funções
-------
resumo_posterior       : Estatísticas descritivas do posterior de σ
calibracao_ic          : Avalia calibração de intervalos de credibilidade
erro_relativo_precos   : RMSE/MAE entre preços preditos e observados
tabela_identificabilidade : Formata resultados de múltiplos experimentos
"""

import numpy as np
from scipy.stats import gaussian_kde
from typing import Sequence


def resumo_posterior(amostras: np.ndarray,
                     sigma_true: float | None = None,
                     niveis: Sequence[float] = (0.50, 0.80, 0.95)
                     ) -> dict:
    """
    Calcula estatísticas descritivas do posterior de σ.

    Parâmetros
    ----------
    amostras   : amostras do posterior p(σ|D), shape (K,)
    sigma_true : valor verdadeiro de σ (opcional — para métricas de acurácia)
    niveis     : níveis de credibilidade para intervalos de credibilidade

    Retorna
    -------
    dict com:
        media, mediana, desvio, moda_kde
        ic_{nivel} : dict com 'baixo' e 'alto'
        vies, cobertura_{nivel} (se sigma_true fornecido)
    """
    res = {
        "media":   float(np.mean(amostras)),
        "mediana": float(np.median(amostras)),
        "desvio":  float(np.std(amostras)),
        "min":     float(np.min(amostras)),
        "max":     float(np.max(amostras)),
    }

    # KDE para estimativa da moda
    try:
        kde   = gaussian_kde(amostras)
        x_kde = np.linspace(amostras.min(), amostras.max(), 500)
        res["moda_kde"] = float(x_kde[np.argmax(kde(x_kde))])
    except Exception:
        res["moda_kde"] = res["mediana"]

    # Intervalos de credibilidade
    for nv in niveis:
        alpha = (1.0 - nv) / 2.0
        lo    = float(np.percentile(amostras, 100 * alpha))
        hi    = float(np.percentile(amostras, 100 * (1.0 - alpha)))
        res[f"ic_{int(nv*100)}"] = {"baixo": lo, "alto": hi}

    # Métricas em relação ao valor verdadeiro
    if sigma_true is not None:
        res["sigma_true"] = float(sigma_true)
        res["vies"]       = float(abs(res["media"] - sigma_true))
        res["vies_rel"]   = float(res["vies"] / sigma_true)
        for nv in niveis:
            ic  = res[f"ic_{int(nv*100)}"]
            res[f"cobertura_{int(nv*100)}"] = float(
                ic["baixo"] <= sigma_true <= ic["alto"]
            )

    return res


def calibracao_ic(lista_resultados: list[dict],
                  nivel: int = 95) -> float:
    """
    Calcula a frequência de cobertura empírica do IC.

    Parâmetros
    ----------
    lista_resultados : lista de dicts produzidos por `resumo_posterior`
    nivel            : nível do IC (50, 80 ou 95)

    Retorna
    -------
    cobertura_empirica : fração de experimentos onde σ_true ∈ IC
    """
    key = f"cobertura_{nivel}"
    coberturas = [r[key] for r in lista_resultados if key in r]
    if not coberturas:
        return float("nan")
    return float(np.mean(coberturas))


def erro_relativo_precos(C_pred: np.ndarray,
                         C_obs: np.ndarray,
                         C_limpo: np.ndarray | None = None) -> dict:
    """
    Métricas de ajuste entre preços preditos e observados.

    Parâmetros
    ----------
    C_pred  : preços preditos pela rede
    C_obs   : preços observados (com ruído)
    C_limpo : preços exatos sem ruído (opcional)

    Retorna
    -------
    dict com RMSE, MAE, MAPE vs. observados (e vs. limpos se fornecido)
    """
    res_obs = {
        "rmse_obs": float(np.sqrt(np.mean((C_pred - C_obs) ** 2))),
        "mae_obs":  float(np.mean(np.abs(C_pred - C_obs))),
        "mape_obs": float(np.mean(np.abs((C_pred - C_obs) / (C_obs + 1e-6))) * 100),
    }
    if C_limpo is not None:
        res_obs.update({
            "rmse_limpo": float(np.sqrt(np.mean((C_pred - C_limpo) ** 2))),
            "mae_limpo":  float(np.mean(np.abs(C_pred - C_limpo))),
            "mape_limpo": float(
                np.mean(np.abs((C_pred - C_limpo) / (C_limpo + 1e-6))) * 100
            ),
        })
    return res_obs


def tabela_identificabilidade(eta_grid: Sequence[float],
                               N_grid: Sequence[int],
                               resultados: list[list[dict]],
                               nivel: int = 95) -> dict:
    """
    Formata resultados de experimentos em grade (η × N) para visualização.

    Parâmetros
    ----------
    eta_grid    : eixo de níveis de ruído
    N_grid      : eixo de tamanhos amostrais
    resultados  : lista[lista[dict]] — resultados[i][j] para (eta_i, N_j)
    nivel       : nível do IC

    Retorna
    -------
    dict com matrizes 'vies', 'largura_ic', 'cobertura' de shape (len(eta), len(N))
    """
    n_eta = len(eta_grid)
    n_N   = len(N_grid)
    vies_m    = np.zeros((n_eta, n_N))
    largura_m = np.zeros((n_eta, n_N))
    cob_m     = np.zeros((n_eta, n_N))

    for i in range(n_eta):
        for j in range(n_N):
            r  = resultados[i][j]
            ic = r.get(f"ic_{nivel}", {"baixo": 0.0, "alto": 0.0})
            vies_m[i, j]    = r.get("vies", float("nan"))
            largura_m[i, j] = ic["alto"] - ic["baixo"]
            cob_m[i, j]     = r.get(f"cobertura_{nivel}", float("nan"))

    return {
        "eta_grid":   list(eta_grid),
        "N_grid":     list(N_grid),
        "vies":       vies_m,
        "largura_ic": largura_m,
        "cobertura":  cob_m,
        "nivel":      nivel,
    }
