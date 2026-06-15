# PINN — Physics-Informed Neural Networks

Implementação de Physics-Informed Neural Networks (PINNs) aplicadas ao pêndulo simples e, num segundo módulo, a Equações Diferenciais Estocásticas (SDEs) no contexto do modelo de Black-Scholes.

> **📌 Observação:** O projeto é composto por dois notebooks independentes. `PINN.ipynb` cobre o pêndulo simples; `sde_B-PINN.ipynb` cobre a extensão Bayesiana para SDEs.

---

## Objetivo

O projeto investiga o uso de PINNs para dois problemas distintos:

**Pêndulo Simples**

| Problema | Descrição |
|:--------:|-----------|
| Direto   | Resolver a EDO não-linear $\theta'' + (g/L)\sin\theta = 0$ sem dados observados |
| Inverso  | Descobrir o parâmetro $g$ a partir de observações sintéticas ruidosas |

**Black-Scholes (B-PINN)**

| Problema | Descrição |
|:--------:|-----------|
| Direto   | Resolver a EDP de Black-Scholes via PINN com posterior Bayesiano |
| Estocástico | Simulação de GBM, Euler-Maruyama e Milstein; precificação Monte Carlo |

---

## Arquitetura

A rede é um MLP configurável com as seguintes características:

- **Ativação Tanh**: suave e infinitamente diferenciável, essencial para resíduos de 2ª ordem.
- **Inicialização Xavier**: preserva a variância dos gradientes ao longo das camadas.
- **Otimização híbrida**: Adam para exploração inicial, L-BFGS para refinamento quasi-Newton.
- **Soft constraints**: condições iniciais e de contorno impostas via termos de penalidade na função de perda.

---

## Estrutura do Repositório

```
PINN/
│
├── PINN.ipynb                  # Notebook principal — pêndulo simples
│
├── src/
│   ├── __init__.py             # API pública do pacote
│   ├── model.py                # Arquitetura PINN (MLP configurável)
│   ├── physics.py              # EDO do pêndulo, resíduos, soluções analítica e numérica
│   ├── data.py                 # Geração de dados sintéticos e pontos de colocação
│   ├── training.py             # Loops Adam + L-BFGS (problema direto e inverso)
│   └── visualization.py        # Figuras interativas Plotly
│
└── B-PINN/
    ├── sde_B-PINN.ipynb        # Notebook secundário — PINNs + SDEs + Black-Scholes
    │
    └── src/
        ├── __init__.py
        ├── stochastic.py       # Simulação de Wiener, GBM, Euler-Maruyama, Milstein
        ├── black_scholes.py    # Fórmula analítica, gregas e volatilidade implícita
        ├── numerics.py         # Solver Crank-Nicolson para a EDP de Black-Scholes
        ├── data.py             # Dados sintéticos e dados reais via yfinance
        ├── metrics.py          # Métricas de avaliação do posterior Bayesiano
        └── plots.py            # Visualizações Plotly
```

---

## Instalação

```bash
git clone https://github.com/mateusjmd/PINN.git
cd PINN
pip install -r requirements.txt
```

---

## Uso

Abra o notebook de interesse diretamente com Jupyter:

```bash
# Problema do pêndulo simples
jupyter notebook PINN.ipynb

# Extensão Bayesiana com SDEs e Black-Scholes
jupyter notebook B-PINN/sde_B-PINN.ipynb
```

Os notebooks são autocontidos e executam as células em ordem sequencial.
