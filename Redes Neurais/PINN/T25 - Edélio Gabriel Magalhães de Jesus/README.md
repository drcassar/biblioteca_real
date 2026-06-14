# 🌌 Uma Jornada pelas Physics-Informed Neural Networks

> *"O que os dados não forem capazes de ensinar à rede, a Física ensina."*

Este repositório é um material didático progressivo sobre **Physics-Informed Neural Networks (PINNs)** — uma abordagem que combina redes neurais com equações diferenciais para resolver problemas físicos com pouca ou nenhuma dependência de dados rotulados.

A jornada foi pensada para estudantes de ciência e tecnologia que já têm familiaridade com Python e noções básicas de redes neurais, mas querem entender PINNs.

---

## 🗺️ Estrutura da Jornada

```
pinns-journey/
│
├── notebooks/                        
│   ├── 00_introduction.ipynb          
│   ├── 01_direct_stationary_vanilla.ipynb
│   ├── 02_direct_stationary_hard.ipynb
│   ├── 03_direct_transient.ipynb
│   ├── 04_inverse_stationary.ipynb
│   ├── 05_inverse_transient.ipynb
│   ├── 06_architectures.ipynb
│   └── 07_sampling.ipynb
│
├── scripts/                            # Código de suporte
│   ├── geral_functions.py              # Funções compartilhadas
│   ├── plot_utils.py                   # Visualizações
│   ├── arquitectures.py                # Arquitetura diferente de MLP
│   ├── ex01_pinn_direct_stationary_vanilla.py
│   ├── ex02_pinn_direct_stationary_hard.py
│   ├── ex03_pinn_direct_transient.py
│   ├── ex04_pinn_inverse_stationary.py
│   ├── ex05_pinn_inverse_transient.py
│   └── sampling_helmholtz.py
│
├── requirements.txt
└── README.md
```

---

## 📖 Os Notebooks

### `00` — Introdução às PINNs
O ponto de partida. Discutimos o que são PINNs, por que elas existem e como a física entra na função de perda. Apresentamos a formulação clássica de Raissi et al. (2019) e a interpretação de Baty (2024), introduzimos a distinção entre *vanilla-PINNs* e *hard-PINNs*, e explicamos a diferença entre problemas diretos e inversos.

**Conceitos introduzidos:** função de perda física, pontos de colocação, training points, vanilla vs hard-PINNs, problemas diretos e inversos.

---

### `01` — Problema Direto Estacionário (Vanilla-PINN)
**Equação de Laplace 2D** — distribuição de potencial em um domínio retangular.

O primeiro exemplo concreto. A rede aprende a solução a partir apenas das condições de contorno, sem nenhum dado interno. Introduzimos os pontos de colocação, a função de perda física e a validação pela solução analítica.

```
∇²u = 0,   (x,y) ∈ [0,1]²
u(x,0) = 0,   u(x,1) = sin(πx),   u(0,y) = u(1,y) = 0
```

---

### `02` — Problema Direto Estacionário (Hard-PINN) ⭐ *plus*
**Equação de Laplace 2D** — o mesmo problema, com uma abordagem diferente.

As condições de contorno são satisfeitas **por construção** via função de tentativa, eliminando completamente o termo $L_{\text{data}}$. Uma comparação direta com o notebook anterior revela as vantagens e limitações de cada abordagem.

```
u_θ(x,y) = A(x,y) + B(x,y) · N_θ(x,y)
```

**Novidades:** hard-PINN, função de tentativa, comparação soft vs hard constraints.

---

### `03` — Problema Direto Transiente
**Equação de Burgers 1D** — o benchmark clássico da literatura de PINNs.

A dimensão temporal entra pela primeira vez. A solução desenvolve um choque em $x=0$ para viscosidades pequenas — um dos problemas mais desafiadores para métodos numéricos clássicos. Introduzimos as condições iniciais e discutimos a solução numérica de referência via método das linhas.

```
u_t + u·u_x = ν·u_xx,   x ∈ [-1,1],   t ∈ [0,1]
u(x,0) = -sin(πx),   ν = 0.01/π
```

**Novidades:** domínio espaço-temporal, condições iniciais, problema transiente

---

### `04` — Problema Inverso Estacionário
**Equação de Poisson-Boltzmann planar** — dupla camada elétrica em superfícies carregadas.

O paradigma se inverte: em vez de calcular a solução a partir dos parâmetros, recuperamos um parâmetro desconhecido a partir de medições esparsas e ruidosas. O potencial de superfície $\tilde{\psi}_0$ é tratado como `nn.Parameter` e otimizado junto com os pesos da rede.

```
d²ψ̃/dx̃² = sinh(ψ̃),   x̃ ∈ [0, 5]
ψ̃(0) = ψ̃₀ (desconhecido),   ψ̃(5) ≈ 0
```

**Novidades:** problema inverso, `nn.Parameter`, dados sintéticos ruidosos, solução de Gouy-Chapman, erro percentual no parâmetro recuperado.

---

### `05` — Problema Inverso Transiente
**Equação de difusão 2D** — recuperação do coeficiente de difusão.

O problema inverso ganha a dimensão temporal. A partir de medições esparsas do campo de concentração $c(x,y,t)$, a PINN recupera o coeficiente de difusão $D$ — diretamente ligado à caracterização de biomoléculas em solução. Os dados de treinamento vêm de uma solução numérica, simulando o que seria observado experimentalmente.

```
∂c/∂t = D(∂²c/∂x² + ∂²c/∂y²),   (x,y) ∈ [0,1]²,   t ∈ [0,1]
c(x,y,0) = exp(-(‖r-r₀‖²)/(2σ²))
```

**Novidades:** problema inverso transiente, dados numéricos como treinamento

---

### `06` — Arquiteturas Avançadas
**Equação de Helmholtz 2D** — arcada magnética solar.

Comparamos a MLP vanilla com uma **ResNet-PINN** em um problema oscilatório que exige profundidade de rede. As skip connections da ResNet facilitam o fluxo do gradiente em redes profundas, o que se traduz em convergência mais rápida para soluções com múltiplos nós.

```
∇²u + c²u = 0,   (x,z) ∈ [-1.5, 1.5] × [0, 3]
```

**Novidades:** ResNet, skip connections, vanishing gradient, comparação de arquiteturas.

---

### `07` — Estratégias de Amostragem
**Equação de Helmholtz 2D** — o mesmo problema, com foco diferente.

A escolha de como amostrar os pontos de colocação afeta diretamente a convergência. Comparamos três estratégias — uniforme, aleatória e **Latin Hypercube Sampling (LHS)** — e discutimos quando cada uma é mais adequada. Mencionamos a amostragem adaptativa como direção futura.

**Novidades:** amostragem uniforme, aleatória e LHS, cobertura do domínio, densidade de pontos.

---

## 📦 Instalação

```bash
git clone https://github.com/EdelioGabriel/REDES_NEURAIS.git
cd REDES_NEURAIS/PINN_DIDATICA
pip install -r requirements.txt
```

**Requisitos principais:**
```
torch >= 2.0
numpy
scipy
plotly
jupyter
```

---

## 🧭 Como navegar

Os notebooks foram pensados para serem lidos **em ordem** — cada um introduz conceitos que os seguintes assumem conhecidos. Dito isso, a partir do `03` os notebooks são relativamente independentes do ponto de vista do código.

Se você já conhece PINNs e quer ir direto a um tópico específico:

| Quero aprender sobre... | Vá para... |
|---|---|
| O que são PINNs | `00` |
| Minha primeira PINN | `01` |
| Condições de contorno exatas | `02` |
| Problemas com choque | `03` |
| Recuperar parâmetros de EDPs | `04` e `05` |
| Redes mais profundas | `06` |
| Como amostrar melhor | `07` |

---

## 📚 Referências principais

| Referência | Relevância |
|---|---|
| Raissi et al. (2019), *J. Comput. Phys.* | Formulação original das PINNs |
| Baty (2024), *arXiv:2403.00599* | Referência didática principal |

---

## ✍️ Sobre o material

Este material foi desenvolvido como repositório didático. O foco está em **entender PINNs de verdade** — não apenas usar uma biblioteca, mas compreender cada escolha de formulação, arquitetura e treinamento.

O código foi escrito para ser legível e progressivo — as funções dos exemplos simples aparecem explicitamente nos notebooks; as mais repetitivas ficam nos scripts de suporte. Os plots foram feitos com Plotly para interatividade.
