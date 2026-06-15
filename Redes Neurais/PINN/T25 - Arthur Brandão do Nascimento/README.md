# **A Trilha Acadêmica - PINN para toxicidade de crioprotetores**

Este projeto aplica uma **Physics-Informed Neural Network (PINN)** ao problema inverso de estimação da constante cinética de toxicidade $k$ em soluções crioprotetoras (CPAs), comparando os resultados com os valores produzidos pelo modelo analítico reduzido de Warner et al. (2022).

## **Para utilizar o código**

1. **Pré-requisitos**:
   Certifique-se de ter o Python instalado e as seguintes bibliotecas:

   ```bash
   pip install numpy pandas matplotlib scikit-learn optuna torch lightning
   ```

2. **Dataset**:

   - Os dados experimentais de viabilidade celular provêm dos arquivos suplementares de Warner et al. (2022), disponibilizados como planilhas `.csv`.
   - Coloque os arquivos `Single-and-Binary-Data-Report.csv` e `Ternary-Data-Report.csv` na pasta `./dados`.
   - Execute primeiro o notebook `extracao_dados_xls.ipynb` para converter os arquivos para o formato adequado (`dados_simples_e_binarios.csv` e `dados_ternarios.csv`).
   - Em seguida, execute `calculo_k_artigo.ipynb` para calcular os valores de referência `resultados_k_artigo.csv`.

3. **Execução**:

   - Abra o arquivo `PINN.ipynb` em um ambiente Jupyter.
   - Execute as células em ordem para realizar o pré-processamento dos dados, a otimização de hiperparâmetros com o **Optuna** e o treinamento da PINN para cada composição.

## **Introdução**

A criopreservação depende de crioprotetores (CPAs) que, paradoxalmente, são tóxicos às próprias células que protegem. Quantificar essa toxicidade é um problema central no design de protocolos de criopreservação. Warner et al. (2022) descrevem o decaimento de viabilidade celular com um modelo cinético de primeira ordem:

$$\frac{dN}{dt} = -k \cdot N(t)$$

onde $N(t)$ é a viabilidade normalizada e $k$ é a constante de toxicidade, estimada por regressão não-linear para cada uma das 87 composições do dataset.

Este projeto substitui a regressão clássica por uma PINN que resolve o mesmo problema inverso — estimar $k$ a partir dos dados de viabilidade — impondo a equação diferencial diretamente na função de perda. Com apenas quatro pontos temporais por composição, a física do sistema atua como regularizador consistente com o fenômeno, reduzindo o espaço de soluções válidas.

## **Estrutura do repositório**

```
.
├── PINN.ipynb               # Notebook principal: PINN e comparação de resultados
├── extracao_dados_brutos.ipynb   # Conversão dos CSVs brutos para formato adequado
├── calculo_k_artigo.ipynb        # Cálculo dos k de referência pelo modelo de Warner et al.
├── dados/
│   ├── Single-and-Binary-Data-Report.csv
│   ├── Ternary-Data-Report.csv
│   ├── dados_simples_e_binarios.csv
│   └── dados_ternarios.csv
└── resultados_k_artigo.csv
```

## **Conclusão**

A PINN produziu menor RMSE em **71 das 87 composições avaliadas (81,6%)**, com RMSE médio de **0,072** contra **0,097** do modelo de referência. A vantagem foi mais pronunciada nas composições de toxicidade intermediária; nas composições de alta toxicidade (acima de 7 mol/L), ambos os modelos apresentam erros maiores, reflexo de um limite intrínseco ao dataset: quando a viabilidade colapsa antes do primeiro tempo experimental disponível (5 min), os quatro pontos temporais não restringem bem o valor de $k$.

## **Referências**

[1] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.

[2] Warner, R. M., et al. (2022). Rapid Quantification of Cryoprotectant Permeability via Numerical Optimization of Differential Scanning Calorimetry Thermograms. Cryobiology. DOI: 10.1016.

[3] CARVALHO, João Cláudio Nunes. O que é uma Physics-Informed Neural Network (PINN)? Medium, 9 ago. 2025. Disponível em: https://joaoclaudionc.medium.com/o-que-%C3%A9-uma-physics-informed-neural-network-pinn-3d0d466693f9. Acesso em: 11 maio. 2026.

[4] PRADHAN, R. The Math Behind Adam Optimizer. Towards Data Science, 2018. Disponível em: https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b/. Acesso em: 11 maio. 2026.

[5] MULTILAYER perceptron. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: https://en.wikipedia.org/wiki/Multilayer_perceptron. Acesso em: 11 maio. 2026.

[6] CASSAR, Daniel Roberto. Multilayer Perceptron em Python puro. [Jupyter Notebook], Ilum – Escola de Ciência, Campinas, 2026.

[7] CASSAR, Daniel Roberto. Construindo e treinando redes neurais com PyTorch e Lightning. [Jupyter Notebook], Ilum – Escola de Ciência, Campinas, 2026.
k.

[8] PYTORCH. MSELoss. PyTorch 2.11 Documentation, [s.d.]. Disponível em: https://docs.pytorch.org/docs/2.11/generated/torch.nn.MSELoss.html. Acesso em: 11 maio. 2026.

[9] PYTORCH. L1Loss. PyTorch 2.11 Documentation, [s.d.]. Disponível em: https://docs.pytorch.org/docs/2.11/generated/torch.nn.L1Loss.html. Acesso em: 11 maio. 2026.

[10] PYTORCH. HuberLoss. PyTorch 2.12 Documentation, [s.d.]. Disponível em: https://docs.pytorch.org/docs/2.12/generated/torch.nn.HuberLoss.html. Acesso em: 11 maio. 2026.

---

**Autor**: Arthur Brandão do Nascimento