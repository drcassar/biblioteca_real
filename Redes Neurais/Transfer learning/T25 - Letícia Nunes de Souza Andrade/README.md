# Transfer Learning — Classificação de Células Sanguíneas

O presente repositório destina-se à abordagem didática do conceito de Transfer Learning, via PyTorch, por meio duas etapas:

1. **Exemplo introdutório** — Classificação binária (avião vs. automóvel) no CIFAR-10 usando MobileNetV2 pré-treinado no ImageNet.
2. **Aplicação biológica** — Classificação de subtipos de leucócitos a partir de imagens de microscopia, com apenas ~347 imagens rotuladas.

## Estrutura

```
Transfer-Learning/
├── transfer_learning.ipynb   # Notebook principal
└── dataset-master/
    ├── JPEGImages/           # Imagens de células sanguíneas (.jpg)
    ├── Annotations/          # Anotações Pascal VOC (.xml)
    └── labels.csv            # Rótulos: NEUTROPHIL, EOSINOPHIL, LYMPHOCYTE, MONOCYTE, BASOPHIL
```

## Requisitos

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow
```

> GPU recomendada, mas o notebook detecta automaticamente CPU/CUDA via `torch.device`.

## Como executar

```bash
git clone https://github.com/LeticiaNunesAndrade/Transfer-Learning.git
cd Transfer-Learning
jupyter notebook transfer_learning.ipynb
```

Execute as células em ordem. O notebook está dividido nas seguintes seções:

| Seção | Conteúdo |
|---|---|
| 1–4 | Fundamentos teóricos de Transfer Learning |
| 5 | Feature Extraction com MobileNetV2 no CIFAR-10 |
| 6 | Fine-tuning aplicado à classificação de leucócitos |

## Dataset de células sanguíneas

- **Fonte:** [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- **Imagens:** 640×480 px, formato JPEG
- **Classes:** NEUTROPHIL (207), EOSINOPHIL (87), LYMPHOCYTE (33), MONOCYTE (21), BASOPHIL (3)
- **Anotações:** Pascal VOC (bounding boxes em XML)

O `labels.csv` mapeia cada imagem à sua classe e é a entrada usada pelo notebook para carregar e dividir os dados.
