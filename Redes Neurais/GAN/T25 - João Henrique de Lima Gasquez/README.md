![ILUM, CNPEM, MINISTÉRIO DA EDUCAÇÃO](https://github.com/ComicDeath/Proton-Collision-Classifier/blob/main/Figuras/ilum_colorida.png)

<h1 align="center">A Trilha da Acadêmica - GANs</h1>

O projeto GANs foi desenvolvido como a segunda entrega da disciplina de Redes Neurais e Algoritmos Genéticos, ministrada no terceiro semestre do Bacharelado em Ciência e Tecnologia da Ilum – Escola de Ciências. O objetivo do trabalho é implementar e analisar modelos generativos adversariais (GANs) de forma didática, com foco na geração de imagens sintéticas a partir de ruído aleatório.

O desenvolvimento inclui a construção das arquiteturas do Gerador e do Discriminador, o treinamento adversarial entre as redes e o uso de datasets de imagens. Também foi realizada a análise da evolução das imagens geradas e observação da loss ao longo das épocas, permitindo avaliar o comportamento do aprendizado durante o treinamento e sua estabilidade ao longo do processo.

Foram utilizados recursos como TensorBoard para monitoramento do treinamento e técnicas de regularização para auxiliar na convergência do modelo. A implementação foi realizada em Python com PyTorch, incluindo organização do pipeline experimental e armazenamento dos resultados.

# Arquiteturas utilizadas

### VGAN (Vanilla Generative Adversarial Network)

Arquitetura básica de redes adversariais generativas composta por um Gerador e um Discriminador totalmente conectados. O treinamento é realizado de forma adversarial, onde o Gerador busca produzir amostras sintéticas a partir de ruído aleatório e o Discriminador aprende a distinguir dados reais de gerados.

### CGAN (Conditional Generative Adversarial Network)

Variante das GANs em que a geração de amostras é condicionada a informações adicionais, como rótulos de classe. Tanto o Gerador quanto o Discriminador recebem essa condição como entrada, permitindo controlar o tipo de dado gerado.

### DCGAN (Deep Convolutional Generative Adversarial Network)

Extensão das GANs que utiliza redes convolucionais profundas no Gerador e no Discriminador, substituindo camadas totalmente conectadas por convoluções.

# Instalação e como usar

Para utilizar o projeto, clone este repositório em sua máquina e abra-o em uma IDE compatível com arquivos `.ipynb`. Em seguida, abra o notebook `neuromante.ipynb`, que contém a introdução teórica, o pré-processamento de dados, o treino dos modelos e sua avaliação, a otimização de hiperparâmetros, a comparação com outros algoritmos, a análise da convergência com diferentes otimizadores, a explicação do modelo por meio da `Permutation Importance`, a discussão dos resultados e a conclusão.

É importante que o ambiente de execução possua todas as bibliotecas necessárias instaladas, incluindo **PyTorch**, **Lightning**, **Scikit**, **Optuna** e demais dependências utilizadas no notebook.  

A pasta `Estudos do Optuna` contém os arquivos `.db` gerados durante a otimização de hiperparâmetros. A pasta `lightning_logs` armazena os arquivos `.csv` referentes às curvas de aprendizado. Além disso, os melhores modelos treinados estão salvos na pasta `Melhores modelos`, com os arquivos `.pth` correspondentes aos melhores parâmetros treinados encontrados para os modelos que utilizam Adam, AdamW e SGD.

# Docente
A matéria de Redes Neurais e Algoritmos Genéticos foi ministrada por:
- **Profº Dr. Daniel Roberto Cassar**
  
# Licença
Distribuído sob a licença GNU General Public License 3.0, cheque `LICENSE` para mais informações.

# Referências para a construção do README
 ROCKETSEAT. **Como fazer um bom README.** Disponível em: https://blog.rocketseat.com.br/como-fazer-um-bom-readme/. Acesso em: 06 de abril de 2026.
