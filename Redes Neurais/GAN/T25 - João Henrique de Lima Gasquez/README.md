![ILUM, CNPEM, MINISTÉRIO DA EDUCAÇÃO](https://github.com/ComicDeath/Proton-Collision-Classifier/blob/main/Figuras/ilum_colorida.png)

<h1 align="center">A Trilha da Acadêmica - GANs</h1>

O projeto "GANs" foi desenvolvido como a segunda entrega da disciplina de Redes Neurais e Algoritmos Genéticos, ministrada no terceiro semestre do Bacharelado em Ciência e Tecnologia da Ilum – Escola de Ciências. O objetivo do trabalho é implementar e analisar modelos generativos adversariais (GANs) de forma didática, com foco na geração de imagens sintéticas a partir de ruído aleatório.

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

Para utilizar o projeto, é necessário clonar este repositório e garantir que todas as dependências estejam instaladas corretamente no ambiente de execução. Recomenda-se o uso de uma IDE compatível com notebooks `.ipynb`, como JupyterLab ou VS Code. Além disso, o projeto utiliza `Git LFS (Large File Storage)` para o gerenciamento de arquivos grandes, sendo necessário tê-lo instalado antes do clone completo do repositório. Após a clonagem, abra o notebook `GAN.ipynb`, que contém todas as etapas do projeto, incluindo todos os códigos e explicações.

O arquivo `utils.py` contém funções auxiliares e é responsável pelo TensorBoard "Logger" usado para captar as informações das redes durante o treinamento.

A pasta `videos/` armazena as saídas visuais geradas durante o treinamento em forma de vídeo, permitindo a análise qualitativa da evolução do desempenho ao longo das épocas.

A pasta `dados/` contém arquivos auxiliares do projeto, incluindo as saídas visuais do treinamento e arquivos `.pt` com os pesos dos modelos treinados.

A pasta `dataset/` armazena os dados brutos utilizados no treinamento dos modelos, incluindo o dataset didático `MNIST` e um dataset próprio de imagens de rostos de personagens de anime (a construção desse dataset é detalhada no notebook do projeto).

A pasta `runs/` contém os logs gerados durante o treinamento das redes neurais, especialmente utilizados pelo TensorBoard, incluindo o armazenamento da loss durante as épocas.

# Docente
A matéria de Redes Neurais e Algoritmos Genéticos foi ministrada por:
- **Profº Dr. Daniel Roberto Cassar**
  
# Licença
Distribuído sob a licença GNU General Public License 3.0, cheque `LICENSE` para mais informações.
