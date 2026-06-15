# Este arquivo serve como script padrão para criar as redes neurais a serem utilizadas
#Importações
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.image import imread
import torch
import torch.nn as nn
import torch.optim as optim

class DataModule(L.LightningDataModule):
    # Classe DataModule, responsável por todo o tratamento dos dados de entrada
    # Importando os dados e transformando em matriz
    def __init__(
            self,
            inputData:np.ndarray,
            targetData:np.ndarray, # Temos um parâmetro de dados de target em caso de uso para denoising 
            n_cores = 0,
            seed = 4002,
            batch_size = 32,
            
    ):
        super().__init__()
        self.n_cores = n_cores
        self.seed = seed
        self.batch_size = batch_size
        self.entrada = inputData
        if targetData.size == 0:
            self.target = self.entrada
        else:
            self.target = targetData
    def setup(self, stage):
        
        # Vamos alimentar cada estágio com a mesma entrada e a mesma saída
        # Estamos trabalhando com aprendizado de máquina não supervisionado comparando a entrada 
        # e a saída, logo, em cada estágio da otimização, os valores permanecerão os mesmos.
        tensor_entrada = torch.tensor(self.entrada, dtype=torch.float32)
        tensor_target = torch.tensor(self.target, dtype=torch.float32)
        if stage == "fit":
            self.input_treino = tensor_entrada
            self.target_treino = tensor_target
            self.input_val = tensor_entrada
            self.target_val = tensor_target

        if stage == "test":
            self.input_test = tensor_entrada
            self.target_test = tensor_target

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_treino, self.target_treino),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_val, self.target_val),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
    def test_dataloader(self):
        return DataLoader(
            TensorDataset(self.input_test, self.target_test),
            batch_size= self.batch_size,
            num_workers=self.n_cores,
            shuffle=False # Não queremos que os dados sejam misturados
        )
    
class Autoencoder(L.LightningModule):
    # Por enquanto, faremos um autoencoder inteiro, que contenha o encoder e o decoder.
    def frobenius(self, x, n_vetores):
        """Usa-se em autoencoders contráteis.
        Params: self, x (vetor de entrada da rede neural), n_vetores (número de vetores que dividem a aproximação de Hutchinson)
        Estima a norma de frobenius do jacobiano da rede neural utilizando o estimador de Hutchinson.
        Return: float"""
        # Estima a norma de Frobenius do jacobiano usando o estimador de Hutchinson
        penalidade = 0
        x = x.clone().requires_grad_(True)
        y = self(x)
        for _ in range(n_vetores):
            v = torch.randint_like(x, 0, 2) * 2 - 1 # Gera um vetor aleatório
            Jv = torch.autograd.grad(y, x, grad_outputs=v, create_graph=True, retain_graph=True)[0] # Estima o produto Jacobiano-vetor
            penalidade += (Jv**2).sum() # Calcula a norma de Frobenius
        return penalidade / n_vetores
    
    def __init__(self, arquitetura_encoder, fun_ativ, fun_perda, penalty:str = "", factor:float = 0, n_vetores:int=5, lr:float=1e-3):
        super().__init__()
        self.fun_perda = fun_perda
        self.factor = factor
        self.lr = lr

        fun_ativ_type = type(fun_ativ)
        encoder_arq = []
        decoder_arq = []
        camadas_ = arquitetura_encoder
        for i in range(len(camadas_) -1):
            encoder_arq.append(nn.Linear(camadas_[i], camadas_[i + 1]))
            encoder_arq.append(fun_ativ)
        print(camadas_)
        r_camadas = camadas_.copy()
        r_camadas.reverse()
        print(r_camadas)
        for i in range(len(r_camadas) - 1):
            decoder_arq.append(nn.Linear(r_camadas[i], r_camadas[i+1]))
            decoder_arq.append(fun_ativ_type())
            
        
        self.encoder = nn.Sequential(*encoder_arq)
        self.decoder = nn.Sequential(*decoder_arq)

        # Funções lambda para o cálculo das loss functions.
        f_perda_pura = lambda self, x, y_pred, y: fun_perda(y_pred, y) if self.training else fun_perda(y_pred, y)
        f_perda_l1 = lambda self, x, y_pred, y: fun_perda(y_pred, y) + (factor * (sum([torch.abs(p).sum() for p in self.encoder.parameters()]) + sum([torch.abs(p).sum() for p in self.decoder.parameters()]))) if self.training else fun_perda(y_pred, y)
        f_perda_c = lambda self, x, y_pred, y: fun_perda(y_pred, y) + (factor * self.frobenius(x, n_vetores)) if self.training else fun_perda(y_pred, y)
        
        # Implementando diferentes tipos de regularização:
        
        if penalty == "l1":
                self.loss_function = f_perda_l1
        elif penalty == "c":
                self.loss_function = f_perda_c
        else:
            self.loss_function = f_perda_pura

        self.perdas_treino = []
        self.perdas_val = []
        self.curva_aprendizado_treino = []
        self.curva_aprendizado_val = []
    print()

    def forward(self, x):
        ls = self.encoder(x)
        out = self.decoder(ls)
        return(out)
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    def training_step(self, batch):
        self.train()
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(self, x, y_pred, y)

        self.log("loss", loss, prog_bar=True)
        self.perdas_treino.append(loss)
        return loss
    def validation_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(self, x, y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.perdas_val.append(loss)
        return loss
    def test_step(self, batch):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_function(self, x, y_pred, y)
        self.log("test_loss", loss, prog_bar=True)        
        return loss
    def on_train_epoch_end(self):
        #Atualiza a curva de aprendizado
        perda_media = torch.stack(self.perdas_treino).mean()
        self.curva_aprendizado_treino.append(float(perda_media))
        self.perdas_treino.clear()

    def on_validation_end(self):
        perda_media = torch.stack(self.perdas_val).mean()
        self.curva_aprendizado_val.append(float(perda_media))
        self.perdas_val.clear()

    def encode(self, x):
        return self.encoder(x)