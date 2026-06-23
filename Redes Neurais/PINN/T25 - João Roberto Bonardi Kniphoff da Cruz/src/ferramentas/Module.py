# ========== Redes neurais ==========
import lightning as L
import torch.nn as nn
import torch
from sys import path
from os import getcwd
path.insert(0, getcwd())
from src.ferramentas.GradNorm import GradNorm
from src.ferramentas.calculo import OperadoresDiferenciais


class Module(L.LightningModule):
    """Os atributos e métodos da rede neural artificial (ANN)"""

    def __init__(self, dimensoes, camadas_o, num_saidas, funcao_ativacao, lr, problema, funcao_ativacao_saida=False):
        super().__init__()

        self.problema = problema

        self.lr = lr

        arquitetura = []
        # ========== Primeira camada ==========
        arquitetura.append(nn.Linear(dimensoes, camadas_o[0]))
        arquitetura.append(funcao_ativacao())

        # ========== Demais camadas ocultas ==========
        for i in range(1, len(camadas_o)):
            arquitetura.append(nn.Linear(camadas_o[i - 1], camadas_o[i]))
            arquitetura.append(funcao_ativacao())

        # ========== Camada de saída ==========
        arquitetura.append(nn.Linear(camadas_o[-1], num_saidas))
        if funcao_ativacao_saida:
            arquitetura.append(funcao_ativacao())
            # arquitetura.append(funcao_ativacao_saida()) # Para uma função diferente

        self.rede = nn.Sequential(*arquitetura)
        self.func_perda = nn.MSELoss()

        # ===== Pesos treináveis
        self.lambda_fisico   = nn.Parameter(torch.tensor(1.0))
        self.lambda_contorno = nn.Parameter(torch.tensor(1.0))
        self.lambda_inicio   = nn.Parameter(torch.tensor(1.0))

        # Parâmetros do GradNorm
        self.L0 = None
        self.alpha = 0.5
        self.beta = 1.0
        self.last_layer = list(self.rede.children())[-1] # Última camada


    def forward(self, X):
        """Define como os valores passam pela rede e geram um resultado"""

        nn_out = self.rede(X) # Passa pela rede
        ansatz = self.problema.aplica_ansatz(X, nn_out)
        
        return ansatz

    def training_step(self, batch, batch_idx):
        dict_losses = self.calcular_losses(batch)

        if self.L0 is None:
            self.L0 = torch.tensor([
                dict_losses['fisica'].item(),
                dict_losses['contorno'].item(),
                dict_losses['inicio'].item()
            ], device=self.device)

        lambdas = torch.stack([
            self.lambda_fisico,
            self.lambda_contorno,
            self.lambda_inicio
        ])

        lambdas = torch.nn.functional.softplus(lambdas) # Evita negativo e é diferenciável

        # normaliza (soma = 3)
        lambdas = 3 * lambdas / lambdas.sum()

        loss_total = (
            lambdas[0] * dict_losses['fisica'] +
            lambdas[1] * dict_losses['contorno'] +
            lambdas[2] * dict_losses['inicio']
        )

        loss_grad = GradNorm.calcula_gardnorm(self, dict_losses['fisica'], dict_losses['contorno'], dict_losses['inicio'], lambdas)
        loss_final = loss_total + self.beta * loss_grad

        # ========== Logging individual ==========
        self.log('train_loss_final', loss_final, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss_total, on_step=False, on_epoch=True, prog_bar=True)

        for nome, loss in dict_losses.items():
            self.log(f'train_{nome}', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log('t_lambda_f', lambdas[0], on_epoch=True)
        self.log('t_lambda_c', lambdas[1], on_epoch=True)
        self.log('t_lambda_i', lambdas[2], on_epoch=True)

        return loss_final
    
    def validation_step(self, batch, batch_idx):
        dict_losses = self.calcular_losses(batch)

        if self.L0 is None:
            self.L0 = torch.tensor([
                dict_losses['fisica'].item(),
                dict_losses['contorno'].item(),
                dict_losses['inicio'].item()
            ], device=self.device)

        lambdas = torch.stack([
            self.lambda_fisico,
            self.lambda_contorno,
            self.lambda_inicio
        ])

        lambdas = torch.nn.functional.softplus(lambdas) # Evita negativo e é diferenciável

        # normaliza (soma = 3)
        lambdas = 3 * lambdas / lambdas.sum()

        loss_total = (
            lambdas[0] * dict_losses['fisica'] +
            lambdas[1] * dict_losses['contorno'] +
            lambdas[2] * dict_losses['inicio']
        )

        # ========== Logging individual ==========
        self.log('val_loss', loss_total, on_step=False, on_epoch=True, prog_bar=True)

        for nome, loss in dict_losses.items():
            self.log(f'val_{nome}', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log('t_lambda_f', lambdas[0], on_epoch=True)
        self.log('t_lambda_c', lambdas[1], on_epoch=True)
        self.log('t_lambda_i', lambdas[2], on_epoch=True)

        return loss_total
    
    def configure_optimizers(self):
        otimizador = torch.optim.Adam(self.parameters(), lr=self.lr)

        return otimizador

    # ========== Métodos auxiliares ==========
    def calcular_losses(self, batch):
        """Chama as funções de cálculo de losses para organizar a entreda para os "steps" """

        for local in ['interior', 'contorno', 'inicio']:
            match local:
                case 'interior':
                    X = batch[local][0] ### Devido ao CombinedLoader
                    requirements = ['u', 'w', 'u_t', 'u_tt', 'u_xx', 'w_xx'] # Tem que calcular tudo

                    nn_out = self(X)
                    u, w = self.problema.separa_canais(nn_out)
                    jacob = OperadoresDiferenciais.jacobiana(self, X)
                    hess = OperadoresDiferenciais.hessiana(self, X)
                    u_t, u_tt, u_xx, w_xx = OperadoresDiferenciais.extrair_derivadas_total(jacob, hess)

                    R1, R2 = self.problema.residuo_fisico(w, u_t, u_tt, u_xx, w_xx)
                    loss = self.func_perda(R1, torch.zeros_like(R1)) + self.func_perda(R2, torch.zeros_like(R2))

                    loss_fisica = loss # É o retorno
                
                case 'contorno':
                    X = batch[local][0]
                    requirements = ['u_xx'] # Apenas a hessiana

                    hess = OperadoresDiferenciais.hessiana(self, X)
                    u_tt, u_xx, w_xx = OperadoresDiferenciais.extrair_derivadas_hess(hess)

                    x = X[:, 0] ### Não X[:, 0:1]
                    mask0 = (x == 0)
                    maskL = (x == 1)
                    u_xx_0 = u_xx[mask0]
                    u_xx_L = u_xx[maskL]
                    loss = self.problema.residuo_contorno(u_xx_0, u_xx_L)

                    loss_contorno = loss # É o retorno
                
                case 'inicio':
                    X   = batch[local][0]
                    requirements = ['u', 'u_t'] # Apenas canal 1 e a jacobiana

                    nn_out = self(X)
                    u, _ = self.problema.separa_canais(nn_out)
                    jacob = OperadoresDiferenciais.jacobiana(self, X)
                    u_t = OperadoresDiferenciais.extrair_derivadas_jacob(jacob)

                    x = X[:,0:1] ### Não X[:, 0:1]
                    loss = self.problema.residuo_inicio(x, u, u_t)

                    loss_inicio = loss # É o retorno

        return {
            'fisica': loss_fisica,
            'contorno': loss_contorno,
            'inicio': loss_inicio
        }