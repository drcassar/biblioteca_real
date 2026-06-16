#region Ponte
from sys import path
path.insert(0, '/home/joao25006/Redes_neurais/PINN-Ponte-de-vidro')

import torch.nn as nn
import torch
from src.logging.logging import TrialLoggingCallback
from lightning.pytorch.loggers import CSVLogger
from src.otimizacao.otimizador import OtimizarExperimento
from src.ferramentas.DataModule import DataModule
from src.ferramentas.Module import Module
import lightning as L
from pathlib import Path
caminho_base = Path(__file__).resolve().parent

import numpy as np



class Ponte():
    """Guarda os parâmetros físicos do problema. Define o ansatz, os resíduos. Trabalha apenas o modelo físico"""

    def __init__(self, comprimento: float, T: float, massa_linear: float, amorteciento: float, rigidez: float):
        """As características da ponte. Valores em S.I.

        Args:
            comprimento (float): Comprimento da ponte em
            T (float): Tempo do intervalo e referência de adimensionalização
            massa_linear (float): Densidade vezes área da seção transversal
            amorteciento (float): termo de amortecimento
            rigidez (float): termo restaurador
        """

        self.massa_linear = massa_linear
        self.amorteciento = amorteciento
        self.rigidez = rigidez
        self.L = comprimento
        self.T = T

        self.perda = nn.MSELoss()


    def aplica_ansatz(self, X, nn_out):
        """Recebe a saída bruta da rede e aplica o ansatz (hard constraint)

        Args:
            X (_type_): O tensor de pontos para apicar o ansatz
            nn_out (_type_): A saída da rede

        Returns:
            _type_: O tensor com u e w
        """

        x = X[:, 0:1] # 0:1 preserva shape coluna
        ansatz = x * (1 - x)
        u = ansatz * nn_out[:, 0:1]
        w = nn_out[:, 1:2]

        return torch.cat((u, w), dim=1)
    
    def separa_canais(self, nn_out):
        """Separa semanticamente as saídas da rede. Distingue u e w

        Args:
            nn_out (_type_): A saída da rede composta de u e w

        Returns:
            _type_: u e w separados
        """

        u = nn_out[:, 0:1]
        w = nn_out[:, 1:2]

        return u, w
    
    def residuo_fisico(self, w, u_t, u_tt, u_xx, w_xx):
        """Monta o resíduo principal da EDP. O chamado resíduo físico. Concidera o sistema adimensional.

        Args:
            w (_type_): w, a segunda  dimensão do problema (derivada segunda de u)
            u_t (_type_): derivada primeira de um em t
            u_tt (_type_): derivada segunda de u em t
            u_xx (_type_): derivada segunda de u em x
            w_xx (_type_): derivada segund de w em x

        Returns:
            _type_: Os resíduos da EDP e do truque da segunda equação
        """

        zeta = self.amorteciento * self.T / self.massa_linear
        Pi = self.rigidez * self.T**2 / (self.massa_linear * self.L**4)

        R1 = u_tt + zeta * u_t + Pi * w_xx
        R2 = w - u_xx

        return R1, R2
    
    def residuo_contorno(self, u_xx_0, u_xx_l):
        """Monta o resíduo das condições de contorno
        
        Args:
            u_xx_0 (_type_): A derivada segunda de u em x no ponto 0
            u_xx_l (_type_): A derivada segunda de u em x no ponto máximo (comprimento l)

        Returns:
            _type_: O resíduo relacionado às condições de contorno
        """    
         
        loss_contorno = self.perda(u_xx_0, torch.zeros_like(u_xx_0)) + self.perda(u_xx_l, torch.zeros_like(u_xx_l))
        ### u_xx_0 e u_xx_l deveriam estar divididos por L0

        return loss_contorno

    def residuo_inicio(self, X, u, u_t):
        """Monta o resíduo da condição inicial

        Args:
            X (_type_): O tensor de pontos incerido na rede
            u (_type_): o valor de u
            u_t (_type_): a primeira derivada de u em t

        Returns:
            _type_: O resíduo relacionado às condições iniciais
        """
         
        ux0 = torch.sin(torch.pi * X) # É um perfil
        loss_inicio = self.perda(u, ux0) + self.perda(u_t, torch.zeros_like(ux0))
        ### u e u_t deveriam estar divididos por L0???

        return loss_inicio
    
    def avaliar_pinn(self, x0, xf, t0, tf, n_pontos_x, n_pontos_t, modelo):
        x_fisico = np.linspace(x0, xf, n_pontos_x)
        t_fisico = np.linspace(t0, tf, n_pontos_t)
        x = x_fisico / self.L
        t = t_fisico / self.T
        X, T = np.meshgrid(x, t)

        XT = np.column_stack([ X.ravel(), T.ravel()])

        XT_tensor = torch.tensor(XT, dtype=torch.float32, device=modelo.device)

        modelo.eval()

        with torch.no_grad():
            U = modelo(XT_tensor)[:, 0].cpu().numpy()

        U = U.reshape(X.shape)
        U_fisico = U ### Definir o atributo amplitude

        return x_fisico, t_fisico, U_fisico
    
    def solucao_analitica(self, x0, xf, t0, tf, n_pontos_x, n_pontos_t):

        x = np.linspace(x0, xf, n_pontos_x)
        t = np.linspace(t0, tf, n_pontos_t)

        X, T = np.meshgrid(x, t)

        # ===== parâmetros físicos =====
        mu = self.massa_linear
        c  = self.amorteciento
        EI = self.rigidez
        L  = self.L

        # ===== amplitude inicial =====
        A = 0.01

        # ===== frequências =====
        alpha = c / (2 * mu)

        omega_n = np.sqrt(
            (EI / mu) * (np.pi / L)**4
        )

        omega_d = np.sqrt(
            omega_n**2 - alpha**2
        )

        # ===== solução =====
        U = (
            A
            * np.sin(np.pi * X / L)
            * np.exp(-alpha * T)
            * np.cos(omega_d * T)
        )
        U = U.reshape(X.shape)
        U_fisico = U

        return x, t, U_fisico
#endregion



if __name__ == '__main__':
    problema = Ponte(comprimento=20, T=0.5, massa_linear=800, amorteciento=2000, rigidez=5e8)

    # ========== configurações de treino ==========
    configs = {
        'experimento': {
            'nome': '1_alfa_300_adimencional',
            'n_trials': 100,
            'max_epochs': 300 
        },
        'otimizador': {},
        'DataModule': {
            'np_interior': 1000,
            'np_contorno': 500,
            'np_inicio': 500,
            'np_interior_val': 500,
            'np_contorno_val': 250,
            'np_inicio_val': 250,
            'dimensoes': 2,
        },
        'Module': {
            'dimensoes': 2,
            'num_saidas': 2,
            'funcao_ativacao': nn.Tanh,
            'problema': problema
        }
    }


    def t(trial, configs5):
        callbacks = (TrialLoggingCallback(trial, 60))

        logger_CSV = CSVLogger(
            save_dir=caminho_base / 'logs',
            name='PINN',
            version=f'trial_{trial.number}'
        )

        # ===== Trainer =====
        treinador = L.Trainer(
            max_epochs=configs['experimento']['max_epochs'],
            logger=logger_CSV,
            callbacks=callbacks,
            devices=1,
            accelerator='gpu',
            enable_progress_bar=False,
            enable_model_summary=False, # Evita prints extras
        )

        return treinador
    
    otimizador = OtimizarExperimento(configs, t, DataModule, Module, caminho_base)
    otimizador.optimize()