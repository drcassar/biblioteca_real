import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

# ----------- #
#     MLP     #
# ----------- #

class MLP(nn.Module):
    def __init__(self, 
                 malha,
                 u_dados,
                 camadas_ocultas,
                 funcao_ativacao,
                 num_dados_entrada,
                 taxa_aprendizado=1e-3,
                 num_targets=1,
                 random_seed=9):
        '''
        MLP para aproximar u(t).
        
        Parâmetros:
            - malha:             malha para treino, valores de t;
            - u_dados:           dados reais para treino, valores de u(t);
            - camadas_ocultas:   lista. Define a arquitetura da rede: cada posição da lista corresponde 
                                 a uma camada e armazena a quantidade de neurônios nessa camada, ex: [2, 3] possui 
                                 2 camadas ocultas, a primeira com 2 e a segunda com 3 neurônios;
            - num_dados_entrada: inteiro. Define o número de atributos que a rede recebe como input;
            - taxa_aprendizado:  float. Define a taxa de aprendizado do otimizador (padrão: 1e-3);
            - num_targets:       inteiro. Define o número de alvos que a rede possui como output (padrão: 1, pois
                                 essa MLP visa aproximar uma função u(t), e funções possuem apenas um output;
            - random_seed:       semente aleatória para inicialização de parâmetros.
        '''
        super().__init__()

        # Setar semente aleatória
        torch.manual_seed(random_seed)

        self.malha   = torch.tensor(malha,   dtype=torch.float32)
        self.u_dados = torch.tensor(u_dados, dtype=torch.float32)
        self.taxa_aprendizado = taxa_aprendizado

        arquitetura = []

        # Primeira camada oculta
        arquitetura.append(nn.Linear(num_dados_entrada, camadas_ocultas[0]))
        arquitetura.append(funcao_ativacao)

        # Demais camadas ocultas
        for i in range(1, len(camadas_ocultas)):
            arquitetura.append(nn.Linear(camadas_ocultas[i - 1], camadas_ocultas[i]))
            arquitetura.append(funcao_ativacao)

        # Camada de saída
        arquitetura.append(nn.Linear(camadas_ocultas[-1], 1))

        self.camadas = nn.Sequential(*arquitetura)
        
    def forward(self, x):
        x = self.camadas(x)
        return x

    def fun_perda(self, u_pred):
        return torch.mean((u_pred - self.u_dados)**2) # MSE

    def treinar(self, num_epocas, verbose=True, intervalo_log=100, frac_adam=0.25):

        # Função pedida pelo otimizador L-BFGS
        def closure():
            '''Função closure para step do L-BFGS, ela recalcula a loss.'''

            # Zera gradiente
            otimizador.zero_grad()

            # Forward pass para CADA MALHA
            u_pred  = self.forward(self.malha)  # Previsão da malha interna

            # Calcula Loss
            loss = self.fun_perda(u_pred)

            # Backpropagation
            loss.backward()
            
            return loss

        # Instancia L-BFGS
        otimizador = optim.LBFGS(self.parameters(), lr=self.taxa_aprendizado)

        # Instancia histórico
        historico = []

        # Loop de treino
        for epoca in range(1, num_epocas+1):

            # Forward pass
            u_pred = self.forward(self.malha)

            # Loss
            loss = self.fun_perda(u_pred)
            historico.append(loss.item())

            # Atualiza parâmetros
            otimizador.step(closure)

            # Mostra resultado (opcional através de verbose)
            if verbose and (epoca % intervalo_log == 0 or epoca == 1):
                print(
                    f"Época {epoca:>{len(str(num_epocas))}}/{num_epocas} | "
                    f"Loss: {loss.item():.2e} | "
                )

        return historico

# ------------ #
#     PINN     #
# ------------ #

class PINN(nn.Module):
    def __init__(
        self,
        malha_interna,
        u_interno,
        malha_inicial,
        u_inicial,
        malha_contorno,
        u_contorno,
        func_campo,
        peso_ed,
        camadas_ocultas,
        num_dados_entrada,
        funcao_ativacao,
        params_ed=None,
        taxa_aprendizado=1e-3,
        num_targets=1,
        random_seed=9,
    ):
        '''
        Parâmetros:

        - malha_interna:     pontos internos da malha (colocation points), usados para
                             calcular o resíduo da equação diferencial;
        - u_interno:         dados experimentais na malha_interna;
        - malha_inicial:     pontos iniciais da malha, definem valores iniciaisde u(t);
        - u_inicial:         valores de u(t=0) nos pontos da malha interna (condição inicial);
        - malha_contorno:    pontos que definem o contorno da malha;
        - u_contorno:        valores de u nos pontos de contorno (condição de contorno);
        - peso_ed:           peso que pondera a loss da equação diferencial na loss total;
        - func_campo:        tensor com os valores de f(u, t) nos pontos internos,
                             que define a equação diferencial  du/dt = f(u, t);
        - camadas_ocultas:   lista com o número de neurônios em cada camada oculta,
                             ex: [32, 32, 32];
        - num_dados_entrada: número de variáveis de entrada da rede, ex: 1 para só t,
                             2 para t e x, etc.;
        - funcao_ativacao:   instância da função de ativação, ex: nn.Tanh();
        - params_ed:         dicionário de strings representando parâmetros da equação diferencial 
                             a serem incluídos como parâmetros da rede a fim de resolver 
                             o problema inverso relacionadas com um chute inicial para cada parâmetro
                             (padrão: None), ex: {'λ1':v1, 'λ2':v2, ...};
        - taxa_aprendizado:  taxa de aprendizado do otimizador (padrão: 1e-3).
        - random_seed:       semente aleatória para inicialização dos parâmetros.
        '''
        super().__init__()

        # Setar semente aleatória
        torch.manual_seed(random_seed)
        
        # requires_grad=True na malha interna pois precisa de du/dt via autograd
        self.malha_interna    = torch.tensor(malha_interna,  dtype=torch.float32, requires_grad=True)
        self.u_interno        = torch.tensor(u_interno,      dtype=torch.float32)
        self.malha_inicial    = torch.tensor(malha_inicial,  dtype=torch.float32)
        self.u_inicial        = torch.tensor(u_inicial,      dtype=torch.float32)
        self.malha_contorno   = torch.tensor(malha_contorno, dtype=torch.float32)
        self.u_contorno       = torch.tensor(u_contorno,     dtype=torch.float32)
        self.func_campo       = func_campo
        self.peso_ed          = peso_ed
        self.taxa_aprendizado = taxa_aprendizado
        if params_ed is not None:
            self.params_ed    = nn.ParameterDict()
            for nome, valor_inicial in params_ed.items():
                self.params_ed[nome] = nn.Parameter(torch.tensor([valor_inicial]))

        # Arquitetura
        arquitetura = []

        # Primeira camada oculta
        arquitetura.append(nn.Linear(num_dados_entrada, camadas_ocultas[0]))
        arquitetura.append(funcao_ativacao)

        # Demais camadas ocultas
        for i in range(1, len(camadas_ocultas)):
            arquitetura.append(nn.Linear(camadas_ocultas[i - 1], camadas_ocultas[i]))
            arquitetura.append(funcao_ativacao)

        # Camada de saída
        # Sem função de ativação na saída
        arquitetura.append(nn.Linear(camadas_ocultas[-1], num_targets))

        self.camadas = nn.Sequential(*arquitetura)

    # Função auxiliar para função de perda
    def derivada(self, u, t, ordem=1):
        '''
        Calcula a derivada de u em relação a t de forma recursiva.

        Parâmetros:
        - u:     tensor de saída da rede;
        - t:     tensor de entrada (deve ter requires_grad=True);
        - ordem: ordem da derivada desejada.
        '''
        du = u
        for _ in range(ordem):
            du = torch.autograd.grad(
                outputs=du,
                inputs=t,
                grad_outputs=torch.ones_like(du),
                create_graph=True,
                retain_graph=True,
            )
        return du[0]

    # ------------------------------ #
    #        Função de Perda         # -> onde a PINN é diferente 1.0
    # ------------------------------ #
    
    def fun_perda(self, u_contorno_pred, u_inicial_pred, u_interno_pred, ordem=1):
        '''
        Calcula a função de perda da PINN:

            L = loss_dados + peso_ed * loss_ed

        - loss_dados: erro quadrático médio nos pontos onde temos valores
        conhecidos (dados experimentais + condição inicial + condição de contorno);
        - loss_ed: erro quadrático médio dos resíduos da equação diferencial nos pontos internos, 
        calculado via diferenciação automática;
        - ordem: ordem da derivada na ED, definida como primeira ordem por padrão.
        
        Retorna: loss_total, loss_interna, loss_inicial, loss_contorno, loss_ed
        '''

        # Loss dos dados
        loss_interna  = torch.mean((u_interno_pred - self.u_interno)**2)   # MSE de u_interno
        loss_inicial  = torch.mean((u_inicial_pred - self.u_inicial)**2)   # MSE de u_inicial
        loss_contorno = torch.mean((u_contorno_pred - self.u_contorno)**2) # MSE de u_contorno
        
        loss_dados = loss_interna + loss_inicial + loss_contorno

        # Loss da Equação Diferencial (ed)
        du = self.derivada(u_interno_pred, self.malha_interna, ordem=ordem) # D(u(t)) via diferenciação automática
        residuo_ed = du - self.func_campo(u_interno_pred, **self.params_ed) # Resíduo: D(u(t)) - f(u, t) = 0
        loss_ed = torch.mean(residuo_ed**2)                                 # MSE de resíduo da Equação Diferencial

        # Loss total
        loss_total = loss_dados + self.peso_ed * loss_ed

        return loss_total, loss_interna, loss_inicial, loss_contorno, loss_ed

    def forward(self, x):
        return self.camadas(x)

    # ------------------------------ #
    #          Treinamento           # -> onde a PINN é diferente 2.0
    # ------------------------------ #
    
    def treinar(self, num_epocas, verbose=True, intervalo_log=100, ordem=1):
        '''
        Executa o loop de treinamento da PINN.

        Parâmetros:
        - num_epocas:    número de épocas de treinamento;
        - verbose:       se True, imprime a perda a cada intervalo_log épocas;
        - intervalo_log: de quantas em quantas épocas imprimir o histórico (log);
        - ordem:         ordem da derivada na ED, definida como primeira ordem por padrão.
                         
        Retorna: histórico com as perdas por época.
        '''

        # Instancia otimizador L-BFGS
        otimizador = optim.LBFGS(self.parameters(), lr=self.taxa_aprendizado)

        # Instancia o dicionário de histórico
        historico = {
            'loss_total':     [],
            'loss_interna':   [],
            'loss_inicial':   [],
            'loss_contorno':  [],
            'loss_ed':        [],
        }

        # Função pedida pelo otimizador L-BFGS
        def closure():
            '''Função closure para step do L-BFGS, ela recalcula a loss total.'''

            # Zera gradiente
            otimizador.zero_grad()

            # Forward pass para CADA MALHA
            u_interno_pred  = self.forward(self.malha_interna)  # Previsão da malha interna
            u_inicial_pred  = self.forward(self.malha_inicial)  # Previsão da malha inicial
            u_contorno_pred = self.forward(self.malha_contorno) # Previsão da malha de contorno

            # Calcula Loss
            loss_total, _, _, _, _ = self.fun_perda(
                u_contorno_pred, u_inicial_pred, u_interno_pred, ordem=ordem)

            # Faz backpropagation da loss total
            loss_total.backward()
               
            return loss_total

        # Loop de treinamento
        for epoca in range(1, num_epocas + 1):

            # Forward pass para CADA MALHA
            u_interno_pred  = self.forward(self.malha_interna)  # Previsão da malha interna
            u_inicial_pred  = self.forward(self.malha_inicial)  # Previsão da malha inicial
            u_contorno_pred = self.forward(self.malha_contorno) # Previsão da malha de contorno

            # Calcula Loss
            loss_total, loss_interna, loss_inicial, loss_contorno, loss_ed = self.fun_perda(
                u_contorno_pred, u_inicial_pred, u_interno_pred, ordem=ordem)

            # Atualiza parâmetros (zero grad e backprop embutidos)
            otimizador.step(closure)

            # Salva histórico
            historico['loss_total'].append(loss_total.item())
            historico['loss_interna'].append(loss_interna.item())
            historico['loss_inicial'].append(loss_inicial.item())
            historico['loss_contorno'].append(loss_contorno.item())
            historico['loss_ed'].append(loss_ed.item())

            # Mostra resultado (opcional através de verbose)
            if verbose and (epoca % intervalo_log == 0 or epoca == 1):
                print(
                    f"Época {epoca:>{len(str(num_epocas))}}/{num_epocas} | "
                    f"Loss: {loss_total.item():.2e} | "
                    f"Interna: {loss_interna.item():.2e} | "
                    f"Inicial: {loss_inicial.item():.2e} | "
                    f"Contorno: {loss_contorno.item():.2e} | "
                    f"ED: {loss_ed.item():.2e}"
                )

        return historico