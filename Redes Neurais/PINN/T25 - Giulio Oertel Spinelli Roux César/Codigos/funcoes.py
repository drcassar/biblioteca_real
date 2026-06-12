import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------- #
#     Funções normalização     #
# ---------------------------- #

def normalizar(v, v_min, v_max):
    '''Normaliza v entre [-1,1].'''
    if not isinstance(v, torch.Tensor): # se for tensor, a conversão atrapalha o torch (requires_grad)
        v, v_min, v_max = np.array(v), np.array(v_min), np.array(v_max)
    return 2*(v - v_min) / (v_max - v_min) - 1

def desnormalizar(v_norm, v_min, v_max):
    '''Desnormaliza v_norm se v_norm está entre [-1,1].'''
    if not isinstance(v_norm, torch.Tensor): # se for tensor, a conversão atrapalha o torch (requires_grad)
        v_norm, v_min, v_max = np.array(v_norm), np.array(v_min), np.array(v_max)
    return ((v_norm + 1) * (v_max - v_min)) / 2 + v_min

# ------------------------ #
#     Funções de campo     #
# ------------------------ #

def func_campo_resfriamento(T, r, T_amb):
    '''Calcula função de campo da lei de resfriamento.'''
    return r*(T_amb - T)

def func_campo_ohs(x, w0):
    '''Calcula função de campo do oscilador harmônico.'''
    return - (w0**2) * x

def func_campo_norm(u_norm, func_campo, t_min, t_max, u_min, u_max, ordem=1, **kwargs):
    '''
    Calcula a função de campo para u normalizado entre -1 e +1, 
    considerando Equação Diferencial de Ordem dada.
    '''
    u = desnormalizar(u_norm, u_min, u_max)
    escala_t = (t_max - t_min) / 2
    escala_u = (u_max - u_min) / 2
    return func_campo(u, **kwargs) * escala_t**ordem / escala_u

# -------------------------------------- #
#     Funções de soluções analíticas     #
# -------------------------------------- #

def solucao_analitica_resfriamento(t, T0, r, T_amb):
    '''Calcula a solução exata: T(t) = (T0 - T_amb) * exp(-r * t) + T_amb'''
    return (T0 - T_amb) * np.exp(-r * t) + T_amb

def solucao_analitica_ohs(t, x0, w0):
    '''Calcula a solução exata: x(t) = x0*(cos(w0*t) + sen(w0*t))'''
    return x0*(np.cos(w0*t) + np.sin(w0*t))

# -------------------------------- #
#     Função para criar malhas     #
# -------------------------------- #

def criar_malhas(t_min, t_max, u0, solucao_analitica, n_dados, nivel_ruido, 
                 norm=False, u_min=None, u_max=None, equidistante=False, random_seed=9):
    '''
    Cria malhas interna, inicial e de contorno e gera dados sintéticos com ruído.

    Parâmetros:
    - t_min, t_max:      listas de valores que formam o intervalo do domínio, ex: [t1_min, t2_min, ...] e 
                         [t1_max, t2_max, ...] em que t1 e t2 indicam argumentos de u(t1, t2, ...);
    - u0:                matriz com valores iniciais a depender da ordem da ED, cada coluna
                         representa um argumento de u, ex: 
                             [[u(0, t2, ...), u'(0, t2, ...), ...],
                              [u(t1, 0, ...), u'(t1, 0, ...), ...], ...];
    - solucao_analitica: função que calcula solução analítica do problema (para criar dados sintéticos);
    - n_dados:           número de pontos na malha interna, usada para treino do modelo;
    - nivel_ruido:       desvio padrão do ruído gaussiano;
    - norm:              booleano que indica normalização das malhas e dados (normaliza no intervalo [-1,1]);
    - u_min, u_max:      limites de u para normalização; 
                         São necessários para normalização apenas, por isso têm valor padrão None;
    - equidistante:      se True, malha interna sorteia pontos equidistantes;
    - random_seed:       semente aleatória, definida em 9 por padrão.
    
    Retorna: 
    - malhas inicial, de contorno e interna; 
    - dados sintéticos de valores de u iniciais, de contorno e internos; 
    '''    
    rng = np.random.default_rng(random_seed)
    
    if norm:
        assert u_min is not None and u_max is not None, \
            'São esperados valores de u_min e u_max para normalização.'

    # Dados experimentais sintéticos
    if equidistante:
        malha_interna = np.linspace(t_min, t_max, n_dados).reshape(-1, 1) # malha de pontos equidistantes
    else:
        malha_interna = np.sort(rng.uniform(t_min, t_max, n_dados)).reshape(-1, 1) # malha de pontos aleatórios
    u_interno = solucao_analitica(malha_interna)                      # dados experimentais nos pontos internos da malha
    u_interno += rng.normal(0, nivel_ruido, u_interno.shape)    # ruído

    # Condição inicial: u(t=0) = u0 
    malha_inicial = np.array([t_min])                                 # malha de pontos iniciais
    u_inicial = np.array(u0)                                          # valores iniciais [[u(0), u'(0), u''(0), ...]]

    # Condição de contorno 
    malha_contorno = np.array([t_max])                                # malha de pontos de contorno 
    u_contorno = solucao_analitica(malha_contorno)                    # valores de u de contorno 

    # Normalização
    if norm:
        malha_interna  = normalizar(malha_interna,  t_min, t_max)
        malha_inicial  = normalizar(malha_inicial,  t_min, t_max)
        malha_contorno = normalizar(malha_contorno, t_min, t_max)

        u_interno      = normalizar(u_interno,      u_min, u_max)
        u_inicial      = normalizar(u_inicial,      u_min, u_max)
        u_contorno     = normalizar(u_contorno,     u_min, u_max)    
    
    return malha_interna, malha_inicial, malha_contorno, u_interno, u_inicial, u_contorno

# ------------------------------------ #
#     Função de previsão do modelo     #
# ------------------------------------ #

def prever(modelo, t_min, t_max, t_min_prev, t_max_prev, 
           norm=False, u_min=None, u_max=None, n_pontos=500):
    '''
    Gera previsões do modelo em um intervalo arbitrário.

    Parâmetros:
    - modelo:                 instância treinada da PINN;
    - t_min, t_max:           intervalo usado no treino, necessário para normalizar
                              corretamente a entrada da rede;
    - t_min_prev, t_max_prev: intervalo de previsão (pode extrapolar o treino);
    - norm:                   booleano, se verdadeiro normaliza entrada e desnormaliza saída;
    - u_min, u_max:           limites de u para desnormalização;
    - n_pontos:               número de pontos na malha de previsão.

    Retorna: t_plot, u_prev  (ambos na escala original)
    '''
    # Malha de previsão
    t_plot = np.linspace(t_min_prev, t_max_prev, n_pontos).reshape(-1, 1)

    # Altera o estado do modelo para inferência
    modelo.eval()
    
    with torch.no_grad():
        t_tensor = torch.tensor(t_plot, dtype=torch.float32)

        # Normalização
        if norm:
            t_tensor = normalizar(t_tensor, t_min, t_max) # com intervalo de treino

        # Previsão
        u_prev = modelo(t_tensor).numpy()

        # Desnormalização
        if norm:
            u_prev = desnormalizar(u_prev, u_min, u_max)

    return t_plot, u_prev



# ----------------------- #
#     Funções de Plot     #
# ----------------------- #

# Configuração global 
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         'Times New Roman',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'axes.linewidth':     0.8,
    'xtick.direction':    'in',   # graduação para dentro
    'ytick.direction':    'in',   # graduação para dentro
    'xtick.top':          True,   # graduação no topo
    'ytick.right':        True,   # graduação à direita
    'xtick.minor.visible': True,  # graduação menor
    'ytick.minor.visible': True,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.8',
    'figure.dpi':         150,
})

# PLOT LOSSES

def plot_losses(historico, nome_modelo, periodica=False, salvar=False, nome_arquivo='losses.png'):
    '''
    Plota as componentes da loss ao longo das épocas.

    Parâmetros:
    - historico:    histórico de losses, o retorno do modelo.treinar();
    - nome_modelo:  nome do modelo para plot correto do histórico e título dos gráficos.
                    Pode ser um de dois valores: 'PINN' ou 'MLP';
    - periodica:    se True, plota loss_periódica;
    - salvar:       se True, salva o gráfico como imagem;
    - nome_arquivo: nome do arquivo de saída (padrão: 'solucao.png').
    '''

    assert nome_modelo == 'MLP' or nome_modelo == 'PINN', \
        "nome_modelo apenas assume um dos valores em {'MLP', 'PINN'}."

    # PLOT PINN

    if nome_modelo == 'PINN':
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        
        epocas = range(1, len(historico['loss_total']) + 1)
        
        # --- Painel esquerdo: todas as losses juntas ---
        axes[0].plot(epocas, historico['loss_total'],    label='Total',    linewidth=1.8, color='black')
        axes[0].plot(epocas, historico['loss_interna'],  label='Interna',  linewidth=1.0, linestyle='--', color='steelblue')
        axes[0].plot(epocas, historico['loss_inicial'],  label='Inicial',  linewidth=1.0, linestyle='-.', color='firebrick')
        axes[0].plot(epocas, historico['loss_contorno'], label='Contorno', linewidth=1.0, linestyle=':',  color='seagreen')
        axes[0].plot(epocas, historico['loss_ed'],       label='ED',       linewidth=1.0, linestyle=(0,(3,1,1,1)), color='darkorange')
        if periodica:
            axes[0].plot(epocas, historico['loss_periodica'], label='Periódica', linewidth=1.0, linestyle='--', color='dodgerblue')

        axes[0].set_yscale('log')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Evolução das losses da PINN')
        axes[0].legend()

    # --- Painel direito: loss total ---
        axes[1].plot(epocas, historico['loss_total'], linewidth=1.8, color='black')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss total')
        axes[1].set_title('Convergência da loss total da PINN')

    # PLOT MLP

    if nome_modelo == 'MLP':
        fig, ax = plt.subplots(figsize=(5, 3.5))

        epocas = range(1, len(historico) + 1)
    
        ax.plot(epocas, historico, linewidth=1.8, color='black')
        ax.set_yscale('log')
        ax.set_xlabel('Época')
        ax.set_ylabel('Loss total')
        ax.set_title('Convergência da loss da MLP')
        ax.tick_params(axis='both')

    plt.tight_layout()

    if salvar:
        plt.savefig(nome_arquivo, bbox_inches='tight')

    plt.show()

# PLOT SOLUÇÕES

def plot_solucao(solucao_analitica, t_plot, u_prev, malha_interna, u_interno,
                 u_inicial, u_contorno, t_min, t_max, nome_modelo,
                 norm=False, u_min=None, u_max=None,
                 salvar=False, nome_arquivo='solucao.png'):
    '''
    Plota comparação entre solução do modelo e a solução analítica.

    Parâmetros:
    - solucao_analitica: função que recebe t (numpy) e retorna u exato;
    - t_plot:            malha em t para plot;
    - u_prev:            previsão do modelo para t_plot;
    - malha_interna:     pontos de colocação usados no treino;
    - u_interno:         valores obtidos experimentalmente;
    - u_inicial:         valores iniciais;
    - u_contorno:        valores no contorno;
    - t_min, t_max:      intervalo de treino, para posicionar dados observados;
    - nome_modelo:       nome do modelo para título dos gráficos;
    - norm:              booleano, se verdadeiro desnormaliza dados observados;
    - u_min, u_max:      limites de u para desnormalização;
    - salvar:            se True, salva o gráfico como imagem;
    - nome_arquivo:      nome do arquivo de saída.
    '''
    if norm:
        assert u_min is not None and u_max is not None, \
            'São esperados valores de u_min e u_max para normalização.'
        
        malha_interna = desnormalizar(malha_interna, t_min, t_max)
        u_interno     = desnormalizar(u_interno,     u_min, u_max)
        u_inicial     = float(desnormalizar(u_inicial.flatten()[0], u_min, u_max))
        u_contorno    = float(desnormalizar(u_contorno.flatten()[0], u_min, u_max))
    else:
        u_inicial  = float(u_inicial.flatten()[0])
        u_contorno = float(u_contorno.flatten()[0])

    # Solução analítica 
    u_analitica = solucao_analitica(t_plot)

    # Erro absoluto
    erro = np.abs(u_prev - u_analitica)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- Painel esquerdo: curvas ---
    axes[0].scatter(t_min, u_inicial,
                    s=30, color='crimson', zorder=5, label='Condição inicial', marker='s')
    axes[0].scatter(t_max, u_contorno,
                    s=30, color='crimson', zorder=5, label='Condição de contorno', marker='^')
    axes[0].scatter(malha_interna, u_interno,
                    s=20, color='crimson', zorder=4, label='Dados observados', marker='o')
    axes[0].plot(t_plot, u_analitica,
                 color='black', linewidth=1.8, label='Solução analítica')
    axes[0].plot(t_plot, u_prev,
                 color='dodgerblue', linewidth=1.8, linestyle='--', label=nome_modelo)
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$u(t)$')
    axes[0].set_title(f'{nome_modelo} vs Solução analítica')
    axes[0].legend()

    # --- Painel direito: erro absoluto ---
    axes[1].plot(t_plot, erro, color='black', linewidth=1.8)
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel(f'$|u_{{\\mathrm{{{nome_modelo}}}}} - u_{{\\mathrm{{anal.}}}}|$')
    axes[1].set_title(f'Erro absoluto de {nome_modelo}')

    plt.tight_layout()

    if salvar:
        plt.savefig(nome_arquivo, bbox_inches='tight')

    plt.show()