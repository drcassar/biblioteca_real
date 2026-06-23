import optuna
import copy
import os


class OtimizarExperimento():
    """Classe destinada a otimização de redes neurais MLP com base no módulo PyTorch Lightning do Python. A classe permite a entrada de um dict contendo as configurações do sistema a ser otimizado. Exemplo:
    configs = {
        'experimento': {
            'nome': 'Teste_classe_otimizacao',
            'n_trials': 5,
            'max_epochs': 100 
        },
        'otimizador': {},
        'DataModule': {
            'num_pontos': 5000,
        },
        'Module': {}
    }"""

    def __init__(self, configs, treiner_fn, DataModule, Module, dir_base):
        """
        Recebe os dados necessários para a otimização pode acontecer.

        Args:
            configs (dict): Um dict de dict contenco os parâmetros e o colca onde colocá-los
            treiner_fn (func): Uma função que recebe "trial" e "config" e retorne o treiner
            DataModule (class): A referência da classe que herda LightningDataModule
            Module (class): A referência da classe que herda LightningModule
            dir_base (Path): O caminho da pasta (pathlib.<sistema>Path) em que o script principal roda. Em pathlib
        """
        
        self.configs = configs
        self.treiner_fn = treiner_fn
        self.DataModule = DataModule # Passar a referência
        self.Module = Module # Passar a referência
        self.dir_base = dir_base

        self.garantir_diretorios()

    
    
    def garantir_diretorios(self):        
        for nome in ['logs', 'db', 'figuras']:
            (self.dir_base / nome).mkdir(exist_ok=True)

    def monta_trial(self, trial):
        configs = copy.deepcopy(self.configs)

        # ========== Hiperparâmetros ==========
        # ===== Batch size (DataModule)
        batch_size = trial.suggest_categorical('batch_size', [int(2)**i for i in range(5, 14)]) # Verificar caso o número de dados mude

        # ===== Learning rate (otimizador)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        # ===== Número de camadas ocultas & Perceptrons por camada (Modelo)
        num_camadas_ocultas = trial.suggest_int('num_layers', 1, 5)
        camadas_o = [trial.suggest_int(f'Neurônios_c{i}', 2, 40, log=True) for i in range(num_camadas_ocultas)]

        # ========== Moldar o dicionário ==========
        configs['otimizador']['lr'] = lr
        configs['DataModule']['batch_size'] = batch_size
        configs['Module']['camadas_o'] = camadas_o

        dm = self.DataModule(**configs['DataModule'])
        modelo = self.Module(**configs['Module'], **configs['otimizador'])

        return dm, modelo, configs
        
    def objetivo(self, trial):
        dm, modelo, configs = self.monta_trial(trial)
        treinador = self.treiner_fn(trial, configs)
        treinador.fit(modelo, datamodule=dm)

        val_loss = treinador.callback_metrics['val_loss'].item()
        
        return val_loss
    
    def optimize(self):
        """
        Roda o otimizador do optuna, criando o estudo e minimizando a loss. A ideia é criar um único .db e vários estudos nele

        Um .db por experiemnto e seus vários estudos.
        """

        nome = self.configs['experimento']['nome']
        caminho_db = self.dir_base / 'db' / nome
        self.estudo = optuna.create_study(
            study_name=nome,
            # storage=f'sqlite:///{self.configs['experimento']['nome']}.db',
            storage=f'sqlite:///{caminho_db}.db', # Talvez usar f'sqlite:///{caminho_dbas_posix()}.db'
            load_if_exists=True,
            direction='minimize'
        )

        self.estudo.optimize(
            self.objetivo,
            n_trials=self.configs['experimento']['n_trials']
        )

        print('Melhores parâmetros:', self.estudo.best_params)