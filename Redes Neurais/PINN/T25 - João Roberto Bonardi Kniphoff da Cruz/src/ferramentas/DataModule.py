import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader # Para permitir vários datasets por step

class DataModule(L.LightningDataModule):
    """Organiza e amostra os pontos no intervalo [0, 1], apenas"""

    trabalhadores_persistentes = True # Facilita a troca, se necessário

    def __init__(self, np_interior, np_contorno, np_inicio, np_interior_val, np_contorno_val, np_inicio_val, dimensoes, num_trabalhadores=5, batch_size=1024):
        """Guardar os parâmetros da amostragem"""

        super().__init__()

        # ========== Número de pontos ==========
        self.np_interior = np_interior
        self.np_contorno = np_contorno
        self.np_inicio = np_inicio
        self.np_interior_val = np_interior_val
        self.np_contorno_val = np_contorno_val
        self.np_inicio_val = np_inicio_val
        
        self.dimensoes = dimensoes
        self.num_trabalhadores = num_trabalhadores
        self.batch_size = batch_size


    def setup(self, stage):
        """Gerar os datasets de treino e validação"""

        # ========== Treino ==========
        # ===== Interior =====
        pontos_interior = torch.rand((self.np_interior, self.dimensoes))
        self.dataset_interior = TensorDataset(pontos_interior)

        # ===== Contorno =====
        pontos_contorno = torch.zeros((self.np_contorno, self.dimensoes))
        pontos_contorno[:, 0] = torch.randint(0, 2, (self.np_contorno,))
        pontos_contorno[:, 1] = torch.rand(self.np_contorno)
        self.dataset_contorno = TensorDataset(pontos_contorno)

        # ===== Condição inicial x[0, L], t = 0 =====
        pontos_inicio = torch.zeros((self.np_inicio, self.dimensoes))
        pontos_inicio[:, 0] = torch.rand(self.np_inicio)
        self.dataset_inicio = TensorDataset(pontos_inicio)


        #  ========== Validação ==========
        # ===== Interior =====
        pontos_interior_val = torch.rand((self.np_interior_val, self.dimensoes))
        self.dataset_interior_val = TensorDataset(pontos_interior_val)

        # ===== Contorno =====
        pontos_contorno_val = torch.zeros((self.np_contorno_val, self.dimensoes))
        pontos_contorno_val[:, 0] = torch.randint(0, 2, (self.np_contorno_val,))
        pontos_contorno_val[:, 1] = torch.rand(self.np_contorno_val)
        self.dataset_contorno_val = TensorDataset(pontos_contorno_val)

        # ===== Condição inicial x[0, L], t = 0 =====
        pontos_inicio_val = torch.zeros((self.np_inicio_val, self.dimensoes))
        pontos_inicio_val[:, 0] = torch.rand(self.np_inicio_val)
        self.dataset_inicio_val = TensorDataset(pontos_inicio_val)

    def train_dataloader(self,):
        """Detorna o objeto CombinedLoader que será usado para treinar a rede"""
        
        loader_interior = DataLoader(
            self.dataset_interior,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=True,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )

        loader_contorno = DataLoader(
            self.dataset_contorno,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=True,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )
        
        loader_inicio = DataLoader(
            self.dataset_inicio,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=True,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )

        loaders = {
            'interior': loader_interior,
            'contorno': loader_contorno,
            'inicio': loader_inicio
        }

        return CombinedLoader(loaders, mode='min_size')

    def val_dataloader(self,):
        """Detorna o objeto CombinedLoader que será usado para validar a rede"""
        
        loader_interior_val = DataLoader(
            self.dataset_interior_val,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=False,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )

        loader_contorno_val = DataLoader(
            self.dataset_contorno_val,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=False,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )
        
        loader_inicio_val = DataLoader(
            self.dataset_inicio_val,
            batch_size=self.batch_size,
            num_workers=self.num_trabalhadores,
            shuffle=False,
            persistent_workers=DataModule.trabalhadores_persistentes,
            pin_memory=True
        )

        loaders = {
            'interior': loader_interior_val,
            'contorno': loader_contorno_val,
            'inicio': loader_inicio_val
        }

        return CombinedLoader(loaders, mode='min_size')