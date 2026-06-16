import os
import sys
import logging

# ========== Silenciar Lightning ==========
os.environ['LIGHTNING_DISABLE_INFO'] = '1'
os.environ['LIGHTNING_SILENCE_WARNINGS'] = '1'
os.environ['PL_DISABLE_SLURM'] = '1'


# ========== Formatter ==========
class TimeColorFormatter(logging.Formatter):

    def format(self, record):
        # Tempo verde
        time_colored = f'\033[92m{self.formatTime(record)}\033[0m'

        levelname = record.levelname
        message = record.getMessage()

        return f'{time_colored} [{levelname}] {message}'


# ========== Logger ==========
logger = logging.getLogger('PINN')
logger.setLevel(logging.INFO) # Nível
# Evita duplicação de handlers
logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout) # Envia para o terminal como stdout
formatter = TimeColorFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
# Evita propagação para root logger
logger.propagate = False



import time
import lightning.pytorch as pl
from datetime import datetime, timedelta


class TrialLoggingCallback(pl.Callback):

    def __init__(self, trial, print_every=10, logger=logger):
        self.trial = trial
        self.logger = logger
        self.print_every = print_every

        self.tempos_epoca = []
        self.janela_suavizacao = 5


    def on_fit_start(self, trainer, pl_module):

        self.start_time = time.time()
        self.last_log_time = time.perf_counter()

        self.logger.info('='*25)
        self.logger.info(f'Iniciando Trial {self.trial.number}')

        self.logger.info('Parâmetros:')

        for k, v in self.trial.params.items():
            self.logger.info(f'  {k}: {v}')

        self.logger.info('='*25)

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.print_every != 0:
            return
        
        metrics = trainer.callback_metrics
        
        # ========== Tempo ==========
        agora = time.perf_counter()
        delta_x_epocas = agora - self.last_log_time
        tempo_por_epoca = delta_x_epocas / self.print_every
        # ===== Histórico
        self.tempos_epoca.append(tempo_por_epoca)
        self.last_log_time = agora

        # ========== Tempo restante ==========
        max_epochs = trainer.max_epochs
        epocas_restantes = max_epochs - epoch
        janela = self.tempos_epoca[-self.janela_suavizacao:]
        tempo_por_epoca_suavizado = sum(janela) / len(janela)
        segundos_restantes = epocas_restantes * tempo_por_epoca_suavizado
        horario_termino = (datetime.now() +timedelta(seconds=segundos_restantes))

        horario_termino = horario_termino.strftime('%H:%M:%S')
        tempo_restante = str(timedelta(seconds=int(segundos_restantes)))

        msg = f'[Trial {self.trial.number}] Epoch {epoch:04d}'
        msg += f' | val_loss: {metrics['val_loss'].item():.5e}'
        msg += f' | Tempo médio por época: {tempo_por_epoca_suavizado:.2f} s -- ({delta_x_epocas:.2f} s - {self.print_every} épocas)'
        self.logger.info(msg)

        msg_restante = f'Término às: {horario_termino}'
        msg_restante += f' | Tempo restante: {tempo_restante}'
        self.logger.info(msg_restante)

    def on_fit_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time

        metrics = trainer.callback_metrics

        self.logger.info('-'*25)
        self.logger.info(f'Trial {self.trial.number} finalizado.')
        self.logger.info(f'    val_loss: {metrics['val_loss'].item():.5e}')

        estudo = self.trial.study

        self.logger.info('')
        self.logger.info('Melhor trial até agora:')
        
        try:
            self.logger.info(f'    Trial {estudo.best_trial.number}')
            self.logger.info(f'        Loss  {estudo.best_value:.5e}')
        
        except:
            self.logger.info('...')

        self.logger.info(f'\n Tempo no trial: {elapsed/60:.2f} min')
        self.logger.info('-'*25)