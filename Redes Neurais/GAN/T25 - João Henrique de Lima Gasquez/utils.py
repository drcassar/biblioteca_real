import os
import numpy as np
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch


class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name

        self.comment = f'{model_name}_{data_name}'
        self.data_subdir = f'{model_name}/{data_name}'

        self.writer = SummaryWriter(log_dir=f"runs/{self.comment}")
        
    def log(self, d_error, g_error, epoch, n_batch, num_batches):

        d_error = self._to_numpy(d_error)
        g_error = self._to_numpy(g_error)

        step = self._step(epoch, n_batch, num_batches)

        self.writer.add_scalar(f'{self.comment}/D_error', d_error, step)
        self.writer.add_scalar(f'{self.comment}/G_error', g_error, step)

    def log_images(self, images, num_images, epoch, n_batch, num_batches,
                   format='NCHW', normalize=True):

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if format == 'NHWC':
            images = images.permute(0, 3, 1, 2)

        step = self._step(epoch, n_batch, num_batches)

        horizontal_grid = vutils.make_grid(images, normalize=normalize, scale_each=True)

        self.writer.add_image(f'{self.comment}/images', horizontal_grid, step)

    def save_epoch_images(self, images, epoch):

        out_dir = f'./dados/imagens/{self.data_subdir}'
        self._make_dir(out_dir)

        grid = vutils.make_grid(images, normalize=True, scale_each=True)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')

        fig.savefig(f'{out_dir}/epoch_{epoch}.png')
        plt.close()

    def display_images(self, images, num_images, epoch, n_batch, num_batches, normalize=True):
            grid = vutils.make_grid(images[:num_images], normalize=normalize, scale_each=True)
            
            grid_np = np.moveaxis(grid.cpu().numpy(), 0, -1)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(grid_np)
            plt.axis('off')
            plt.show()

    def display_status(self, epoch, num_epochs, n_batch, num_batches,
                       d_error, g_error, d_pred_real, d_pred_fake):

        d_error = self._to_numpy(d_error)
        g_error = self._to_numpy(g_error)

        d_pred_real = self._to_numpy(d_pred_real)
        d_pred_fake = self._to_numpy(d_pred_fake)

        print(f'Época: [{epoch}/{num_epochs}] Batch: [{n_batch}/{num_batches}]')
        print(f'D Loss: {d_error:.4f} | G Loss: {g_error:.4f}')
        print(f'D(x): {d_pred_real.mean():.4f} | D(G(z)): {d_pred_fake.mean():.4f}')

    def save_models(self, generator, discriminator, epoch, save_every=100):
        if epoch == 1 or epoch % save_every == 0:
            out_dir = f'./dados/modelos/{self.data_subdir}'
            self._make_dir(out_dir)
    
            torch.save(generator.state_dict(), f'{out_dir}/G_epoch_{epoch}.pt')
            torch.save(discriminator.state_dict(), f'{out_dir}/D_epoch_{epoch}.pt')
        else:
            return

    def close(self):
        self.writer.close()

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        os.makedirs(directory, exist_ok=True)