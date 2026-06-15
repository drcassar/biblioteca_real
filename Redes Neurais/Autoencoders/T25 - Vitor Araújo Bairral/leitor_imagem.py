import numpy as np
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt

def read_image(filename):
    """Recebe como parâmetro o nome do arquivo de imagem
    retorna um vetor coluna com os dados da imagem."""

    matrix = imread(filename)
    shape = matrix.shape
    vector = matrix.flatten()
    n_pixels = vector.shape
    return vector, shape, n_pixels

def ruido(vector):
    noise = np.random.normal(0, 0.1, vector.shape)
    vetor_com_ruido = vector + noise 
    vetor_com_ruido = np.sqrt((vetor_com_ruido / max(vetor_com_ruido))**2)
    return vetor_com_ruido

def rebuild(vector, shape, outname="saida.png"):

    matrix = vector.reshape(shape)
    imsave(outname, matrix, cmap="gray", dpi=300)