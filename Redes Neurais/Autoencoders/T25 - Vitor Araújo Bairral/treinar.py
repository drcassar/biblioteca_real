# Este código treina um autoencoder sem regularização
import multiprocessing
multiprocessing.freeze_support()
import leitor_imagem
import lightning as L
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F
import numpy as np
import sys
import rede_neural
if __name__ == "__main__":
    args = sys.argv[1:]

    argumentos = {"image": args[0],
                "denoise": False,
                "penalty": "",
                "factor": 1e-4,
                "patience": 5,
                "lr": 1e-3,
                "epochs": 500,
                "decay": 0.5,
                "depth": 3,
                "device": "cpu",
                "cores": 0,
                "fit": True}
    for i in range(len(args)):
        # -d Checa se é denoise
        if args[i] == "-d":
            argumentos["denoise"] = True
            argumentos["penalty"] = "d"
        # -l1, -kl, -c Checa penalidade
        if args[i] == "-l1" or args[i] == "-c":
            argumentos["penalty"] = args[i][1:]
            try:
                argumentos["factor"] = float(args[i+1])
            except:
                print("Fator de penalidade inválido. Usando valor padrão de 1e-4")
        # -p Checa paciencia
        if args[i] == "-p":
            try:
                argumentos["patience"] = int(args[i+1])
            except:
                print("Fator de paciênica inválido. Usando valor padrão de 5")
        if args[i] == "-e":
            try:
                argumentos["epochs"] = int(args[i+1])
            except:
                print("Número de épocas inválido. Usando valor padrão de 500")
        if args[i] == "-lr":
            try:
                argumentos["lr"] = float(args[i+1])
            except:
                print("Taxa de aprendizado inválida. Usando valor padrão de 1e-3")
        if args[i] == "-n":
            assert int(args[i+1]) % 2 == 0, "O número de camadas deve ser par."
            try:
                argumentos["depth"] = int(args[i+1])/2
            except:
                print("Número de camadas inválido. Usando valor padrão de 6")
        if args[i] == "-f": 
            try:
                assert float(args[i+1]) < 1 and float(args[i+1]) > 0, "O fator de compressão deve ser um float dentro de ]0;1[."
                argumentos["decay"] = float(args[i+1])
            except:
                print("Número de épocas inválido. Usando valor padrão de 0.5")
        if args[i] == "-gpu":
            argumentos["device"] = "gpu"
            argumentos["cores"] = int(args[i+1])
        
        elif args[i] == "-cpu":
            argumentos["device"] = "cpu"
            argumentos["cores"] = int(args[i+1])
        
        if args[i] == "-nf":
            argumentos["fit"] = False


    # Carregando os dados
    imagem, shape, n_pixels = leitor_imagem.read_image(argumentos["image"])
    n_pixels = n_pixels[0]
    print("-"*20, "INICIANDO TREINADOR DE REDES NEURAIS", "-"*20)
    print("TIPO DE REDE NEURAL: {}".format("Denoise" if argumentos["penalty"] == "d" else "Esparso" if argumentos['penalty'] == "l1" else "Contrátil" if argumentos["penalty"]=="c" else "Incompleto"))

    
    copia_imagem = imagem.copy()
    imagem = imagem.reshape(1, -1)
    noised = np.array([])
    if argumentos["denoise"]:
        noised = leitor_imagem.ruido(copia_imagem)
        noised = noised.reshape(1, -1)
        nome = f"noised_{args[0]}.png"

        print(f"IMAGEM COM RUÍDO GERADA COM NOME: {nome}")
        leitor_imagem.rebuild(noised, shape, nome)


    # Treinando a rede neural:
    ## Checando callback:
    NUM_EPOCHS = argumentos["epochs"]

    if argumentos["patience"] == 0:
        treinador = L.Trainer(max_epochs=NUM_EPOCHS, accelerator=argumentos["device"], devices=argumentos["cores"])
        print("SEM PARADA ANTECIPADA")
    else:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=argumentos["patience"],
            mode='min',
            verbose=False   
        )
        print(f"PARADA ANTECIPADA COM {argumentos["patience"]} ÉPOCAS")
        
        treinador = L.Trainer(callbacks=[early_stopping], max_epochs=NUM_EPOCHS, accelerator=argumentos["device"], devices=1, num_sanity_val_steps=0)

    def treinar():
        arquitetura = []
        fundura = int(argumentos["depth"])
        decay = argumentos["decay"]

        # Preenchendo a arquitetura da rede neural:
        for i in range(fundura):
            arquitetura.append(int(int(n_pixels) * decay**(i)))

        print(f"NÚMERO DE CAMADAS: {fundura} \nNEURÔNIOS POR CAMADA: {arquitetura} \nNEURÔNIOS TOTAIS: {sum(arquitetura)} \n")
        

        autoencoder = rede_neural.Autoencoder(arquitetura, 
                                            nn.Sigmoid(), 
                                            nn.MSELoss(), 
                                            penalty=argumentos["penalty"], 
                                            factor=argumentos["factor"])
        dm = rede_neural.DataModule(imagem, noised)
        if argumentos["fit"] == True:
            print("-"*20, "INICIANDO TREINAMENTO")
            treinador.fit(autoencoder, dm)
                    
            # Plotando e salvando a curva de aprendizado:
            ca_treino = autoencoder.curva_aprendizado_treino
            ca_val = autoencoder.curva_aprendizado_val

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharex=True)
            ax1.plot(ca_treino, "r")
            ax1.set_title("Curva de aprendizado do treino")
            ax1.set_xlabel("Épocas")
            ax1.set_ylabel("Loss")

            ax2.plot(ca_val, "b")
            ax2.set_title("Curva de aprendizado da validação")
            ax2.set_xlabel("Épocas")
            ax2.set_ylabel("Loss")
            name = f"{argumentos["image"]}_{argumentos["penalty"]}"

            plt.savefig(f"{name}.png")
            save = f"{name}.pth"
            torch.save(autoencoder.state_dict(), save)
        return autoencoder, dm
    autoencoder, dm = treinar()


