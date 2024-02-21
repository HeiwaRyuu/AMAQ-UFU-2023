# Aluno : Vítor Carvalho Marx Lima
# Matricula : 11821ECP015
# Disciplina : Aprendizagem de Maquina

import numpy as np
import random as rd
import matplotlib.pyplot as plt


def main():
    # Data Initialization
    x = np.array([[i] for i in np.arange(0, 1.1, 0.1)])  # Input reshaped for consistency
    t = np.array([-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000])  # Target
    amostras = len(x)  # Number of samples

    # Neural Network Parameters
    entradas = 1
    neur = 200
    vsai = 1  # Number of output neurons
    alfa = 0.005
    errotolerado = 0.05
    max_epochs = 10000  # Maximum number of iterations to prevent infinite loop

    # Weight Initialization
    vanterior = np.random.uniform(-1, 1, (entradas, neur))
    v0anterior = np.random.uniform(-1, 1, (neur))
    wanterior = np.random.uniform(-0.2, 0.2, (neur, vsai))
    w0anterior = np.random.uniform(-0.2, 0.2, (vsai))

    listaciclo = []
    listaerro = []

    # Training Loop
    ciclo = 0
    errototal = float('inf')

    while errotolerado < errototal and ciclo < max_epochs:
        errototal = 0
        for padrao in range(amostras):
            # Forward pass
            zin = np.dot(x[padrao], vanterior) + v0anterior
            z = np.tanh(zin)
            yin = np.dot(z, wanterior) + w0anterior
            y = np.tanh(yin)

            # Compute error
            erro = t[padrao] - y
            errototal += np.sum(0.5 * (erro ** 2))

            # Backpropagation
            delta_k = erro * (1 - y ** 2)
            delta_k = delta_k.reshape(1, -1)  # Ensure delta_k is 2D

            delta_j = (1 - z ** 2) * (delta_k @ wanterior.T)
            
            # Weight updates
            wanterior += alfa * z.reshape(-1, 1) @ delta_k  # z reshaped for correct dimensionality
            w0anterior += alfa * delta_k.flatten()  # Ensure scalar addition

            delta_j = delta_j.flatten()  # Flatten delta_j for the outer product
            vanterior += alfa * np.outer(x[padrao], delta_j)  # np.outer for correct dimensions
            v0anterior += alfa * delta_j  # Direct addition

        listaciclo.append(ciclo)
        listaerro.append(errototal)
        if ciclo % 1000 == 0:
            print(f'Epoch {ciclo}: Total Error = {errototal}')
        ciclo += 1

    # Plotting the error over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(listaciclo, listaerro, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')
    plt.title('Error Progression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()