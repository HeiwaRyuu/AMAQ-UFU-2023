
## Aluno: Vitor Carvalho Marx Lima
## Matrícula: 11821ECP015
## Data: 20/09/2023
## Disciplina: Aprendizagem de Máquina

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Adaline:
    def __init__(self, alpha=0.01, n_iterations=200, theta=0.01):
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.theta = theta


    def init_weights(self, X):
        self.weights = 0.5 - random.random()
        print("self.weights: ", self.weights)
        self.bant = 0.5
    

    def calcualte_y(self, X, index):
        y = 0
        y += self.weights*X[index]

        y += self.bant

        return y
    
    
    def update_weights(self, X, t, y, index):
        self.weights = self.weights + self.alpha*(t[index]-y)*X[index]
        self.bant = self.bant + self.alpha*(t[index]-y)


    def training(self, X, t):
        print("LEN X: ", len(X))
        print("LEN t: ", len(t))
        self.init_weights(X)
        erroquadratico = 0
        while self.n_iterations > 0: ##training limit
            for i in range(len(X)):
                ## activation function
                y = self.calcualte_y(X, i)
                ## quadratic error
                print(f"t[{i}] = {t[i]} | y = {y}")
                erroquadratico = erroquadratico + (t[i]-y)**2
                ## update weights
                self.update_weights(X, t, y, i)

            self.n_iterations -= 1


    def calculate_error(self, t, y):
        error = abs(t-y)

        return error

    
    def test(self, X, t):
        results = []
        for i in range(len(X)):
            y = self.calcualte_y(X, i)
            results.append([X[i], y])
            print(f"t({i}) = {t[i]} y({i}) = {y}")
            error = self.calculate_error(t[i], y)
            if error <= self.theta:
                print("Nice!")
                print("error: ", error)
                print("theta: ", self.theta)
            else:
                print("Too far!")
                print("error: ", error)
                print("theta: ", self.theta)
        
        return results



def clean_read_data(element_pair):
    chars_to_remove = []
    last_char = ''
    for i, char in enumerate(element_pair):
        if i == 0 and char == ' ':
            chars_to_remove.append(i)
        elif char == ' ' and last_char == ' ':
            chars_to_remove.append(i)
        elif i == len(element_pair)-1 and char == ' ': 
            chars_to_remove.append(i)

        last_char = char


    count = 0
    for i, index in enumerate(chars_to_remove):
        if i == 0:
            element_pair = element_pair[:index] + element_pair[index+1:]
            count += 1
        else:
            element_pair = element_pair[:index-count] + element_pair[index+1-count:]
            count += 1

    return element_pair


def read_observations():
    ## READ CONTENTS FROM A TXT NAMED "basedeobservacoes.txt"
    with open('basedeobservacoes.txt', 'r') as file:
        unparsed_data = file.read().splitlines()
        unparsed_data.pop(0)

    X = []
    y = []

    for element_pair in unparsed_data:
        element_pair = clean_read_data(element_pair)
        element_pair = element_pair.split(' ')
        element_pair = [float(element) for element in element_pair]

        X.append(element_pair[0])
        y.append(element_pair[1])

    return X, y


def test_split(X, y, test_size=0.2):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    already_in_test = []

    test_size = int(round(len(X)*test_size, 0))
    
    
    for i in range(test_size):
        if i == 0:
            test_index = np.random.randint(0, len(X))
            X_test.append(X[test_index])
            y_test.append(y[test_index])
            already_in_test.append(test_index)

        test_index = np.random.randint(0, len(X))
        while test_index in already_in_test:
            test_index = np.random.randint(0, len(X))

        X_test.append(X[test_index])
        y_test.append(y[test_index])

    X_train = [element for element in X if element not in X_test]
    y_train = [element for element in y if element not in y_test]
    
    return X_train, X_test, y_train, y_test


## PLOT POIUNTS WITH LABELS
def plot_points(X_test, t_test, results, annotate=False):
    fig, ax = plt.subplots()
    ax.scatter(X_test, t_test, label='Points')
    ax.plot([element[0] for element in results], [element[1] for element in results])
    if annotate:
        point_labels = [(f"{round(X_test[i],1)}, {round(t_test[i],1)}") for i in range(len(X_test))]
        for i, txt in enumerate(point_labels):
            ax.annotate(txt, (X_test[i], t_test[i]))

    plt.legend()


def plot_linear_regression(X, t, results):
    # PLOT LINEAR REGRESSION
    lr = LinearRegression()
    lr.fit(np.array(X).reshape(-1,1), np.array(t).reshape(-1,1))
    y_pred = lr.predict(np.array(X).reshape(-1,1))
    plt.plot(X, y_pred, label='Linear Regression')
    plt.legend()


def main():
    ## READ FILE
    X, t = read_observations()

    ## DIVIDE INTO TRAIN AND TEST
    X_train, X_test, t_train, t_test = test_split(X, t, test_size=0.2)


    ## INSTANTIATE ADALINE
    adaline = Adaline(alpha=0.001, n_iterations=200)

    ## TRAINING
    adaline.training(X_train, t_train)

    ## TEST
    results = adaline.test(X_test, t_test)

    ## PLOT LINEAR REGRESSION
    plot_points(X_test, t_test, results, annotate=False)


    ## PLOT LINEAR REGRESSION
    plot_linear_regression(X, t, results)


    ## FIND PEARSON CORRELATION
    pearson_correlation = np.corrcoef(X, t)[0,1]
    print("Pearson Correlation: ", pearson_correlation)

    ## FIND DET COEFFICIENT
    corr_matrix = np.corrcoef(X, t)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    
    print("Det Coefficient: ", R_sq)


    plt.show()

if __name__ == "__main__":
    main()
    # read_observations()
    # clean_read_data("    10.343534   8.3096467 ")