## DEFININDO CLASSE PERCEPTRON // REVIEWED
class Perceptron:
    def __init__(self, weights, bias):
        self.weights = [0 for x in range(len(weights))]
        self.bias = 0
        self.learning_rate = 0.5


    def train(self, inputs, output):
        
        y = self.predict(inputs)

        for i in range(len(inputs)):
            self.weights[i] += inputs[i] * output * self.learning_rate
        self.bias += output * self.learning_rate

        if y!= output:
            self.train(inputs=inputs, output=output)
    

    def predict(self, inputs, threshold=0):
        y_res = 0
        for i in range(len(inputs)):
            y_res += inputs[i] * self.weights[i]
        y_res += self.bias
        
        return 1 if y_res >= threshold else -1