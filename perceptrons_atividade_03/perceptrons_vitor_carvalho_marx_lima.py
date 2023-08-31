## Aluno: Vitor Carvalho Marx Lima
## Matrícula: 11821ECP015
## Data: 23/08/2023
## Disciplina: Aprendizagem de Máquina

from PIL import Image

## FUNÇÃO UTILIZADA PARA IMPRIMIR OS PIXELS DA IMAGEM
def turn_image_into_array(im_name, img_extension):
    im = Image.open(im_name+img_extension)
    size = im.size 
    pix = im.load()
    x = size[0]
    y = size[1]
    pix = im.load()
    bg_color = pix[0, 0]
    img_array = []
    for i in range(x):
        for j in range(y):
            if pix[i, j] != bg_color:
                img_array.append(1)
            else:
                img_array.append(-1)

    return img_array

## DEFININDO CLASSE PERCEPTRON
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
        
## neurônio 01 e kanji 01, output 1;neurônio 01 e kanji 02, output 02; neurônio 02 e kanji 01, output -1; neurônio 02 e kanji 02, output -1

def predict_character(neuron_1, neuron_2, inputs):
        y1 = neuron_1.predict(inputs)
        y2 = neuron_2.predict(inputs)
        if y1 == 1 and y2 == -1:
            return 'kanji 1: 鬱 -> うつ (depressão)'
        elif y1 == -1 and y2 == 1:
            return 'kanji 2: 幸 -> さいわい (felicidade)'


def main():

    img_array_1 = turn_image_into_array(im_name='kanji_1_modified', img_extension='.png')
    img_array_2 = turn_image_into_array(im_name='kanji_2_modified', img_extension='.png')

    zero_weights = [0 for i in range(len(img_array_1))]

    neuron_1 = Perceptron(weights=zero_weights, bias=0)
    neuron_2 = Perceptron(weights=zero_weights, bias=0)

    neuron_1.train(inputs=img_array_1, output=1)
    neuron_1.train(inputs=img_array_2, output=-1)
    neuron_2.train(inputs=img_array_1, output=-1)
    neuron_2.train(inputs=img_array_2, output=1)

    print(f"Teste imagem 01 (kanji 01): {predict_character(neuron_1, neuron_2, inputs=img_array_1)}")
    print(f"Teste imagem 02 (kanji 02): {predict_character(neuron_1, neuron_2, inputs=img_array_2)}")


if __name__ == '__main__':
    main()