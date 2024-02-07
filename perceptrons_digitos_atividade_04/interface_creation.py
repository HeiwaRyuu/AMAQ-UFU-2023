import os
import tkinter as tk
import customtkinter as ctk
from perceptron_class import Perceptron

## GLOBAL VARIABLES
## CANVAS SETTINGS
GRID_SIZE = 5
CANVAS_WIDTH = 500
CANVAS_HEIGHT = 700
CANVA_START = 0
INITIAL_COLOR = "white"
SECONDARY_COLOR = "black"
XPADDING = 5
## BUTTON GRID SETTINGS
GRID_BTN_WIDTH = CANVAS_WIDTH/GRID_SIZE
GRID_BTN_HEIGHT = GRID_BTN_WIDTH
BTN_CORNER_RADIUS = 0
## FUNCTION BUTTONS
FUNCTION_BUTTON_WIDTH = 140
FUNCTION_BUTTON_HEIGHT = 50
FUNCTION_BUTTON_WIDTH_OFFSET = (CANVAS_WIDTH/2) - ((FUNCTION_BUTTON_WIDTH*2)/2)
FUNCTION_BUTTON_HEIGHT_OFFSET = FUNCTION_BUTTON_HEIGHT
FUNCTION_BUTTON_Y_POSITION = GRID_SIZE*GRID_BTN_HEIGHT+FUNCTION_BUTTON_HEIGHT_OFFSET
## MEANING LABEL
MEANING_LABEL_WIDTH = 140
MEANING_LABEL_HEIGHT = 20
MEANING_LABEL_WIDTH_OFFSET = (CANVAS_WIDTH/2) - (MEANING_LABEL_WIDTH*2/2)
MEANING_LABEL_HEIGHT_OFFSET = (FUNCTION_BUTTON_HEIGHT_OFFSET/2) - (MEANING_LABEL_HEIGHT/2)
MEANING_LABEL_Y_POSITION = GRID_SIZE*GRID_BTN_HEIGHT+MEANING_LABEL_HEIGHT_OFFSET
## PREDICTION LABEL
PREDICTION_LABEL_WIDTH = 140
PREDICTION_LABEL_HEIGHT = 20
PREDICTION_LABEL_WIDTH_OFFSET = (CANVAS_WIDTH/2) - (PREDICTION_LABEL_WIDTH*2/2)
PREDICTION_LABEL_HEIGHT_OFFSET = (FUNCTION_BUTTON_HEIGHT_OFFSET/2) - (PREDICTION_LABEL_HEIGHT/2)
PREDICTION_LABEL_Y_POSITION = FUNCTION_BUTTON_Y_POSITION+FUNCTION_BUTTON_HEIGHT+PREDICTION_LABEL_HEIGHT_OFFSET

## NEURONS SETTINGS
NUMBER_OF_NEURONS = 10


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}")
        self.title("Perceptron Digit Training")
        self.resizable(False, False)
        self.buttons = []
        self.neurons = []
        self.list_of_predictions = []
        self.current_neuron_count = 0
        self.trained_neurons = []

        ## ADDING UI ELEMENTS
        self.generate_button_grid()
        ## GENERATE PREDICTION FILE IF NOT EXISTS
        self.read_list_of_predictions()
        ## INITIALIZING THE NEURONS
        self.generate_neuron()
        ## MEANING LABELS
        self.lbl_meaning = ctk.CTkLabel(self, text="Meaning: ", width=MEANING_LABEL_WIDTH, height=MEANING_LABEL_HEIGHT)
        self.lbl_meaning.place(x=MEANING_LABEL_WIDTH_OFFSET, y=MEANING_LABEL_Y_POSITION)
        self.entry_meaning_value = ctk.CTkEntry(self, placeholder_text="CTkEntry", width=MEANING_LABEL_WIDTH, height=MEANING_LABEL_HEIGHT, fg_color=INITIAL_COLOR)
        self.entry_meaning_value.place(x=MEANING_LABEL_WIDTH_OFFSET+MEANING_LABEL_WIDTH, y=MEANING_LABEL_Y_POSITION)
        ## PREDICTION LABELS
        self.lbl_prediction = ctk.CTkLabel(self, text="Prediction: ", width=PREDICTION_LABEL_WIDTH, height=PREDICTION_LABEL_HEIGHT)
        self.lbl_prediction.place(x=PREDICTION_LABEL_WIDTH_OFFSET, y=PREDICTION_LABEL_Y_POSITION)
        self.lbl_prediction_value = ctk.CTkLabel(self, text="?", width=PREDICTION_LABEL_WIDTH, height=PREDICTION_LABEL_HEIGHT)
        self.lbl_prediction_value.place(x=PREDICTION_LABEL_WIDTH_OFFSET+PREDICTION_LABEL_WIDTH, y=PREDICTION_LABEL_Y_POSITION)
        ## ADDING THE BUTTONS
        self.btn_clear = ctk.CTkButton(self, text="Clear", command=self.clear_board, width=FUNCTION_BUTTON_WIDTH, height=FUNCTION_BUTTON_HEIGHT)
        self.btn_clear.place(x=FUNCTION_BUTTON_WIDTH_OFFSET-XPADDING, y=FUNCTION_BUTTON_Y_POSITION)
        self.btn_train = ctk.CTkButton(self, text="Train", command=self.train_neurons, width=FUNCTION_BUTTON_WIDTH, height=FUNCTION_BUTTON_HEIGHT)
        self.btn_train.place(x=FUNCTION_BUTTON_WIDTH_OFFSET+FUNCTION_BUTTON_WIDTH+XPADDING, y=FUNCTION_BUTTON_Y_POSITION)
        

    def generate_button_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.buttons.append(ctk.CTkButton(self, text="", width=GRID_BTN_WIDTH, height=GRID_BTN_HEIGHT, fg_color=INITIAL_COLOR, corner_radius=BTN_CORNER_RADIUS))
                self.buttons[-1].place(x=GRID_BTN_WIDTH*j, y=GRID_BTN_HEIGHT*i)
                self.buttons[-1].configure(command=lambda x=self.buttons[-1]:self.change_btn_color(x))


    def change_btn_color(self, btn):
        if btn.cget("fg_color") == INITIAL_COLOR:
            btn.configure(fg_color=SECONDARY_COLOR)
        else:
            btn.configure(fg_color=INITIAL_COLOR)

        self.predict()


    def clear_board(self):
        for btn in self.buttons:
            btn.configure(fg_color=INITIAL_COLOR)
        

    def generate_neuron(self):
        for i in range(NUMBER_OF_NEURONS):
            self.neurons.append(Perceptron(weights=[0 for i in range(GRID_SIZE*GRID_SIZE)], bias=0))
        print("Neurons generated!")


    def generate_color_vector(self):
        color_vector = []
        for btn in self.buttons:
            if btn.cget("fg_color") == INITIAL_COLOR:
                color_vector.append(-1)
            else:
                color_vector.append(1)

        return color_vector


    def train_neurons(self):
        print("Training...")
        color_vector = self.generate_color_vector()
        for i, neuron in enumerate(self.neurons):
            if self.current_neuron_count not in self.trained_neurons:
                if i == self.current_neuron_count:
                    neuron.train(inputs=color_vector, output=1)
                    self.trained_neurons.append(i)
                else:
                    neuron.train(inputs=color_vector, output=-1)
            else:
                print(f"Neuron {i} already trained!")

        if self.current_neuron_count == NUMBER_OF_NEURONS-1:
            self.current_neuron_count = 0
        else:
            self.current_neuron_count += 1

        if self.entry_meaning_value.get() == "":
            print("RETURNING...")
            return
        ## IF AND ITEM IS NOT IN THE LIST, INSERT, IF AN ITEM IS ALREADY IN THE LIST, UPDATE ITS MEANING
        is_in_prediction_list, prediction_list_index = self.check_if_in_prediction_list(color_vector=color_vector)
        if (not is_in_prediction_list):
            self.list_of_predictions.append([color_vector, self.entry_meaning_value.get()])
        else:
            self.list_of_predictions[prediction_list_index][1] = self.entry_meaning_value.get()
        print("color_vector: ", color_vector)
        print("list_of_predictions: ", self.list_of_predictions)

        self.predict()
        self.save_list_of_predictions()
    

    def read_list_of_predictions(self):
        if os.path.isfile('predictions.txt'):
            with open('predictions.txt', 'r') as f:
                prediction_str = f.readlines()
                for pred_str in prediction_str:
                    color_vect, value = pred_str.split(':')
                    color_vect = color_vect.strip('[]').split(', ') ## CONVERT STR LIST REPRESENTATION TO ACTUAL LIST
                    color_vect = [eval(i) for i in color_vect] ## CONVERT ELEMENTS OF THE LIST BACK TO INT
                    value = eval(value.split('\n')[0]) ## CONVERT VALUE TO INT
                    lst = [color_vect, value]
                    self.list_of_predictions.append(lst)
        else:
            with open('predictions.txt', 'w') as f: ## CREATES FILE IT NOT EXISTS
                pass


    def save_list_of_predictions(self):
        with open('predictions.txt', 'w+') as f:
            for prediction in self.list_of_predictions:
                pred_str = str(prediction[0])+':'+str(prediction[1])+'\n'
                f.write(pred_str)
        

    def check_if_in_prediction_list(self, color_vector):
        for i, item in enumerate(self.list_of_predictions):
            if item[0] == color_vector:
                return True, i
        return False, None
    

    def predict(self):
        print("Predicting...")
        color_vector = self.generate_color_vector()
        is_in_prediction_list, prediction_list_index = self.check_if_in_prediction_list(color_vector=color_vector)
        if is_in_prediction_list:
            print("is in prediction list")
            for i, neuron in enumerate(self.neurons):
                if (neuron.predict(inputs=color_vector) == 1):
                    self.lbl_prediction_value.configure(text=f"{self.list_of_predictions[prediction_list_index][1]}")
                    print(f"NEURON {i} PREDICTED {self.list_of_predictions[prediction_list_index][1]}")
                    break
                else:
                    self.lbl_prediction_value.configure(text="?")
        else:
            self.lbl_prediction_value.configure(text="?")

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()