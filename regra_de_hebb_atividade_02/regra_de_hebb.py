## ALUNO: Vítor Carvalho Marx Lima
## MATRÍCULA: 11821ECP015
# DATA DE CRIAÇÃO: 15/08/2023
# DESCRIÇÃO: Implementação da regra de Hebb para treinamento de neurônios artificiais

class HebbNeuron:
    def __init__(self, weight_01, weight_02, bias):
        self.weight_01 = 0
        self.weight_02 = 0
        self.bias = 0


    def train(self, input_01, input_02, output):
        input_01, input_02, output = convert_inputs(input_01, input_02, output)

        self.weight_01 += input_01 * output
        self.weight_02 += input_02 * output
        self.bias += output

        print(f"input_01: {input_01} | input_02: {input_02} | output: {output} | weight_01: {self.weight_01} | weight_02: {self.weight_02} | bias: {self.bias}")

    def train_not(self, input_01, output):
        input_01, input_02, output = convert_inputs(input_01, 0, output)

        self.weight_01 += input_01 * output
        self.bias += output


    def predict(self, input_01, input_02):
        input_01, input_02, output = convert_inputs(input_01, input_02, 0)
        return 1 if input_01 * self.weight_01 + input_02 * self.weight_02 + self.bias >= 0 else 0


def convert_inputs(input_01, input_02, output):
    if input_01 <= 0:
        input_01 = -1
    if input_02 <= 0:
        input_02 = -1
    if output <= 0:
        output = -1

    return input_01, input_02, output


def train_and_test_function(table_name, truth_table, one_entry_flag=False, num_of_entries=4):
    print(f"TESTE PARA A FUNÇÃO: {table_name}")
    or_neuron = HebbNeuron(0, 0, 0)
    if one_entry_flag == 1:
        for i in range(0, num_of_entries):
            or_neuron.train_not(truth_table[i][0], truth_table[i][1])
    else:
        for i in range(0, num_of_entries):
            or_neuron.train(truth_table[i][0], truth_table[i][1], truth_table[i][2])

    print(f"{table_name} Neuron weights and bias: w01={or_neuron.weight_01} | w02={or_neuron.weight_02} | b={or_neuron.bias}:")

    flag_prediction = False

    if one_entry_flag == 1:
        for i in range(0, num_of_entries):
            prediction = or_neuron.predict(truth_table[i][0], 0)
            print(f"Input: {truth_table[i][0]} | Output: {prediction}")
            if prediction != truth_table[i][1]:
                flag_prediction = True
    else:
        for i in range(0, num_of_entries):
            prediction = or_neuron.predict(truth_table[i][0], truth_table[i][1])
            print(f"Input: {truth_table[i][0]} {truth_table[i][1]} | Output: {prediction}")
            if prediction != truth_table[i][2]:
                flag_prediction = True

    if flag_prediction == False:
        print("Tudo OK!\n")
    else:
        print("ERRO! Tabela verdade com erro!\n")
            
    return or_neuron.weight_01, or_neuron.weight_02, or_neuron.bias


def main():
    # Or truth table with inputs and output matrix
    identity_a_thruth_table = [[0, 0], [1, 1]]                                # IDENTITY A
    identity_b_thruth_table = [[0, 0], [1, 1]]                                # IDENTITY B
    not_A_thruth_table = [[1, 0]]                                             # NOT A
    not_B_thruth_table = [[0, 1]]                                             # NOT B
    and_thruth_table = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1,1,1]]             # AND
    A_and_not_B_truth_table = [[0, 0, 0], [0, 1, 1], [1, 0, 0], [1,1,0]]      # A AND NOT B
    not_A_and_B_truth_table = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1,1,0]]      # NOT A AND B
    not_A_and_not_B_truth_table = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1,1,0]]  # NOT A AND NOT B
    or_thruth_table = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1,1,1]]              # OR
    A_or_not_B_truth_table = [[0, 0, 1], [0, 1, 0], [1, 0, 1], [1,1,1]]       # A OR NOT B
    not_A_or_B_truth_table = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1,1,0]]       # NOT A OR B
    not_A_or_not_B_truth_table = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1,1,0]]   # NOT A OR NOT B
    xor_thruth_table = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1,1,0]]             # XOR
    xnor_thruth_table = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1,1,1]]            # XNOR
    if_A_than_B_truth_table = [[0, 0, 1], [0, 1, 1], [1, 0, 0], [1,1,1]]      # IF A THAN B
    if_B_than_A_truth_table = [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1,1,1]]      # IF B THAN A


    train_and_test_function("IDENTITY A", identity_a_thruth_table, one_entry_flag=True, num_of_entries=2)               ## OK
    train_and_test_function("IDENTITY B", identity_b_thruth_table, one_entry_flag=True, num_of_entries=2)               ## OK
    train_and_test_function("NOT A", not_A_thruth_table, one_entry_flag=True, num_of_entries=1)                         ## OK
    train_and_test_function("NOT B", not_B_thruth_table, one_entry_flag=True, num_of_entries=1)                         ## OK
    train_and_test_function("AND", and_thruth_table, one_entry_flag=False, num_of_entries=4)                            ## OK
    train_and_test_function("A AND NOT B", A_and_not_B_truth_table, one_entry_flag=False, num_of_entries=4)             ## OK
    train_and_test_function("NOT A AND B", not_A_and_B_truth_table, one_entry_flag=False, num_of_entries=4)             ## OK
    train_and_test_function("NOT A AND NOT B", not_A_and_not_B_truth_table, one_entry_flag=False, num_of_entries=4)     ## OK
    train_and_test_function("OR", or_thruth_table, one_entry_flag=False, num_of_entries=4)                              ## OK
    train_and_test_function("A OR NOT B", A_or_not_B_truth_table, one_entry_flag=False, num_of_entries=4)               ## OK
    train_and_test_function("NOT A OR B", not_A_or_B_truth_table, one_entry_flag=False, num_of_entries=4)               ## OK
    train_and_test_function("NOT A OR NOT B", not_A_or_not_B_truth_table, one_entry_flag=False, num_of_entries=4)       ## OK
    train_and_test_function("XOR", xor_thruth_table, one_entry_flag=False, num_of_entries=4)                            ## ERRO
    train_and_test_function("XNOR", xnor_thruth_table, one_entry_flag=False, num_of_entries=4)                          ## ERRO
    train_and_test_function("IF A THAN B", if_A_than_B_truth_table, one_entry_flag=False, num_of_entries=4)             ## OK
    train_and_test_function("IF B THAN A", if_B_than_A_truth_table, one_entry_flag=False, num_of_entries=4)             ## OK


if __name__ == "__main__":
    main()


