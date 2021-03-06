

# Matthew O'Brien
# CS 445 Homework 2
# Neural Network on letter recognition



import numpy as np
import random
import math
import matplotlib.pyplot as plt
learning_rate = 0.3
momentum = 0.3
hidden_units = 8
number_epochs = 50


# Perceptron class that will be trained for the data
class Outer_Percep:

    # Initializing data
    def __init__(self, letter):
        self.letter = letter
        self.weights = self.randomweights()
        self.bias = random.uniform(-.25,.25)
        self.accuracy = 0.0
        self.previous_weight_change = [0] * hidden_units
        self.weight_change = [0] * hidden_units
        self.bias_change = 0
        self.prev_bias = 0

    # Defining the random weights of each layer.
    def randomweights(self):
        weights = []
        for i in range(hidden_units):
            weights.append(random.uniform(-.25,.25))
        return weights

    # Calculating the output values of each node.
    def activation_output(self,test_set):

        output_value = np.dot(self.weights,test_set) + self.bias
        output_value = sigmoid_function(output_value)

        return output_value

    # The Back Prop function for the outer nodes.
    def back_prop(self,out_value,loss):

        for i in range(hidden_units):
            self.weight_change[i] = learning_rate*loss*(out_value[i])
        for i in range(hidden_units):
            self.weights[i] = self.weights[i] + self.weight_change[i] + (momentum * self.previous_weight_change[i])
        for i in range(hidden_units):
            self.previous_weight_change[i] = self.weight_change[i]

        self.bias_change = learning_rate*loss*1
        self.bias = self.bias + self.bias_change + (momentum*self.prev_bias)
        self.prev_bias = self.bias_change
        # self.bias = self.bias + (momentum)*(learning_rate)*(loss)

# Inner set of perceptron nodes
class Inner_Percep:

    def __init__(self):
        self.weights = self.randomw()
        self.bias = random.uniform(-.25,.25)
        self.accuracy = 0.0
        self.previous_weight_change = [0] * 16
        self.weight_change = [0] * 16
        self.bias_change = 0
        self.prev_bias = 0

    # Randomizing the weights of the nodes
    def randomw(self):
        weights = []
        for i in range(16):
            weights.append(random.uniform(-.25,.25))
        return weights

    # Calculating the activation of the hidden node set
    def activation_hidden(self,test_set):

        input_value = test_set[1:]
        output_value = np.dot(self.weights,input_value) + self.bias
        output_value = sigmoid_function(output_value)

        return output_value

    # The Back Pro for the hidden node set
    def back_prop(self,out_value, loss):

        for i in range(16):
            self.weight_change[i] = learning_rate*loss*out_value[i]
        for i in range(16):
            self.weights[i] = self.weights[i] + self.weight_change[i] + (momentum * self.previous_weight_change[i])
        for i in range(16):
            self.previous_weight_change[i] = self.weight_change[i]

        self.bias_change = learning_rate*loss*1
        self.bias = self.bias + self.bias_change + (momentum*self.prev_bias)
        self.prev_bias = self.bias_change
        # self.bias = self.bias + (momentum)*(learning_rate)*(loss)


def train_function(inner_layer, out_layer, test_set):

    # Temp values used for storing activations, expected values, and loss terms
    loss_list = []
    inner_loss_list = []
    expected_outputs = []
    inner_delta_list = []
    sigma_value = 0
    to_test = test_set[1:]

    # Expected array for the training set
    expected_letter = test_set.astype(int)[0] - 1
    for i in range(26):
        expected_outputs.append(.1)
    expected_outputs[expected_letter] = 0.9

    # More temps for the activations of the nodes, stored into arrays.
    inner_activation = []
    outer_activation = []

    # activation value for inner layer
    for i in range(len(inner_layer)):
        inner_activation.append(inner_layer[i].activation_hidden(test_set))

    # activation value for outer layer
    for i in range(len(out_layer)):
        outer_activation.append(out_layer[i].activation_output(inner_activation))

    # loss terms for outer layer
    for i in range(26):
        loss = (outer_activation[i])*((1 - outer_activation[i])*(expected_outputs[i]-outer_activation[i]))
        loss_list.append(loss)

    # back prop on outer layer weights
    for i in range(26):
        out_layer[i].back_prop(inner_activation,loss_list[i])

    # Sigma term calculations
    for j in range(hidden_units):
        for i in range(len(loss_list)):
            sigma_value = sigma_value + out_layer[i].weights[j] * loss_list[i]
        inner_loss_list.append(sigma_value)

    # Calculating the inner loss term
    for i in range(hidden_units):
        inner_delta = (inner_activation[i])*(1-inner_activation[i])*(inner_loss_list[i])
        inner_delta_list.append(inner_delta)

    # Back prop on inner layer weights
    for i in range(hidden_units):
        inner_layer[i].back_prop(to_test,inner_delta_list[i])

    # Figuring out the guessed letter from the outer layer
    net_guess = np.argmax(outer_activation)
    net_guess += 1

    return net_guess


# Testing function. Same as training minus the back prop
def test_net(testing_data, out_layer, hidden_layer):

    # Temps for activations of each layer
    inner_activation = []
    outer_activation = []

    # activation value for inner layer
    for i in range(len(hidden_layer)):
        inner_activation.append(hidden_layer[i].activation_hidden(testing_data))

    # activation value for outer layer
    for i in range(len(out_layer)):
        outer_activation.append(out_layer[i].activation_output(inner_activation))

    # The returned guess from the net
    net_guess = np.argmax(outer_activation)
    net_guess += 1
    return net_guess


# Creating the outer layer of perceptrons
def create_output():

    perceptron_list = []
    for i in range(1,27):
        perceptron_list.append(Outer_Percep(i))
    return perceptron_list


# Creating the inner layer of perceptrons
def create_hidden():

    perceptron_list = []

    for i in range(hidden_units):
        perceptron_list.append(Inner_Percep())
    return perceptron_list


# The defined sigmoid function
def sigmoid_function(x):
        x = 1 / (1 + math.exp(-x))
        return x


# Making the numpy array from the data
def make_training_data(textfile):

    # Formatting for the array from the text files
    temp_a_data = np.genfromtxt(textfile, delimiter=',', dtype = 'O')
    # Loop to traverse the file
    for i in range(len(temp_a_data)):
        # Converts the first letter of each line into an integer. Required for the numpy array
        temp_a_data[i,0] = ord(temp_a_data[i,0].lower()) - 96.
    # Makes the data numpy floats
    array_a_data = temp_a_data.astype(np.float32)

    # Divide data normalization function
    # array_a_data[:,1:] = array_a_data[:, 1:] / float(15.0)
    for i in range(16):
        mu = np.mean(array_a_data[:, i+1])
        std = np.std(array_a_data[:, i+1])
        array_a_data[:, i+1] = (array_a_data[:, i+1] - mu) / std
    # Gives back the new numpy array of floats.
    return array_a_data


# Main function for testing
def main():

    # Create the training and testing data
    training_data = make_training_data("letter_data_halves1.txt")
    testing_data = make_training_data("letter_data_halves2.txt")

    # Create the layers of the network
    out_layer = create_output()
    hidden_layer = create_hidden()
    test_accuracy_report = []
    training_accyracy_report = []
    epochs_ran = []

    # Loop for the number of epochs
    for j in range(number_epochs):
        count = 0
        counter = 0

        # Training the neural net
        for i in range(len(training_data)):
            train_function(hidden_layer,out_layer,training_data[i])

        # Running the test function on the original training data after training
        for i in range(len(training_data)):
            expected_out = test_net(training_data[i], out_layer, hidden_layer)
            if expected_out == training_data[i, 0]:
                counter += 1
        training_accuracy = float(counter/len(training_data)*100)
        # Report the accuracy
        print("Accuracy on training data for Epoch number:", j+1,"Accuracy:",training_accuracy)
        training_accyracy_report.append(training_accuracy)
        # Running the test function of the test data after the training
        for i in range(len(testing_data)):
            expected_out = test_net(testing_data[i], out_layer, hidden_layer)
            if expected_out == testing_data[i, 0]:
                count += 1

        accuracy = float(count/len(testing_data)*100)
        # Report the accuracy
        print("Accuracy on testing data for Epoch number:", j+1,"Accuracy:",accuracy)
        test_accuracy_report.append(accuracy)
        np.random.shuffle(training_data)
        epochs_ran.append(j+1)

    # The basic graphing tools used for the data analysis
    # plt.plot(epochs_ran,test_accuracy_report, 'bs',epochs_ran,training_accuracy_report, 'g^')
    plt.plot(epochs_ran,test_accuracy_report, epochs_ran, training_accyracy_report)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.axis([1,50,0,80])
    plt.show()
main()
