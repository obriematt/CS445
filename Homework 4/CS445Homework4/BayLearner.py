# Matthew O'Brien
# CS 445 Homework #4


import numpy as np
small_number = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001


# Processing and creating the test and train sets
def data_processing_function(datafile):
    # Load the data into a numpy array
    data = np.loadtxt('spambase.data', delimiter = ',')

    # Find the negative and positive data sets. Count them. Found the minimum of the two
    negative_data = data[data[:,-1] == 0]
    positive_data = data[data[:,-1] == 1]

    # Split the negative and positive values into two groups
    positive_one = positive_data[:len(positive_data)/2]
    positive_two = positive_data[len(positive_data)/2:]
    negative_one = negative_data[:len(negative_data)/2]
    negative_two = negative_data[len(negative_data)/2:]

    # Create Test and Train sets
    test_set = np.vstack((positive_one,negative_one))
    train_set = np.vstack((positive_two, negative_two))

    # Probability of the training set being spam or not spam
    probability_spam = np.mean(train_set[:,-1])
    probability_not_spam = 1 - probability_spam

    # Average negative and positive for the train set pieces
    mean_positive = np.mean(positive_two[:,:-1], axis=0, dtype=np.float64)
    mean_negative = np.mean(negative_two[:,:-1], axis=0, dtype=np.float64)

    # Standard Deviation of negative and positive pieces for the train set
    stdev_positive = np.std(positive_two[:,:-1], axis=0, dtype=np.float64, ddof=1)
    stdev_negative = np.std(negative_two[:,:-1], axis=0, dtype=np.float64, ddof=1)

    # Getting rid of the zeros to stop divide by zero error
    # Wasn't actually needed, but useful for addition testing with more data.
    stdev_positive[stdev_positive == 0] = 0.000000000001
    stdev_negative[stdev_negative == 0] = 0.000000000001

    return test_set, mean_positive, mean_negative, stdev_positive, stdev_negative, probability_spam, probability_not_spam


# Gaussian function used to calculate the probabilities.
def gaussian_function(mean, standard_dev, test_set):

    inner_probability = np.zeros(test_set.shape)
    inner_probability = (1. / ((np.sqrt(2. * np.pi))*standard_dev)) * (np.exp(-((test_set - mean) ** 2.)/(2. * (standard_dev)**2.)))
    # Avoiding the log(0) by finding the values == 0, and setting them to a small number.
    inner_probability[inner_probability == 0] = small_number
    inner_probability = np.sum(np.log10(inner_probability), axis=1)
    return inner_probability


# This builds the prediction set given the gaussian probabilities of each test input.
def build_prediction_set(inner_positive_probability, inner_negative_probability, probability_spam, probability_not_spam, test_set):

    # The chances of the input being positive or negative
    probability_pos = inner_positive_probability + np.log10(probability_spam)
    probability_neg = inner_negative_probability + np.log10(probability_not_spam)

    # Build the prediction set based on which probability is more likely
    predictions = np.zeros(probability_pos.shape)
    for i in range(len(predictions)):
        if(probability_pos[i] > probability_neg[i]):
            predictions[i] = 1
        if(probability_neg[i] > probability_pos[i]):
            predictions[i] = 0
        if(probability_pos[i] == probability_neg[i]):
            predictions[i] = np.random.randint(0,2)
    return predictions


# Accuracy calculation function
def accuracy_function_test(predictions, test_set):
    count = 0
    for i in range(len(test_set)):
        if(predictions[i] == test_set[i,-1]):
            count += 1
    accuracy = count / len(test_set)
    return accuracy


# Precision calculation function
def precision_function_test(predictions, test_set):
    false_positive = 0
    true_positive = 0
    for j in range(len(predictions)):
        if(predictions[j] == 1 and test_set[j,-1] == 0):
            false_positive += 1.
        if(predictions[j] == 1 and test_set[j,-1] == 1):
            true_positive += 1.
    precision = true_positive / (true_positive+false_positive)
    return precision


# Recall calculation function
def recall_function_test(predictions, test_set):
    false_negative = 0
    true_positive = 0
    for j in range(len(predictions)):
        if predictions[j] == 0 and test_set[j,-1] == 1:
            false_negative += 1.
        if predictions[j] == 1 and test_set[j,-1] == 1:
            true_positive += 1.
    recall = true_positive / (true_positive+false_negative)
    return recall


# Confusion matrix builder function
def confusion_matrix_builder(predictions, test_set):
    matrix = np.zeros(shape=(2,2),dtype=int)
    for i in range(len(predictions)):
        if predictions[i] == 1 and test_set[i,-1] == 1:
            matrix[1,1] += 1
        if predictions[i] == 1 and test_set[i,-1] == 0:
            matrix[1,0] += 1
        if predictions[i] == 0 and test_set[i,-1] == 1:
            matrix[0,1] += 1
        if predictions[i] == 0 and test_set[i,-1] == 0:
            matrix[0,0] += 1
    return matrix


def main():

    print("My Main")

    # Gather the probabilistic model information from the data and build the test set
    test_set, mean_positive, mean_negative, stdev_positive, stdev_negative, probability_spam, probability_not_spam = data_processing_function("spambase.data")

    # Calculate the probabilities for the test set to be negative or positive.
    inner_positive_probability = gaussian_function(mean_positive,stdev_positive,test_set[:,:-1])
    inner_negative_probability = gaussian_function(mean_negative, stdev_negative, test_set[:,:-1])

    # Build the prediction set based on the probabilities
    predictions = build_prediction_set(inner_positive_probability, inner_negative_probability, probability_spam, probability_not_spam, test_set)

    # Calculate the accuracy, precision, and recall of the predictions.
    accuracy = accuracy_function_test(predictions,test_set)
    precision = precision_function_test(predictions,test_set)
    recall = recall_function_test(predictions,test_set)
    matrix = confusion_matrix_builder(predictions,test_set)

    # Outputs
    print("Acccuracy:",100*accuracy)
    print("Precision:", 100*precision)
    print("Recall:", 100*recall)
    print("Confusion Matrix")
    print(matrix)



main()
