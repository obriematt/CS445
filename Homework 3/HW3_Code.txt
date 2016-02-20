# Matthew O'Brien
# CS 445 Homework #3


import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

# Option to change the value of K if interested
k_fold = 10
num_threshold = 200


# Processing and creating the test and train sets
def data_processing_function(datafile):
    # Load the data into a numpy array
    data = np.loadtxt('spambase.data', delimiter = ',')

    # Find the negative and positive data sets. Count them. Found the minimum of the two
    negative_data = data[data[:,-1] == 0]
    positive_data = data[data[:,-1] == 1]
    positive_count = positive_data.shape[0]
    negative_count = negative_data.shape[0]
    balanced_count = min(positive_count,negative_count)
    negative_data = negative_data[:balanced_count]
    positive_data = positive_data[:balanced_count]

    # Split the negative and positive values into two groups
    positive_one = positive_data[:len(positive_data)/2]
    positive_two = positive_data[len(positive_data)/2:]
    negative_one = negative_data[:len(negative_data)/2]
    negative_two = negative_data[len(negative_data)/2:]

    # Create Test and Train sets
    test_set = np.vstack((positive_one,negative_one))
    train_set = np.vstack((positive_two, negative_two))

    # Normalize the data set
    norm = preprocessing.StandardScaler().fit(train_set[:,:-1],train_set[:,-1])
    train_set[:,:-1] = norm.transform(train_set[:,:-1])
    test_set[:,:-1] = norm.transform(test_set[:,:-1])

    # Randomize the data sets. PDF says to random the training only. But going to do both anyway.
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    return train_set,test_set


# K Fold function for finding best C parameter. Experiment 1
def cross_validate(train_set):

    # Split data into approx K groupings
    data_split = np.array_split(train_set,k_fold)
    accuracy = 0
    c_star = 0

    # Find the best C parameter
    for i in range(10):
        # Create C parameters, SVM, and set the accuracy to check
        C_param = float(i+1)/10
        SVM = SVC(kernel='linear', C=C_param)
        prev_accuracy = accuracy

        # reset accuracy
        accuracy = 0

        for j in range(k_fold):
            to_train = list(data_split)
            validation = to_train.pop(j)
            to_train = np.vstack(to_train)

            # first parameter all features, second parameter label
            SVM.fit(to_train[:,:-1], to_train[:,-1])

            # Find Prediction for train set with SVM
            predictions = SVM.predict(validation[:,:-1])
            accuracy = accuracy + metrics.accuracy_score(validation[:,-1], predictions)

        # Find the average accuracy
        accuracy = accuracy / k_fold

        # Check to see if accuracy is better. Set the new C parameter if true
        if accuracy > prev_accuracy:
            c_star = C_param

    return c_star


# Experiment 1 training for the SVM
def experiment1_train_svm(c_star,train_set,test_set):
    # Create the new SVM
    SVM_one = SVC(kernel='linear', C=c_star, probability=True)
    SVM_one.fit(train_set[:, :-1], train_set[:, -1])
    new_predictions = SVM_one.predict_proba(test_set[:, :-1])
    new_predictions = new_predictions[:,-1]

    # Copy to preserve the original predictions for use in ROC function
    predictions = np.copy(new_predictions)

    # Change the predictions into 1 and 0 for comparison against test set.
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    # Calculate the accuracy, precision, and recall of the SVM
    accuracy = metrics.accuracy_score(test_set[:,-1], predictions)
    precision = metrics.precision_score(test_set[:,-1], predictions)
    recall = metrics.recall_score(test_set[:,-1], predictions)

    return new_predictions, accuracy, precision, recall, SVM_one


# Creating the recall and FPR for the ROC curve
def roc_creation_function(new_predictions, test_set):

    # Variables and list for storage
    recall_list = []
    false_positive_list = []

    # The possible thresholds and loop to get through it.
    threshold = 1.0 / num_threshold
    for i in range(num_threshold):
        false_positive = 0.
        true_negative = 0.
        current_threshold = (i+1)*threshold
        predictions = np.copy(new_predictions)
        predictions[predictions >= current_threshold] = 1
        predictions[predictions < current_threshold] = 0
        for j in range(len(predictions)):
            if(predictions[j] == 1 and test_set[j,-1] == 0):
                false_positive += 1.
            if(predictions[j] == 0 and test_set[j,-1] == 0):
                true_negative += 1.
        false_positive_rate = false_positive / (false_positive+true_negative)
        false_positive_list.append(false_positive_rate)
        recall = metrics.recall_score(test_set[:,-1],predictions)
        recall_list.append(recall)
    recall_list = list(reversed(recall_list))
    false_positive_list = list(reversed(false_positive_list))
    return recall_list, false_positive_list


# Function to plot the ROC Curve
def plot_roc_curve(false_positive_list, recall_list):
    plt.plot(false_positive_list, recall_list)
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.axis([0,1,0,1])
    plt.show()


# Experiment 2. Feature selections
def experiment_two_feature_selection(svm,train_set,test_set,c_star):

    # Find the weight vectors
    weight_vectors = svm.coef_
    # Absolute these for argsort to accurately find the max to min vector
    weight_vectors = np.absolute(weight_vectors)
    accuracies = np.zeros(58)

    # Sort the indexes of the weight vectors, then flip to largest to smallest.
    sorted_index = np.argsort(weight_vectors)
    reverse_index = np.fliplr(sorted_index)[0]

    # Print for the Experiment 2 to find best 5 features.
    print("Top 5 weight vectors for Experiment 2: ", reverse_index[:5])
    for i in range(2,len(reverse_index)+1):
        selector = reverse_index[:i]
        split_train = train_set[:,selector]
        split_test = test_set[:,selector]    # Create the new SVM
        SVM_one = SVC(kernel='linear', C=c_star, probability=True)
        SVM_one.fit(split_train, train_set[:, -1])
        new_predictions = SVM_one.predict(split_test)

        # Calculate the accuracy, precision, and recall of the SVM
        accuracy = metrics.accuracy_score(test_set[:,-1], new_predictions)
        accuracies[i] = accuracy
    return accuracies


# Experiment 3 function. Very similar to 2 just randomized
def experiment_three_feature_selection(svm,train_set,test_set,c_star):

    # Find the weight vectors
    weight_vectors = svm.coef_
    accuracies = np.zeros(58)

    # Sort the indexes of the weight vectors, then flip to largest to smallest.
    sorted_index = np.argsort(weight_vectors)
    reverse_index = np.fliplr(sorted_index)[0]

    # Randomizing the indices for the experiment
    np.random.shuffle(reverse_index)
    for i in range(2,len(reverse_index)+1):
        selector = reverse_index[:i]
        split_train = train_set[:,selector]
        split_test = test_set[:,selector]

        # Create the new SVM
        SVM_one = SVC(kernel='linear', C=c_star, probability=True)
        SVM_one.fit(split_train, train_set[:, -1])
        new_predictions = SVM_one.predict(split_test)

        # Calculate the accuracy, precision, and recall of the SVM
        accuracy = metrics.accuracy_score(test_set[:,-1], new_predictions)
        accuracies[i] = accuracy

        # Was not entirely sure if this part was needed.
        # Reshuffled this way every time it was a randomized set of m features.
        # np.random.shuffle(reverse_index)
    return accuracies


# Basic plot function for Experiment 2 and 3
def plot_accuracy_graph(accuracies):
    number_features = np.arange(58)
    plt.plot(number_features, accuracies)
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Features")
    plt.axis([0,58,0,1])
    plt.show()


# Main Function.
def main():
    print("My main")
    # Creation of the two sets
    train_set, test_set = data_processing_function("spambase.data")

    # Running K-Fold to find best C parameter
    c_param = cross_validate(train_set)
    print("Experiment 1, C* value: ",c_param)

    # Experiment 1 on the best C parameter.
    predictions, accuracy, precision, recall, svm = experiment1_train_svm(c_param,train_set,test_set)
    print("Experiment 1, Precision Score: ", precision*100)
    print("Experiment 1, Accuracy Score: ", accuracy*100)
    print("Experiment 1, Recall Score: ", recall*100)

    # Experiment 2
    accuracy_experiment_two = experiment_two_feature_selection(svm,train_set,test_set,c_param)

    # Experiment 3
    accuracy_experiment_three = experiment_three_feature_selection(svm,train_set,test_set,c_param)

    # ROC Curve generation
    recall_list, false_positive_list = roc_creation_function(predictions,test_set)
    plot_roc_curve(false_positive_list,recall_list)
    plot_accuracy_graph(accuracy_experiment_two)
    plot_accuracy_graph(accuracy_experiment_three)

main()

