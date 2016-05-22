# Matthew O'Brien
# CS 445 Homework #5
# K-mean clusters

import numpy as np
from random import randint
from scipy import stats
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
k_clusters = 10

# Builds the arrays for the data
def data_processing_function(datafile):

    # Loading the data into a numpy array
    data = np.loadtxt(datafile, delimiter=',')
    return data


# Basic creation of centroids from the data
def create_centroids(train_data):
    centroids = np.zeros([k_clusters,64], dtype=float)
    for i in range(len(centroids)):
        centroids[i] = train_data[(randint(0,len(train_data))), :]
    return centroids


# Calculate a single euclidean distance
def euclidean_dist(centroid, data):
    row = np.zeros([len(data)])
    row = np.sqrt(np.sum((centroid-data)**2,axis=1))
    return row


# Find all of the Euclidean distances
def all_euclidean_dist(centroids, train_array):
    all_e_dist = np.zeros([len(train_array),len(centroids)])
    for i in range(len(centroids)):
        all_e_dist[:,i] = euclidean_dist(centroids[i,:],train_array[:,:-1])
    return all_e_dist


# Assign the cluster's labels
def assign_clusters(distances):
    assignments = np.zeros([len(distances)])
    for i in range(len(distances)):
        assignments[i] = np.argmin(distances[i])
    return assignments


# Find new centroids based on their assignments and data
def retrain_centroids(centroids, assignments, data):
    for i in range(len(centroids)):
        centroids[i] = np.mean(data[assignments == i, :-1], axis=0)
    return centroids


# The initial "train" or making new centroids based on the data
def train_centroids(centroids, data):
    distances = all_euclidean_dist(centroids,data)
    assignments = assign_clusters(distances)
    centroids = retrain_centroids(centroids,assignments,data)
    return centroids, assignments


# Find the predictions for the centroids and the data
def find_predictions(centroid, data):
    distance = all_euclidean_dist(centroid,data)
    assignments = assign_clusters(distance)
    centroid_label = np.zeros(k_clusters)
    train_predictions = np.zeros(data.shape[0])
    for i in range(len(centroid)):
        centroid_label[i] = stats.mode(data[assignments == i, -1])[0]
        train_predictions[assignments == i] = centroid_label[i]
    return centroid_label,train_predictions


# Find the predictions for the test data
def pred_test_results(centroid, data, label):
    distance = all_euclidean_dist(centroid,data)
    assignments = assign_clusters(distance)
    test_prediction = np.zeros(data.shape[0])
    for i in range(len(centroid)):
        test_prediction[assignments == i] = label[i]
    return test_prediction


# Find the best cluster out of 5 randomly generated centroids
def find_best_clusters(train_data):
    sse_prev = 0
    best_centroids = np.zeros(k_clusters)
    for i in range(5):
        centroids = create_centroids(train_data[:,:-1])
        for i in range(100):
            centroids, assignments = train_centroids(centroids, train_data)
        sss = find_sss(centroids)
        sse = find_sse(centroids, train_data, assignments)
        if sse < sse_prev or sse_prev == 0:
            best_centroids = centroids
            best_sss = sss
            best_sse = sse
        sse_prev = sse
    entropy = find_entropy(train_data, assignments)
    return best_centroids, best_sss, best_sse, entropy


# Find the SSE of the centroids
def find_sse(centroids, data, assign):
    sse = 0
    for i in range(k_clusters):
        sse += np.sum(euclidean_dist(centroids[i], data[assign == i, :-1])**2)
    return sse


# Find the SSS of the centroids.
def find_sss(centroids):
    sss = 0
    for i in range(k_clusters):
        for j in range(k_clusters):
           sss = sss + (centroids[i] - centroids[j])**2
    sss = np.sum(sss)
    return sss


# Find the Entropy of each cluster
def find_entropy(data, assignment):
    ratios = np.zeros(10)
    entropy = np.zeros(k_clusters)
    m_entropy = 0
    for i in range(k_clusters):
        for j in range(10):
            ratios[j] = float(len(data[(assignment == i) & (data[:, -1] == j)])) / float(len(data[assignment == i]))
        log_ratio = np.log2(ratios)
        log_ratio[log_ratio == np.log2(0)] = 0
        entropy[i] = -np.sum(ratios * log_ratio)
        m_entropy += entropy[i] * len(data[assignment == i])
    m_entropy /= len(data)
    return m_entropy


# Main function.
def main():
    # Create the testing arrays
    test_array = data_processing_function("optdigits.test")
    train_array = data_processing_function("optdigits.train")

    # Find the best cluster and corresponding SSS
    centroids,sss, sse, entropy = find_best_clusters(train_array)

    # Find the predictions for each corresponding cluster
    centroid_label, train_prediction = find_predictions(centroids,train_array)

    # Run the testing data against the clusters and gain predictions
    test_prediction = pred_test_results(centroids, test_array,centroid_label)

    # The accuracy against the training data
    accuracy = metrics.accuracy_score(train_array[:,-1], train_prediction)

    # The accuracy for the testing data
    test_accuracy = metrics.accuracy_score(test_array[:, -1], test_prediction)

    # Confusion matrix!
    confused_matrix = metrics.confusion_matrix(test_array[:,-1],test_prediction)

    # Print statements for values
    print("These are the corresponding labels for each cluster:",centroid_label)
    print("This is the accuracy for the training data: ", accuracy*100)
    print("This is the accuracy for the testing data: ", test_accuracy*100)
    print("This is the best SSS: ",sss)
    print("This is the best SSE: ",sse)
    print("The mean Entropy is: ",entropy)
    print("The confusion matrix: ")
    print(confused_matrix)
    if k_clusters == 10:
        for i in range(k_clusters):
            plt.imsave('exp1_image_%i.png' %i, np.array(centroids[i,:]).reshape(8,8), cmap=cm.gray)
    if k_clusters == 30:
        for i in range(k_clusters):
            plt.imsave('exp2_image_%i.png' %i, np.array(centroids[i,:]).reshape(8,8), cmap=cm.gray)
main()
