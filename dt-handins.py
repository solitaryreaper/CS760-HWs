'''
Created on Oct 7, 2013

@author: excelsior
'''

from models import dtree, dataset
dtlearn = __import__("dt-learn")

import matplotlib.pyplot as plt
from random import randrange
import math

TRAIN_DATASET_FILE_PATH = "data/heart_train.arff"
TEST_DATASET_FILE_PATH = "data/heart_test.arff"

"""
    Generates various graphs for analyzing the decision tree learning algorithm.
"""

# Generats a subset - stratified and random from the given dataset
def get_stratified_random_data_set(training_dataset, size):
    if size == len(training_dataset.examples):
        return training_dataset
    
    examples = training_dataset.examples
    class_label_map = dtree.get_class_labels_map(examples)
    class_labels = class_label_map.keys()
    first_class_ratio = class_label_map[class_labels[0]]/ float(len(examples))
    second_class_ratio = class_label_map[class_labels[1]]/ float(len(examples))
    
    first_class_examples_new_set_cnt = int(math.ceil(first_class_ratio*size))
    second_class_examples_new_set_cnt = size - first_class_examples_new_set_cnt
    
    first_class_label_examples = [example for example in examples if example.class_label == class_labels[0]]
    second_class_label_examples = [example for example in examples if example.class_label == class_labels[1]]
    stratified_examples = []
    
    visited_indices = []
    for i in xrange(0, first_class_examples_new_set_cnt):
        # Keep looping till you find one index which has not been added to the set yet
        while 1 == 1:
            rand_chosen_index = randrange(len(first_class_label_examples))
            if rand_chosen_index not in visited_indices:
                stratified_examples.append(first_class_label_examples[rand_chosen_index-1])
                visited_indices.append(rand_chosen_index)
                break
    #print "Stratified indices first label : " + str(visited_indices) + " for size " + str(size)
        
    visited_indices = []
    for i in xrange(0, second_class_examples_new_set_cnt):
        while 1 == 1:
            rand_chosen_index = randrange(len(second_class_label_examples)) 
            if rand_chosen_index not in visited_indices:            
                stratified_examples.append(second_class_label_examples[rand_chosen_index-1])
                visited_indices.append(rand_chosen_index)
                break
    #print "Stratified indices second label : " + str(visited_indices) + " for size " + str(size)
    
    return dataset.Dataset(training_dataset.name, training_dataset.features, training_dataset.output_labels, stratified_examples)

"""
    Plots the test set accuracy with the number of instances in training set
"""
def get_learning_curve_with_number_of_instances():
    train_dataset = dtlearn.get_dataset_from_file(TRAIN_DATASET_FILE_PATH)
    test_dataset = dtlearn.get_dataset_from_file(TEST_DATASET_FILE_PATH)
    
    leaf_threshold = 4
    training_set_sizes_repetitions = {}
    training_set_sizes_repetitions[25] = 10
    training_set_sizes_repetitions[50] = 10
    training_set_sizes_repetitions[100] = 10
    training_set_sizes_repetitions[200] = 1
    
    test_set_accuracies_with_size = {}
    avg_test_data_accuracy, min_test_data_accuracy, max_test_data_accuracy = [], [], []
    for size in sorted(training_set_sizes_repetitions.keys()):
        print "Current training set size : " + str(size)
        num_repetitions = training_set_sizes_repetitions[size]
        test_set_accuracies = []
        for counter in xrange(0, num_repetitions):
            stratified_training_data_set = get_stratified_random_data_set(train_dataset, size)
            training_dataset_dtree = dtree.learn_dtree(stratified_training_data_set, leaf_threshold)
            test_set_accuracy = dtree.test_dtree(training_dataset_dtree, test_dataset)
            test_set_accuracies.append(test_set_accuracy)
            
        test_set_accuracies_with_size[size] = test_set_accuracies
        avg_test_data_accuracy.append(sum(test_set_accuracies)/float(len(test_set_accuracies)))
        min_test_data_accuracy.append(min(test_set_accuracies))
        max_test_data_accuracy.append(max(test_set_accuracies))
        
    training_data_set_sizes = sorted(training_set_sizes_repetitions.keys())
    print training_data_set_sizes
    print avg_test_data_accuracy
    print min_test_data_accuracy
    print max_test_data_accuracy
    print test_set_accuracies_with_size
    
    # Plot the graph
    plt.figure()

    plt.plot(training_data_set_sizes, avg_test_data_accuracy, label="avg accuracy vs size", marker='o')
    plt.plot(training_data_set_sizes, min_test_data_accuracy, label="min accuracy vs size", marker='o')
    plt.plot(training_data_set_sizes, max_test_data_accuracy, label="max accuracy vs size", marker='o')
            
    plt.xlabel("Training Data Size")
    plt.ylabel("Test Set Accuracy")
    plt.title("Test Set Accuracy vs Training Data Size")
    
    plt.xlim(0, 201)
    plt.ylim(0, 1) 
 
    plt.grid(True)
    plt.legend(loc="lower right")
    
    plt.savefig("graphs/accuracy_vs_size.png")    

"""
    Plots the test set accuracy as the value of "m" leaf threshold is varied
    TODO : How to mark the data points ?
"""
def get_test_set_accuracy_with_leaf_thresholds():
    train_dataset = dtlearn.get_dataset_from_file(TRAIN_DATASET_FILE_PATH)
    test_dataset = dtlearn.get_dataset_from_file(TEST_DATASET_FILE_PATH)
  
    leaf_thresholds = [2, 5, 10, 20]
    accuracy_with_leaf_thresholds = []
    for leaf_threshold in leaf_thresholds:
        print "Running test for leaf threshold : " + str(leaf_threshold)
        training_dataset_dtree = dtree.learn_dtree(train_dataset, leaf_threshold)
        test_set_accuracy = dtree.test_dtree(training_dataset_dtree, test_dataset)
        accuracy_with_leaf_thresholds.append(test_set_accuracy)
        print "Test set accuracy for leaf threshold " + str(leaf_threshold) + " is " + "{0:.2f}".format(test_set_accuracy)

    # Plot the graph
    plt.figure()

    plt.plot(leaf_thresholds, accuracy_with_leaf_thresholds, label="accuracy vs m-value",marker='o')
    plt.xlabel("Leaf Threshold")
    plt.ylabel("Test Set Accuracy")
    plt.title("Test Set Accuracy vs Leaf Threshold")
    
    plt.xlim(0, 22)
    plt.ylim(0, 1) 
 
    plt.grid(True)
    plt.legend(loc="upper left")
    
    plt.savefig("graphs/accuracy_vs_m.png")

if __name__ == '__main__':
    get_test_set_accuracy_with_leaf_thresholds()
    #get_learning_curve_with_number_of_instances()