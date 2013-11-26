'''
Created on Oct 7, 2013

@author: excelsior
'''

from models import dataset, constants
from utils import file_reader
from algos import naive_bayes, tan

import matplotlib.pyplot as plt
from random import randrange
import math

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from decimal import *

# Round of the decimal output to 16 decimal places
getcontext().prec = 16

TRAIN_DATASET_FILE_PATH = "data/lymph_train.arff"
TEST_DATASET_FILE_PATH = "data/lymph_test.arff"

ALGO_NAIVE_BAYES = "naive_bayes"
ALGO_TAN = "tan"

"""
    Generates various graphs for analyzing the decision tree learning algorithm.
    1) Effect of varying the training data size on test data accuracy for TAN and Naive Bayes model.
"""

"""
    Generates a subset - stratified and random from the given dataset. With replacement sampling 
    approach is used to ensure that the same instance is not repeated again for the dataset.
"""
def get_stratified_random_data_set(training_dataset, size):
    
    # If size of expected stratified set equals overall size of data set, return overall dataset
    if size == len(training_dataset.examples):
        return training_dataset
    
    examples = training_dataset.examples
    class_label_map = training_dataset.get_class_labels_map()
    class_labels = class_label_map.keys()
    first_class_ratio = class_label_map[class_labels[0]]/ Decimal(len(examples))
    second_class_ratio = class_label_map[class_labels[1]]/ Decimal(len(examples))
    
    first_class_examples_new_set_cnt = int(round(first_class_ratio*size))
    second_class_examples_new_set_cnt = size - first_class_examples_new_set_cnt
    
    first_class_label_examples = [example for example in examples if example.class_label == class_labels[0]]
    second_class_label_examples = [example for example in examples if example.class_label == class_labels[1]]
    stratified_examples = []
    
    visited_indices = []
    for i in xrange(first_class_examples_new_set_cnt):
        # Keep looping till you find one index which has not been added to the set yet
        while 1 == 1:
            rand_chosen_index = randrange(len(first_class_label_examples))
            if rand_chosen_index not in visited_indices:
                stratified_examples.append(first_class_label_examples[rand_chosen_index-1])
                visited_indices.append(rand_chosen_index)
                break
        
    visited_indices = []
    for i in xrange(second_class_examples_new_set_cnt):
        while 1 == 1:
            rand_chosen_index = randrange(len(second_class_label_examples)) 
            if rand_chosen_index not in visited_indices:            
                stratified_examples.append(second_class_label_examples[rand_chosen_index-1])
                visited_indices.append(rand_chosen_index)
                break
    
    output_attribute = dataset.Label(constants.CLASS_LABEL_MARKER, training_dataset.get_output_labels())
    return dataset.Dataset(training_dataset.name, training_dataset.features, output_attribute, stratified_examples)

"""
    Plots the test set accuracy with the number of instances in training set
"""
def get_learning_curve_with_number_of_instances():
    train_dataset = file_reader.get_dataset_from_file(TRAIN_DATASET_FILE_PATH)
    test_dataset = file_reader.get_dataset_from_file(TEST_DATASET_FILE_PATH)
    
    training_set_sizes_repetitions = {}
    training_set_sizes_repetitions[25]  = 4
    training_set_sizes_repetitions[50]  = 4
    training_set_sizes_repetitions[100] = 1
    
    nb_test_set_accuracies_with_size, tan_test_set_accuracies_with_size = {}, {}
    avg_nb_test_data_accuracy, avg_tan_test_data_accuracy = [], []
    for size in sorted(training_set_sizes_repetitions.keys()):
        num_repetitions = training_set_sizes_repetitions[size]
        
        nb_test_set_accuracies, tan_test_set_accuracies = [], []
        for counter in xrange(0, num_repetitions):
            stratified_training_data_set = get_stratified_random_data_set(train_dataset, size)
            
            # Test for Naive Bayes Model
            naive_bayes_model = naive_bayes.Naivebayes(stratified_training_data_set)
            test_set_accuracy = naive_bayes.evaluate_naive_bayes_model(naive_bayes_model, test_dataset)
            nb_test_set_accuracies.append(test_set_accuracy*100)
            logger.info("Naive Bayes accuracy : " + str(test_set_accuracy*100) + " % ")
            
            # Test for TAN model
            tan_model = tan.TAN(stratified_training_data_set)
            test_set_accuracy = tan.evaluate_tan_model(tan_model, test_dataset)
            tan_test_set_accuracies.append(test_set_accuracy*100)
            logger.info("TAN accuracy : " + str(test_set_accuracy*100) + " % ")          
            
        nb_test_set_accuracies_with_size[size] = nb_test_set_accuracies
        avg_nb_test_data_accuracy.append(sum(nb_test_set_accuracies)/Decimal(len(nb_test_set_accuracies)))
        
        tan_test_set_accuracies_with_size[size] = tan_test_set_accuracies
        avg_tan_test_data_accuracy.append(sum(tan_test_set_accuracies)/Decimal(len(tan_test_set_accuracies)))        
        
    training_data_set_sizes = sorted(training_set_sizes_repetitions.keys())

    logging.info("\n\n Training data set sizes : " + str(training_data_set_sizes))
    logging.info("\n\n NB test set accuracies : " + str(avg_nb_test_data_accuracy))
    logging.info("\n\n TAN test set accuracies : " + str(avg_tan_test_data_accuracy))
    
    # Plot the graph
    plt.figure()

    plt.plot(training_data_set_sizes, avg_nb_test_data_accuracy, label="Naive Bayes", marker='H')
    plt.plot(training_data_set_sizes, avg_tan_test_data_accuracy, label="TAN", marker='H')
            
    plt.xlabel("Training set size")
    plt.ylabel("Test set accuracy (%)")
    plt.title("Test accuracy vs Training set size")
    
    plt.xlim(0, 101)
    plt.ylim(0, 100) 
 
    plt.grid(True)
    plt.legend(loc="lower right")
    
    plt.savefig("graphs/hw3_accuracy_vs_size_run2.png")    

if __name__ == '__main__':
    get_learning_curve_with_number_of_instances()