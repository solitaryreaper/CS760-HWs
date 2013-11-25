'''
Created on Nov 20, 2013

@author: excelsior
'''

'''
    Build a Naive Bayes model using the training data set
'''
from decimal import *

# Round of the decimal output to 16 decimal places
getcontext().prec = 16

class Naivebayes(object):
    
    def __init__(self, dataset):
        self.train_dataset = dataset

    '''
        Prints the naive bayes net structure
        The format is <feature name> <parent name>.
    '''
    def print_inference_network(self):
        for feature in self.train_dataset.get_features():
            print feature.name + " " + self.train_dataset.output.get_output_attribute_name()
            
    '''
        Gets the probability that the current test example would have the specified classification
        label.Specifically, calculate the following :
        P(label| FEATURES) = 
            P(label)*P(F1|label)*...P(Fn|label) / 
            P(label)*P(F1|label)*...P(Fn|label) + P(~label)*P(F1|~label)*...P(Fn|~label)
        Use LaPlace estimates in all probabilities
    '''
    def get_label_probabilistic_score(self, example, label):
        labels = self.train_dataset.get_output_labels()
        total_examples = len(self.train_dataset.get_examples())
        label0_examples = self.train_dataset.get_count_examples_with_label(labels[0])
        label1_examples = self.train_dataset.get_count_examples_with_label(labels[1])
        
        # Get the basic probability of any class label occuring using laplace estimates
        # P(LABEL) and P(~LABEL) calculation here
        label0_cond_prob_score = (label0_examples + 1) / Decimal(total_examples + 2)
        label1_cond_prob_score = (label1_examples + 1) / Decimal(total_examples + 2)
                
        # Get the probability of P(Feature | LABEL) and P(Feature | ~LABEL) for each of the
        # feature in the current example
        for f_key, f_value in example.feature_val_dict.items():
            num_values_for_feature = len(f_key.get_feature_values())
            label0_feature_value_examples = self.train_dataset.get_count_examples_with_label_and_feature_value(labels[0], f_key, f_value)
            label1_feature_value_examples = self.train_dataset.get_count_examples_with_label_and_feature_value(labels[1], f_key, f_value)

            # Add imaginary example as per laplace estimates 
            laplace_label0_examples = label0_examples + num_values_for_feature
            laplace_label1_examples = label1_examples + num_values_for_feature                                   
            label0_cond_prob_score = label0_cond_prob_score * ((label0_feature_value_examples + 1)/Decimal(laplace_label0_examples))
            label1_cond_prob_score = label1_cond_prob_score * ((label1_feature_value_examples + 1)/Decimal(laplace_label1_examples))
            
        # Final conditional probability for class label for this example using feature values
        label0_score = Decimal(label0_cond_prob_score) /Decimal(label0_cond_prob_score + label1_cond_prob_score)
        label1_score = Decimal(label1_cond_prob_score) /Decimal(label0_cond_prob_score + label1_cond_prob_score)
        
        # Choose which class label is more likely to be the correct classification for this example
        # by considering the class label with higher probability
        score = 0.0
        if label == labels[0]:
            score = label0_score
        else:
            score = label1_score
            
        return score

'''
    Test the accuracy of the naive bayes model on a test dataset
'''
def evaluate_naive_bayes_model(naive_bayes_model, test_dataset):
    labels = test_dataset.output.get_output_attribute_values()
    
    # Print the naive bayes network structure
    naive_bayes_model.print_inference_network()
    print ""
    
    # Test all the examples in the test dataset
    success_count = 0 # number of examples with same actual and predicted label
    for example in test_dataset.examples:
        label0_score = naive_bayes_model.get_label_probabilistic_score(example, labels[0])
        label1_score = naive_bayes_model.get_label_probabilistic_score(example, labels[1])
        
        best_score = 0.0
        predicted_label = None
        if label0_score > label1_score:
            best_score = label0_score
            predicted_label = labels[0]
        elif label1_score > label0_score:
            best_score = label1_score
            predicted_label = labels[1]
        # If both labels are equally probable, output the first label as default label
        else:
            best_score = label0_score
            predicted_label = labels[0]

        best_score = label0_score if label0_score > label1_score else label1_score

        actual_label = example.class_label
        predicted_label = labels[0] if label0_score > label1_score else labels[1]
        if actual_label == predicted_label:
            success_count = success_count + 1

        print predicted_label + " " + actual_label + " " + str(best_score)        
        
    print ""

    # Report the test dataset accuracy    
    print str(success_count)
        
        