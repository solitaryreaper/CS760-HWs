'''
Created on Nov 20, 2013

@author: excelsior
'''

import math
from decimal import *

# Round of the decimal output to 16 decimal places
getcontext().prec = 16

'''
    Build a TAN (Tree Augmented Network) model based on the training data set
'''
class TAN(object):

    # A matrix that contains the mutual information gain for each feature in the dataset
    inter_feature_weight_matrix = None
    
    # The maximal spanning tree between the features in the dataset using the conditional mutual
    # information gain between the features as the edge weight
    maximal_spanning_feature_tree = None
    
    # A TAN bayesian inference network enriched with conditional probability table for each feature
    # node.
    tan_graph = None
    
    def __init__(self, dataset):
        self.train_dataset = dataset
        self.generate_tan_model()

    '''
        Generates the actual TAN model and data structures for the TAN algorithm
        
        1) Generate the mutual information gain values between every two features
        2) Calculate a maximum spanning tree between the features. This would give a relationship
           between a feature and its children.
        3) Pivot the above tree to generate the relationship between a child and all its parents.
        4) Calculate the CPT(Conditional Probability Table) for each child node.
    '''
    def generate_tan_model(self):
        features = self.train_dataset.get_features()
        
        print "Generating conditional mutual information gain between features matrix .."
        self.inter_feature_weight_matrix = get_mutual_info_gain_bw_features(self.train_dataset)
        
        print "Generating the maximal spanning tree between the features .."
        self.maximal_spanning_feature_tree = get_maximal_spanning_feature_tree(self.inter_feature_weight_matrix, features)
        
        print "Generating the TAN graph between features and the class label .."
        self.tan_graph = get_tan_graph(self.maximal_spanning_feature_tree, self.train_dataset)
            
    '''
        Prints the inference network computed by TAN model
    '''
    def print_inference_network(self):
        print "Printing the inference network .."
        for feature in self.train_dataset.get_features():
            tan_node = self.tan_graph.get(feature)
            
            parents_str_rep = ""
            if tan_node.get_parents():
                parents_str_rep = " ".join(p.get_name() for p in tan_node.get_parents())
                parents_str_rep = parents_str_rep + " " + tan_node.get_output().get_output_attribute_name()
            else:
                parents_str_rep = self.train_dataset.get_output().get_output_attribute_name()
            print feature.get_name() + " " + parents_str_rep
        
    '''
        Determine the probability of a specified label given the test example
        
        Calculate P(label | Feature values in example) using the TAN graph
        
        P(L|F1, F2, ... Fn) = P(F1, F2, ... Fn)/P(F1, F2, .. Fn)
    '''
    def get_label_probabilistic_score(self, example, label, tan_graph):
        total_examples = len(self.train_dataset.get_examples())
        label_examples = self.train_dataset.get_count_examples_with_label(label)
        label_cond_prob_score = (label_examples + 1) / Decimal(total_examples + 2)
                
        # Calculate numerator = P(F1, F2, ... Fn) using TAN graph
        feature_val_map = example.feature_val_dict
        prob_num = 1.0
        for f, f_value in feature_val_map.iteritems():
            curr_f_prob = get_cpt_score_for_feature(f, example, label, tan_graph, self.train_dataset)
            prob_num = Decimal(prob_num) * curr_f_prob
        
        return prob_num*label_cond_prob_score
   
#### Utility functions ###############

'''
    Calculates the CPT (Conditional Probability Table) score for a feature, given its parents
    from the TAN graph.
'''
def get_cpt_score_for_feature(feature, example, label, tan_graph, dataset):
    tan_node = tan_graph.get(feature)
    feature_value = example.get_value_for_feature(feature)
    feature_parents = tan_node.get_parents()
     
    parent_feature_val_map = {}
    feature_val_map = {}
    if feature_parents:
        for f in feature_parents:
            parent_feature_val_map[f] = example.get_value_for_feature(f)
            
        feature_val_map = dict(parent_feature_val_map) 
        
    feature_val_map[feature] = feature_value  
     
    ex_with_parent_f_values_label = dataset.get_count_examples_with_label_and_feature_values(label, parent_feature_val_map)
    ex_with_f_val_parent_f_values_label = dataset.get_count_examples_with_label_and_feature_values(label, feature_val_map)
    
    num_feature_values = len(feature.get_feature_values())
    cpt_score = (ex_with_f_val_parent_f_values_label + 1)/Decimal(ex_with_parent_f_values_label + num_feature_values)
    
    return Decimal(cpt_score)

'''
    Generates the conditional mutual information gain between features
'''
def get_mutual_info_gain_bw_features(dataset):
    inter_feature_weight_matrix = {}
    features = dataset.get_features()
    for f_out in features:
        weight_vector = {}
        for f_in in features:
            mutual_weight = None
            if f_out == f_in:
                # Mutual information gain between same features is not applicable.
                mutual_weight = -1.0
            else:
                mutual_weight = get_mutual_info_gain(f_in, f_out, dataset)
            
            weight_vector[f_in] = mutual_weight
        inter_feature_weight_matrix[f_out] = weight_vector
        
    for f in features:
        weight_vector = inter_feature_weight_matrix.get(f)

        vector_rep = ""
        for f1 in features:
            val = weight_vector[f1]
            vector_rep = vector_rep + " " + str(val)
        print "Weight vector for feature : " + f.get_name() + " is " + vector_rep
                
    return inter_feature_weight_matrix

'''
    Determines the conditional mutual information gain between any two features
    
    Calculate the mutual information gain for each possible value combination of feature1 value,
    feature2 value and output values. Sum each mutual information gain to determine the aggregate
    mutual information gain for each feature pair.
'''
def get_mutual_info_gain(feature1, feature2, dataset):
    mutual_info_gain = 0.0
    output = dataset.get_output()
    for f1_value in feature1.get_feature_values():
        for f2_value in feature2.get_feature_values():
            for op_value in output.get_output_attribute_values():
                curr_mutual_info_gain = get_mutual_info_gain_for_values(feature1, f1_value, feature2, f2_value, op_value, dataset)
                mutual_info_gain = Decimal(mutual_info_gain) + curr_mutual_info_gain
            
    return mutual_info_gain
        
'''
    Calculates the mutual information gain for specific features values and a class label
    
    P(xi, xj | y) = P(xi, xj, y)* log2(P(xi, xj|y)/ (P(xi|y)* P(xj|y)) 
'''            
def get_mutual_info_gain_for_values(feature1, f1_val, feature2, f2_val, label, dataset):
    feature_val_map = {}
    feature_val_map[feature1] = f1_val
    feature_val_map[feature2] = f2_val
    
    total_examples = Decimal(len(dataset.get_examples()))
    cnt_ex_with_xi_xj_y = Decimal(dataset.get_count_examples_with_label_and_feature_values(label, feature_val_map))
    cnt_ex_with_label_y = Decimal(dataset.get_count_examples_with_label(label))
    cnt_ex_with_xi_y = Decimal(dataset.get_count_examples_with_label_and_feature_value(label, feature1, f1_val))
    cnt_ex_with_xj_y = Decimal(dataset.get_count_examples_with_label_and_feature_value(label, feature2, f2_val))
    
    num_f1_values = len(feature1.get_feature_values())
    num_f2_values = len(feature2.get_feature_values())
    num_label_values = len(dataset.get_output().get_output_attribute_values())
    
    # Add appropriate laplace estimate values
    p_xi_xj_y = Decimal((cnt_ex_with_xi_xj_y + 1) / (total_examples + num_f1_values*num_f2_values*num_label_values))
    p_xi_xj_given_y = Decimal((cnt_ex_with_xi_xj_y + 1)/ (cnt_ex_with_label_y + num_f1_values*num_f2_values))
    p_xi_given_y = Decimal((cnt_ex_with_xi_y + 1)/ (cnt_ex_with_label_y + num_f1_values))
    p_xj_given_y = Decimal((cnt_ex_with_xj_y + 1)/ (cnt_ex_with_label_y + num_f2_values))
    
    info_gain = p_xi_xj_y*Decimal(math.log((p_xi_xj_given_y/(p_xi_given_y*p_xj_given_y)), 2))
    return info_gain

'''
    Generates the maximal spanning tree for all the features.
    
    A maximal spanning tree is a tree that includes all the vertices such that the sum of the edges
    included is maximum.
    
    Parent feature ==> Child features graph.
'''
def get_maximal_spanning_feature_tree(inter_feature_weight_matrix, features):
    max_spanning_feature_tree = {}
    new_vertices = []
    new_vertices.append(features[0])
    print "Added " + str(features[0].name) + " as the root vertex .."
    while len(new_vertices) != len(features):
        max_wgt_edge_source, max_wgt_edge_target = None, None
        max_weighted_edge = -1.0
        
        # Find the maximal weighted edge such that the edge source has already been explored for
        # the current spanning tree but the edge target has still not been explored.
        for edge_source in new_vertices:
            weighted_edge_targets = inter_feature_weight_matrix[edge_source]
            for edge_target, weight in weighted_edge_targets.iteritems():
                if edge_target in new_vertices:
                    continue
                
                if weight > max_weighted_edge:
                    max_wgt_edge_source, max_wgt_edge_target = edge_source, edge_target
                    max_weighted_edge = weight
                                
        edges = []
        if max_wgt_edge_source in max_spanning_feature_tree:
            edges = max_spanning_feature_tree.get(max_wgt_edge_source)
        edges.append(max_wgt_edge_target)
        max_spanning_feature_tree[max_wgt_edge_source] = edges
        new_vertices.append(max_wgt_edge_target)
        
        print "Added a new edge. Source:" + max_wgt_edge_source.get_name() + ", Target:" + max_wgt_edge_target.get_name() + ", Weight:" + str(max_weighted_edge)
        
    return max_spanning_feature_tree

'''
    Generate the TAN(Tree Augmented Network) graph, establishing the relationship between
    a feature node and all its parents (including class label). 
    
    Child Feature --> Parent Features graph
'''
def get_tan_graph(maximal_spanning_feature_tree, dataset):
    output = dataset.get_output()
    
    # Determine the parents for each child feature node.
    tan_graph = {}
    for parent_f, children_f in maximal_spanning_feature_tree.iteritems():
        for f in children_f:
            parents = []
            if f in tan_graph:
                parents = tan_graph.get(f)
            parents.append(parent_f)
            tan_graph[f] = parents

    # Map each child feature node to its parent feature nodes
    tan_node_graph = {}
    for child_f in dataset.get_features():
        parents_f = tan_graph.get(child_f)            
        tan_node = TANNode(child_f, parents_f, output)
        tan_node_graph[child_f] = tan_node

    return tan_node_graph

'''
    Model object to represent a node in the TAN graph.
'''    
class TANNode(object):
    def __init__(self, feature, feature_parents, output):
        self.feature = feature
        self.output = output
        self.feature_parents = feature_parents
        
    def get_parents(self):
        return self.feature_parents
    
    def get_output(self):
        return self.output

'''
    Test the accuracy of the naive bayes model on a test dataset
'''
def evaluate_tan_model(tan_model, test_dataset):
    labels = test_dataset.output.get_output_attribute_values()
    
    # Print the naive bayes network structure
    tan_model.print_inference_network()
    print ""
    
    # Test all the examples in the test dataset
    success_count = 0 # number of examples with same actual and predicted label
    for example in test_dataset.examples:
        label0_numertr_score = tan_model.get_label_probabilistic_score(example, labels[0], tan_model.tan_graph)
        label1_numertr_score = tan_model.get_label_probabilistic_score(example, labels[1], tan_model.tan_graph)
        
        label0_score = Decimal(label0_numertr_score)/Decimal(label0_numertr_score + label1_numertr_score)
        label1_score = Decimal(label1_numertr_score)/Decimal(label0_numertr_score + label1_numertr_score)        
        
        best_score = label0_score if label0_score >= label1_score else label1_score

        actual_label = example.class_label
        predicted_label = labels[0] if label0_score >= label1_score else labels[1]
        if actual_label == predicted_label:
            success_count = success_count + 1

        print predicted_label + " " + actual_label + " " + str(best_score)        
        
    print ""

    # Report the test dataset accuracy    
    print str(success_count)
