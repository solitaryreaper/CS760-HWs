'''
Created on Oct 5, 2013

@author: excelsior
'''
import constants
import math

"""
    Models the decision tree data structure.
"""

class DecisionTreeNode(object):
    
    # Initialize a decision tree node with the reaching examples and attributes
    # By default, assume node is for nominal feature.
    def __init__(self, examples, features):
        self.examples = examples
        self.features = features
        
        self.best_feature = None
        self.node_value = None
        self.node_type = constants.FEATURE_NODE
        self.is_node_for_nominal_feature = True
        self.feature_value_nodes = None
        
        # For numeric features, the point of split also needs to be recorded
        self.numeric_split_threshold = None

# Driver for learning decision tree from the training data set
def learn_dtree(dataset, leaf_threshold):
    default_class_label = dataset.get_default_class_label()
    return _build_dtree(dataset.get_examples(), dataset.get_features(), default_class_label, 
                        default_class_label, leaf_threshold)

# Pretty prints the decision tree
def print_dtree(root, prefix):
    if root is None:
        return
    
    # Print leaf/class node
    if root.node_type == constants.CLASS_NODE:
        print prefix + " : " + root.best_feature
    # Print feature node
    else:
        root_name = root.best_feature.name
        feature_value_nodes = root.feature_value_nodes
        
        if root.is_node_for_nominal_feature:
            operator = " = "
            for node in feature_value_nodes:
                print root_name + " " + operator + " " + node.node_value
        else:
            pass
    

# Core algorithm that builds the decision tree. This method is recursively called to generate the
# entire decision tree.
def _build_dtree(examples, features, parent_majority_class, default_class, leaf_threshold):
    
    # Returns a leaf node with majority class label if either examples or attributes are exhausted
    if not examples or not features:
        return create_leaf_node(examples, features, parent_majority_class)

    class_labels_map = get_class_labels_map(examples)
    
    # Return a leaf node if all instances have the same classification
    if len(class_labels_map.keys) == 1:
        pure_class_label = class_labels_map.keys()[:1]
        return create_leaf_node(examples, features, pure_class_label)
        
    majority_class = get_majority_class(class_labels_map)
    # Check if number of instances remaining is less than threshold. If yes, create a leaf node
    if len(examples) < leaf_threshold:
        if majority_class is None:
            return create_leaf_node(examples, features, default_class)
        else:
            return create_leaf_node(examples, features, majority_class)
    
    # Choose the best feature
    best_feature = get_best_feature(examples, features)
    root = create_feature_node(examples, features)
    
    # Create links for each feature value
    feature_value_nodes = []
    
    # Handle nominal/discrete features first
    if best_feature.is_nominal_feature():
        for feature_value in best_feature.get_feature_values():
            filtered_examples = get_examples_with_feature_value(examples, feature_value, 
                                                                        best_feature, constants.EQUALS)
            filtered_features = features.remove(best_feature)
            feature_value_node = _build_dtree(filtered_examples, filtered_features, majority_class, 
                                              default_class)
            feature_value_nodes.append(feature_value_node)
    else:
        best_split_value = get_best_split_threshold(examples, best_feature)
        root.numeric_split_threshold = best_split_value
        
        less_than_value_examples = get_examples_with_feature_value(examples, best_split_value, best_feature, constants.LESS_THAN_OR_EQUAL_TO)
        greater_than_value_examples = get_examples_with_feature_value(examples, best_split_value, best_feature, constants.GREATER_THAN)
        
        less_than_feature_value_node = _build_dtree(less_than_value_examples, features, majority_class, 
                                              default_class)
        greater_than_feature_value_node = _build_dtree(greater_than_value_examples, features, majority_class, 
                                              default_class)
        
        feature_value_nodes.append(less_than_feature_value_node)      
        feature_value_nodes.append(greater_than_feature_value_node)  
    
    root.feature_value_nodes = feature_value_nodes
     
    return root

# Creates a leaf node in the decision tree
def create_leaf_node(examples, features, class_label):
    node = DecisionTreeNode(examples, features)
    node.best_feature = None
    node.node_value = class_label
    node.node_type = constants.CLASS_NODE
    
    return node
    
# Creates an intermediate node in the decision tree    
def create_feature_node(examples, features, best_feature):
    node = DecisionTreeNode(examples, features)
    node.best_feature = best_feature.name
    node.node_type = constants.FEATURE_NODE
    
    return node
    
# Core algorithm to determine the best feature    
def get_best_feature(examples, features):
    best_feature = None
    max_info_gain = -999
    for feature in features:
        info_gain = None
        if feature.is_nominal_feature():
            info_gain = get_info_gain_for_nominal_feature(examples, feature)
        else:
            info_gain = get_best_info_gain_for_numeric_feature(examples, features)
        
        if info_gain > max_info_gain:
            best_feature = feature
        
    return best_feature

# Gets the information gain for a nominal feature
def get_info_gain_for_nominal_feature(examples, feature):
    total_info = get_info(examples)
    
    feature_info = 0.0
    for value in feature.get_feature_values():
        filtered_examples = get_examples_with_feature_value(value, constants.EQUALS)
        feature_info += get_info(filtered_examples)

    return total_info - feature_info

# Gets the information gain for a numeric feature by splitting at a threshold that maximizes the
# information gain
def get_best_info_gain_for_numeric_feature(examples, feature):
    total_info = get_info(examples)
        
    best_split_value = get_best_split_threshold(examples, feature)
    feature_info = get_info_for_numeric_value_split(examples, feature, best_split_value)
    
    return total_info - feature_info

# What is the information gain if this numeric value is used for splitting the examples ?    
def get_info_for_numeric_value_split(examples, feature, value):
    feature_info = 0.0
    less_than_value_examples = get_examples_with_feature_value(examples, value, feature, constants.LESS_THAN_OR_EQUAL_TO)
    greater_than_value_examples = get_examples_with_feature_value(examples, value, feature, constants.GREATER_THAN)
    
    feature_info += get_info(less_than_value_examples)
    feature_info += get_info(greater_than_value_examples)
    
    return feature_info        

# Gets the total information contained in the examples    
def get_info(examples):
    if not examples or len(examples) == 0:
        return 0.0
    
    total_examples = len(examples)
    class_labels_map = get_class_labels_map(examples)
    
    class_labels = class_labels_map.keys()
    first_class_examples = class_labels_map[class_labels[0:1]]
    second_class_examples = class_labels_map[class_labels[1:2]]
    
    # Calculate the total information at this node
    first_class_info, second_class_info = 0.0, 0.0
    if first_class_examples > 0:
        first_class_info = -(first_class_examples/total_examples)*math.log(first_class_examples/total_examples, 2)
    if second_class_examples > 0:
        second_class_info = -(second_class_examples/total_examples)*math.log(second_class_examples/total_examples, 2)
    total_info =  first_class_info + second_class_info
    
    return total_info
        
# Determine the point of split for a numeric feature which maximizes information gain    
def get_best_split_threshold(examples, feature):
    best_split_threshold = None
    best_info_gain = -999.0
    candidate_split_values = get_candidate_split_values(examples, feature)
    for split_value in candidate_split_values:
        info_gain = get_info(examples) - get_info_for_numeric_value_split(examples, feature, split_value)
        if info_gain > best_info_gain:
            info_gain = best_info_gain
            best_split_threshold = split_value
    
    return best_split_threshold

# Determines the set of candidate splits points for a numeric feature
def get_candidate_split_values(examples, feature):
    candidate_split_values = []
    
    # build a value bucketed data structure first
    example_value_map = {}
    for example in examples:
        feature_value = example.get_value_for_feature(feature)
        if feature_value in example_value_map:
            feature_value_examples = example_value_map[feature_value]
        else:
            feature_value_examples = []
         
        feature_value_examples.append(example)
        example_value_map[feature_value] = feature_value_examples
    
    
    feature_values_sorted = example_value_map.keys().sort()
    
    prev_feature_value = feature_values_sorted[0:1]
    for value in feature_values_sorted[1:]:
        next_feature_value = value
        prev_value_examples = example_value_map[prev_feature_value]
        next_value_examples = example_value_map[next_feature_value]
        
        is_eligible_for_split = is_eligible_for_numeric_split(prev_value_examples, next_value_examples)
        if is_eligible_for_split:
            candidate_split_values.append(prev_feature_value + next_feature_value / 2)
        
        prev_feature_value = next_feature_value
        
    return candidate_split_values

# Determines if the adjacent sets for a numeric value are eligible for numeric split
def is_eligible_for_numeric_split(prev_value_examples, next_value_examples):
    is_eligible_for_numeric_split = False
    prev_value_class_map = get_class_labels_map(prev_value_examples)
    next_value_class_map = get_class_labels_map(next_value_examples)
    
    prev_value_class_labels = prev_value_class_map.keys()
    next_value_class_labels = next_value_class_map.keys()
    
    # Both labels are present in both adjacent sets
    if len(prev_value_class_labels) == 2 and len(next_value_class_labels) == 2:
        is_eligible_for_numeric_split = True
    
    # One set just has one class label and other set has both labels
    elif math.fabs(len(prev_value_class_labels) - len(next_value_class_labels)) == 1:
        is_eligible_for_numeric_split = True
        
    # Both sets have a single class label and they are different
    elif prev_value_class_labels != next_value_class_labels:
        is_eligible_for_numeric_split = True
    else:
        pass
        
    
    return is_eligible_for_numeric_split


#################### Utility functions #############################################################

# Gets a map of classification labels and their respective counts
def get_class_labels_map(examples):
    class_labels_map = {}
    for example in examples:
        curr_class_label = example.class_label
        if curr_class_label in class_labels_map:
            class_labels_map[curr_class_label] = class_labels_map[curr_class_label] + 1
        else:
            class_labels_map[curr_class_label] = 1
            
    return class_labels_map

# Determines the majority class in the examples
def get_majority_class(class_labels_map):
    class_labels = class_labels_map.keys()    
    first_class_label_instances = class_labels_map[class_labels[0:1]]
    second_class_label_instances = class_labels_map[class_labels[1:2]]
    if first_class_label_instances == second_class_label_instances:
        return None
    elif first_class_label_instances > second_class_label_instances:
        return class_labels[0:1]
    else:
        return class_labels[1:2]
    
# Filter out all the examples with the specific value for a feature    
def get_examples_with_feature_value(examples, feature_value, feature, operator):
    filtered_examples = []
    
    for example in examples:
        example_value = example.get_value_for_feature(feature)
        if operator == constants.EQUALS and example_value == feature_value:
            filtered_examples.append(example)
        elif operator == constants.LESS_THAN_OR_EQUAL_TO and example_value <= feature_value:
            filtered_examples.append(example)
        elif operator == constants.GREATER_THAN and example_value > feature_value:
            filtered_examples.append(example)
        else:
            pass
        
    return filtered_examples    