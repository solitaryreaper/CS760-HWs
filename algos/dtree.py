'''
Created on Oct 5, 2013

@author: excelsior
'''
from models import constants
import math

"""
    Models the decision tree data structure.
"""

class DecisionTreeNode(object):
    
    # Initialize a decision tree node with the reaching examples and attributes
    # By default, assume node is for nominal feature.
    def __init__(self, examples, features):
        self.examples = examples # actual examples that reached this node
        self.features = features # actual features that reached this node
        
        self.best_feature = None # best feature determined for this node
        self.node_type = constants.FEATURE_NODE # Feature or leaf node ?
        self.is_node_for_nominal_feature = True # Numeric or nomnial feature ?
        self.feature_value_nodes = None # Link to the child nodes for each feature value

        # feature value of parent node that led to the creation of this child node
        # For numeric feature, this should be a logical expression like <feature> <= 0.5.
        # Think of this as a link between the parent and the child node.
        self.parent_feature_value = None
        self.parent_feature_name = None
        
        self.class_label = None # In case , this is a leaf node
      
    # Pretty string representation for debugging a node
    def __str__(self):
        if self.node_type != constants.CLASS_NODE:
            return "Feature Node : " + str(self.best_feature.name) + ", " + str(self.node_value)
        else:
            return "Class Node : " + str(self.node_value)
        
    # Returns a string representation showing the number of different class labels and their counts
    def class_labels_count(self, output_labels):
        class_labels_map = get_class_labels_map(self.examples)
        
        display_str = ""
        if output_labels[0] in class_labels_map:
            display_str += str(class_labels_map[output_labels[0]]) + " "
        else:
            display_str += " 0 "
        if output_labels[1] in class_labels_map:
            display_str += str(class_labels_map[output_labels[1]]) + " "
        else:
            display_str += " 0 "
        
        display_str = display_str.strip()
        display_str = display_str.replace("  " , " ")
        display_str = "[" + display_str +  "]"
            
        return display_str
    
# Driver for learning decision tree from the training data set
def learn_dtree(dataset, leaf_threshold):
    default_class_label = dataset.get_default_class_label()
    return _build_dtree(dataset.get_examples(), dataset.get_features(), default_class_label, leaf_threshold)

"""
    Core algorithm that builds the decision tree. This method is recursively called to generate the
    entire decision tree.
    
    Rules:
    1) Whenever stopping criterion is reached, choose majority class label. If no majority class
       label exists, then choose the default class label.
       
    
"""
def _build_dtree(examples, features, default_class, leaf_threshold):

    # Returns a leaf node with majority class label if examples is empty
    if examples is None or len(examples) == 0:
        return create_leaf_node(examples, features, default_class)

    class_labels_map = get_class_labels_map(examples)
    majority_class = get_majority_class(class_labels_map)
        
    # Returns a leaf node with majority class label if features are empty
    if features is None or len(features) == 0:
        return create_leaf_node(examples, features, majority_class)
    
    # Return a leaf node if all instances have the same classification
    if len(class_labels_map.keys()) == 1:
        pure_class_label = class_labels_map.keys()[0]
        return create_leaf_node(examples, features, pure_class_label)

    # Check if number of instances remaining is less than threshold. If yes, create a leaf node
    if len(examples) < leaf_threshold:
        if majority_class is None:
            return create_leaf_node(examples, features, default_class)
        else:
            return create_leaf_node(examples, features, majority_class)
    
    # Choose the best feature
    best_feature = get_best_feature(examples, features)
    # This can happen when no feature has positive information gain.
    if best_feature is None:
        return create_leaf_node(examples, features, majority_class)
    
    root = create_feature_node(examples, features, best_feature)
    
    # Create links for each feature value.
    feature_value_nodes = []
    
    # Handle nominal/discrete features first
    if best_feature.is_nominal_feature():
        for feature_value in best_feature.get_feature_values():
            filtered_examples = get_examples_with_feature_value(examples, feature_value, 
                                                                best_feature, constants.EQUALS)
            filtered_features = [feature for feature in features if feature != best_feature]
            feature_value_node = _build_dtree(filtered_examples, filtered_features, default_class, leaf_threshold)
            
            feature_value_node.parent_feature_name = best_feature.name
            feature_value_node.parent_feature_value = " = " + str(feature_value)
            feature_value_nodes.append(feature_value_node)
    # Handle numeric features here
    else:
        best_split_value = get_best_split_threshold(examples, best_feature)
        
        less_than_value_examples = get_examples_with_feature_value(examples, best_split_value, best_feature, constants.LESS_THAN_OR_EQUAL_TO)
        greater_than_value_examples = get_examples_with_feature_value(examples, best_split_value, best_feature, constants.GREATER_THAN)
        
        less_than_feature_value_node = _build_dtree(less_than_value_examples, features, default_class, leaf_threshold)
        
        less_than_feature_value_node.parent_feature_name = best_feature.name
        less_than_feature_value_node.parent_feature_value = " <= " + str("{0:.6f}".format(best_split_value))
        less_than_feature_value_node.is_node_for_nominal_feature = False
        
        greater_than_feature_value_node = _build_dtree(greater_than_value_examples, features, default_class, leaf_threshold)
        
        greater_than_feature_value_node.parent_feature_name = best_feature.name
        greater_than_feature_value_node.parent_feature_value = " >  " + str("{0:.6f}".format(best_split_value))
        greater_than_feature_value_node.is_node_for_nominal_feature = False
        
        feature_value_nodes.append(less_than_feature_value_node)      
        feature_value_nodes.append(greater_than_feature_value_node)  
    
    root.feature_value_nodes = feature_value_nodes
     
    return root

# Pretty prints the decision tree
def print_dtree(root, op_labels, prefix):
    if root is None or root.feature_value_nodes is None:
        return
    
    root_child_nodes = root.feature_value_nodes
    
    for node in root_child_nodes:
        label_count_display = node.class_labels_count(op_labels)
        display_str = prefix + " " + node.parent_feature_name + " " + node.parent_feature_value + " " + label_count_display
        if node.node_type == constants.CLASS_NODE:
            display_str += " : " + node.class_label
            
        print display_str
        
        # Intend the child nodes by a tab and prefix the tab with a pipe separator to link various
        # values of a feature vertically.
        print_dtree(node, op_labels, prefix + "|\t")

"""
    Evaluate the decision tree against all examples in a test data set. Print the example,
    actual class label and predicted class label for each example.
"""
def test_dtree(dtree_root, test_dataset):
    
    test_examples = test_dataset.examples
    default_class_label = test_dataset.output_labels[0]      
    total_examples = len(test_examples)
    correctly_classified_examples = 0
    
    features = test_dataset.features
    for example in test_examples:
        predicted_class_label = get_predicted_class_label_for_example(dtree_root, example, default_class_label)
        actual_class_label = example.class_label
        if actual_class_label == predicted_class_label:
            correctly_classified_examples += 1
        
        example_str = ""
        for feature in features:
            example_str = example_str + example.get_value_for_feature(feature) + " "
        example_str = example_str + " | " + actual_class_label
        example_str = example_str + " | " + predicted_class_label
        print example_str
        
    print "\nCorrectly classified examples : " + str(correctly_classified_examples)
    print "Total test examples : " + str(total_examples)
    
    test_set_accuracy = correctly_classified_examples/float(total_examples)
    print "Test set accurancy : " + "{0:.2f}".format(test_set_accuracy*100) + " % "
    return test_set_accuracy
    
# Predicts the class label of an example with the decision tree constructed using training data set
def get_predicted_class_label_for_example(dtree_root, example, default_class_label):
    
    # In case the decision tree is empty, return the default class label which is the first class
    # label in the dataset file.
    if dtree_root is None:
        return default_class_label
    
    # If leaf/class node has reached then return the class label at leaf node
    predicted_class_label = None
    if dtree_root.node_type == constants.CLASS_NODE:
        predicted_class_label = dtree_root.class_label
    else:
        curr_feature = dtree_root.best_feature
        curr_feature_example_value = example.get_value_for_feature(curr_feature)
        
        # next node which has to be traversed in the d-tree
        dtree_child = None
        
        # For nominal features, just do a value lookup
        child_nodes = dtree_root.feature_value_nodes
        if curr_feature.is_nominal_feature():
            for child in child_nodes:
                # TODO : Bad encoding. This should be removed.
                parent_feature_value = get_decoded_node_value(child.parent_feature_value)
                if parent_feature_value == curr_feature_example_value:
                    dtree_child = child
                    break
        else:
            less_than_equal_to_child = child_nodes[0]
            greater_than_child = child_nodes[1]
            
            # TODO : This is bad. Weird encoding to get the results
            split_value = get_decoded_node_value(less_than_equal_to_child.parent_feature_value);
            if float(curr_feature_example_value) <= float(split_value):
                dtree_child = less_than_equal_to_child
            else:
                dtree_child = greater_than_child
        
        predicted_class_label = get_predicted_class_label_for_example(dtree_child, example, default_class_label)

    return predicted_class_label

# Removes the operator encoding from the node value to return the numeric value. While  storing
# the value at d-tree node, I also encoded the operator along with it. This was a bad decision
# which I could have avoided by also storing the operator seprately from the value.
def get_decoded_node_value(value):
    decoded_value = value
    decoded_value = decoded_value.replace("<", "")
    decoded_value = decoded_value.replace(">", "")
    decoded_value = decoded_value.replace(" ", "")
    decoded_value = decoded_value.replace("=", "")
    
    return decoded_value.strip()

# Creates a leaf node in the decision tree
def create_leaf_node(examples, features, class_label):
    node = DecisionTreeNode(examples, features)
    node.best_feature = None
    node.class_label = class_label
    node.node_type = constants.CLASS_NODE
    
    return node
    
# Creates an intermediate node in the decision tree    
def create_feature_node(examples, features, best_feature):
    node = DecisionTreeNode(examples, features)
    node.best_feature = best_feature
    node.node_type = constants.FEATURE_NODE
    
    return node

def pretty_print_features(features):
    features_print = [feature.name for feature in features]
    return str(features_print)
    
"""
    Core algorithm to determine the best feature
    
    Rules :
    1) In case two best features have the same information gain, choose the one which appeared
    before in the ARFF file.
    
"""    
def get_best_feature(examples, features):
    best_feature = None
    max_info_gain = -999
    for feature in features:
        info_gain = None
        if feature.is_nominal_feature():
            info_gain = get_info_gain_for_nominal_feature(examples, feature)
        else:
            info_gain = get_best_info_gain_for_numeric_feature(examples, feature)
        
        if info_gain > max_info_gain:
            best_feature = feature
            max_info_gain = info_gain
       
    # Best feature is only applicable for any positive information gain
    if max_info_gain <= 0:
        best_feature = None
    
    return best_feature

# Gets the information gain for a nominal feature
def get_info_gain_for_nominal_feature(examples, feature):
    total_examples_cnt = len(examples)
    total_info = get_info(examples, feature)
    
    feature_info = 0.0
    for value in feature.get_feature_values():
        filtered_examples = get_examples_with_feature_value(examples, value, feature, constants.EQUALS)
        filtered_examples_cnt = len(filtered_examples)
        feature_value_info = (filtered_examples_cnt/float(total_examples_cnt))*get_info(filtered_examples, feature)
        feature_info += feature_value_info

    return total_info - feature_info

# Gets the information gain for a numeric feature by splitting at a threshold that maximizes the information gain
def get_best_info_gain_for_numeric_feature(examples, feature):
    total_info = get_info(examples, feature)
        
    best_split_value = get_best_split_threshold(examples, feature)
    # Couldn't find any suitable split value. Ignore this feature
    if best_split_value is None:
        return 0.0
    
    feature_info = get_info_for_numeric_value_split(examples, feature, best_split_value)

    return total_info - feature_info

# What is the information contained in this numeric value ?    
def get_info_for_numeric_value_split(examples, feature, value):
    total_examples_cnt = len(examples)
        
    feature_info = 0.0
    less_than_value_examples = get_examples_with_feature_value(examples, value, feature, constants.LESS_THAN_OR_EQUAL_TO)
    greater_than_value_examples = get_examples_with_feature_value(examples, value, feature, constants.GREATER_THAN)
    
    less_than_value_examples_cnt = len(less_than_value_examples)
    greater_than_value_examples_cnt = len(greater_than_value_examples)
    
    feature_info += (less_than_value_examples_cnt/float(total_examples_cnt))*get_info(less_than_value_examples, feature)
    feature_info += (greater_than_value_examples_cnt/float(total_examples_cnt))*get_info(greater_than_value_examples, feature)
    
    return feature_info        

# Gets the total information contained in the examples
def get_info(examples, feature):
    if examples is None or len(examples) == 0:
        return 0.0
    
    total_examples = len(examples)
    class_labels_map = get_class_labels_map(examples)
    
    class_labels = class_labels_map.keys()
    first_class_examples = 0 
    second_class_examples = 0
    first_class_examples = class_labels_map[class_labels[0]]
    if len(class_labels) > 1:
        second_class_examples = class_labels_map[class_labels[1]]
    
    # Calculate the total information at this node
    first_class_info, second_class_info = 0.0, 0.0
    if first_class_examples > 0:
        first_class_info = -(first_class_examples/float(total_examples))*math.log(first_class_examples/float(total_examples), 2)
    if second_class_examples > 0:
        second_class_info = -(second_class_examples/float(total_examples))*math.log(second_class_examples/float(total_examples), 2)
    total_info =  first_class_info + second_class_info

    return total_info
        
# Determine the point of split for a numeric feature which maximizes information gain    
def get_best_split_threshold(examples, feature):
    best_split_threshold = None
    best_info_gain = -999.0
    candidate_split_values = get_candidate_split_values(examples, feature)
    # Couldn't find any candidate values to split on. Ignore this feature
    if candidate_split_values is None or len(candidate_split_values) == 0:
        return None
    
    #print "Split candidates for feature " + feature.name + " are " + str(candidate_split_values) + " for examples " + str(len(examples))
    for split_value in candidate_split_values:
        info_gain = get_info(examples, feature) - get_info_for_numeric_value_split(examples, feature, split_value)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split_threshold = split_value
    
    return best_split_threshold

# Determines the set of candidate splits points for a numeric feature
def get_candidate_split_values(examples, feature):
    candidate_split_values = []
    
    # build a value bucketed data structure first
    example_value_map = {}
    for example in examples:
        feature_value = float(example.get_value_for_feature(feature))
        if feature_value in example_value_map:
            feature_value_examples = example_value_map[feature_value]
        else:
            feature_value_examples = []
         
        feature_value_examples.append(example)
        example_value_map[feature_value] = feature_value_examples
    
    feature_values_sorted = sorted(example_value_map.keys())
    
    # If only one numeric value for this feature, then information gain is 0. So, simply return.
    if len(feature_values_sorted) <= 1:
        return []
    
    prev_feature_value = feature_values_sorted[0]
    for value in feature_values_sorted[1:]:
        next_feature_value = value
        prev_value_examples = example_value_map[prev_feature_value]
        next_value_examples = example_value_map[next_feature_value]
        
        is_eligible_for_split = is_eligible_for_numeric_split(prev_value_examples, next_value_examples)
        if is_eligible_for_split:
            candidate_split_values.append(float(prev_feature_value + next_feature_value) / 2.0)
        
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

    first_class_label_instances = 0
    second_class_label_instances = 0
    first_class_label_instances = class_labels_map[class_labels[0]]
    if len(class_labels) > 1:
        second_class_label_instances = class_labels_map[class_labels[1]]
        
    if first_class_label_instances == second_class_label_instances:
        return None
    elif first_class_label_instances > second_class_label_instances:
        return class_labels[0]
    else:
        return class_labels[1]
    
# Filter out all the examples with the specific value for a feature    
def get_examples_with_feature_value(examples, feature_value, feature, operator):
    filtered_examples = []
    
    for example in examples:
        example_value = example.get_value_for_feature(feature)
        if operator == constants.EQUALS and example_value == feature_value:
            filtered_examples.append(example)
        elif operator == constants.LESS_THAN_OR_EQUAL_TO and float(example_value) <= float(feature_value):
            filtered_examples.append(example)
        elif operator == constants.GREATER_THAN and float(example_value) > float(feature_value):
            filtered_examples.append(example)
        else:
            pass
        
    return filtered_examples    