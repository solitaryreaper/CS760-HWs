'''
Created on Oct 4, 2013

@author: excelsior
'''

import sys, getopt, re
from models import dataset, dtree, constants

"""
    Implementation of the decision tree learning algorithm involving both nominal and numeric 
    features. The dataset is assumed to be present in the ARFF format.
"""

""" 
    Read the dataset file and build the internal memory data structures containing all
    the instances data.
"""
def get_dataset_from_file(file_path):
    file = open(file_path, 'r')

    is_data_read_started = False
    dataset_name = None
    features = []
    output_labels = []
    examples = []
      
    for line in file:
        if not line.strip() or line.startswith(constants.COMMENT_MARKER):
            continue
                    
        # Remove all white-space characters
        line = re.sub( '\s+', ' ', line).strip()
        
        # Read data line here
        if is_data_read_started:
            example = get_example_from_line(line, features)
            examples.append(example)
        # Read metadata line here
        else:
            # read dataset name
            if line.startswith(constants.DATASET_NAME_MARKER):
                dataset_name = get_line_without_marker(line, constants.DATASET_NAME_MARKER)
            # read output class labels
            elif constants.CLASS_LABEL_MARKER in line:
                class_line = get_line_without_marker(line, constants.CLASS_LABEL_MARKER)
                split_line = class_line.split("{")
                output_labels = split_line[1].replace("{", "").replace("}", "").strip().split(constants.VALUE_DELIMITER)
                output_labels = [label.strip() for label in output_labels]
            # read features
            elif line.startswith(constants.FEATURE_NAME_MARKER):
                feature = get_feature_from_line(line)
                features.append(feature)
            elif line.startswith(constants.DATA_MARKER):
                is_data_read_started = True
            else:
                pass
        
    file.close()
    
    return dataset.Dataset(dataset_name, features, output_labels, examples)
        
# Remove leading marker from a line in data file and trim any extra spaces
def get_line_without_marker(line, marker):
    return line.replace(marker, "").strip()

# Extracts example from an example line in ARFF data file
def get_example_from_line(line, features):
    split_line = line.split(constants.VALUE_DELIMITER)
    
    feature_value_dict = {}
    for index in range(len(features)):
        feature_value_dict[features[index].name] = split_line[index]
        
    class_label = split_line[-1] # class label is at the end of the example line
    
    return dataset.Example(feature_value_dict, class_label)

# Extracts feature from a feature line in ARFF data file
def get_feature_from_line(line):
    line_wo_marker = get_line_without_marker(line, constants.FEATURE_NAME_MARKER)
    line_wo_marker = line_wo_marker.replace("'", "")
    line_wo_marker = re.sub( '\s+', ' ', line_wo_marker).strip()
    tokens = line_wo_marker.split("{")

    name, type, data_type, values = None, None, None, None
        
    # Numeric feature
    if len(tokens) == 1:
        temp_tokens = tokens[0].split(" ")
        name = temp_tokens[0].strip()
        data_type = temp_tokens[1].strip()
        type = constants.NUMERIC_FEATURE
    # Nominal feature
    else:
        name = tokens[0].strip()
        temp_tokens = tokens[1].replace("{", "").replace("}", "").strip().split(constants.VALUE_DELIMITER)
        values = [value.strip() for value in temp_tokens]
        type = constants.NOMINAL_FEATURE
    
    return dataset.Feature(name, type, data_type, values)

# Program driver
def main(argv):
    assert len(argv) == 3, " Please provide correct arguments as follows : python dt-learn.py <train file path> <test file path> <leaf threshold>"
    train_dataset_file_path, test_dataset_file_path, leaf_threshold = argv[0], argv[1], int(argv[2])
    
    # 1) load the training data set
    train_dataset = get_dataset_from_file(train_dataset_file_path)
    
    # 2) generate a decision tree using training data set
    training_dataset_dtree = dtree.learn_dtree(train_dataset, leaf_threshold)
    
    print "\n\n=================== DECISION TREE =====================================\n\n"
    dtree.print_dtree(training_dataset_dtree, train_dataset.output_labels, " ")
    
    # 3) load the test data set
    test_dataset = get_dataset_from_file(test_dataset_file_path)
    
    # 4) evaluate the decision tree using the test data set
    print "\n\n================== TEST DATA SET EVALUATION ==========================\n\n"
    dtree.test_dtree(training_dataset_dtree, test_dataset)

if __name__ == '__main__':
    main(sys.argv[1:])