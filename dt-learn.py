'''
Created on Oct 4, 2013

@author: excelsior
'''

import sys, getopt
from models import dataset
from models import constants

"""
    Implementation of the decision tree learning algorithm involving
    both nominal and numeric features. The dataset is assumed to be
    present in the ARFF format.
"""

""" 
    Read the dataset file and build the internal memory data structures containing all
    the instances data
"""
def get_dataset_from_file(file_path):
    file = open(file_path, 'r')

    is_data_read_started = False
    dataset_name = None
    features = []
    output_labels = []
    examples = []
      
    for line in file:
        if not line:
            continue
        
        # Read data line here
        if is_data_read_started:
            example = get_example_from_line(line, features)
            print str(example)
            examples.append(example)
        # Read metadata line here
        else:
            if line.startswith(constants.DATASET_NAME_MARKER):
                dataset_name = get_line_without_marker(line, constants.DATASET_NAME_MARKER)
                print str(dataset_name)
            elif constants.CLASS_LABEL_MARKER in line:
                class_line = get_line_without_marker(line, constants.CLASS_LABEL_MARKER)
                split_line = class_line.split("'")
                output_labels = split_line[2].replace("{", "").replace("}", "").strip().split(constants.VALUE_DELIMITER)
                output_labels = [label.strip() for label in output_labels]
                print str(output_labels)                
            elif line.startswith(constants.FEATURE_NAME_MARKER):
                feature = get_feature_from_line(line)
                print str(feature)
                features.append(feature)
            elif line.startswith(constants.DATA_MARKER):
                is_data_read_started = True
            else:
                pass
        
    file.close()
    
    #return Dataset(dataset_name, features, output_labels, examples)
        
#Utility methods

# Remove leading marker from a line in data file and trim any extra spaces
def get_line_without_marker(line, marker):
    return line.replace(marker, "").strip()

# Returns an example object corresponding to the current instance in the data file
def get_example_from_line(line, features):
    line_wo_marker = get_line_without_marker(line, constants.FEATURE_NAME_MARKER)
    split_line = line_wo_marker.split(constants.VALUE_DELIMITER)
    
    feature_value_dict = {}
    for index in range(len(features)):
        feature_value_dict[features[index].name] = split_line[index]
    class_label = split_line[-1]
    
    return dataset.Example(feature_value_dict, class_label)

def get_feature_from_line(line):
    line_wo_marker = get_line_without_marker(line, constants.FEATURE_NAME_MARKER)
    split_line = line_wo_marker.split("'")
    
    name, type, data_type, values = None, None, None, None
    name = split_line[1].replace("'", "").strip()
    if "{" in split_line[2]:
        type = constants.NOMINAL_FEATURE
        values = split_line[2].replace("{", "").replace("}", "").strip().split(constants.VALUE_DELIMITER)
        values = [value.strip() for value in values]
    else:
        type = constants.NUMERIC_FEATURE
        data_type = split_line[2].strip()
    
    return dataset.Feature(name, type, data_type, values)

# Program driver
def main(argv):
    assert len(argv) == 3, " Please provide correct arguments : python dt-learn.py $1 $2 $3"
    train_file_path, test_file_path, leaf_threshold = argv[0], argv[1], argv[2]
    
    # 1) load the training data set
    get_dataset_from_file(train_file_path)
    
    # 2) generate a decision tree using training data set
    
    # 3) load the test data set
    
    # 4) evaluate the decision tree using the test data set

if __name__ == '__main__':
    main(sys.argv[1:])