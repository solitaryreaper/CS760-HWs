'''
Created on Nov 20, 2013

@author: excelsior
'''

from models import constants, dataset
import re

""" 
    Read the dataset file and build the internal memory data structures containing all
    the instances data.
"""
def get_dataset_from_file(file_path):
    data_file = open(file_path, 'r')

    is_data_read_started = False
    dataset_name = None
    features = []
    output_labels = []
    examples = []
      
    for line in data_file:
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
        
    data_file.close()
    
    output_attribute = dataset.Label(constants.CLASS_LABEL_MARKER, output_labels)
    return dataset.Dataset(dataset_name, features, output_attribute, examples)
        
# Remove leading marker from a line in data file and trim any extra spaces
def get_line_without_marker(line, marker):
    return line.replace(marker, "").strip()

# Extracts example from an example line in ARFF data file
def get_example_from_line(line, features):
    split_line = line.split(constants.VALUE_DELIMITER)
    
    feature_value_dict = {}
    for index in range(len(features)):
        feature_value_dict[features[index]] = split_line[index]
        
    class_label = split_line[-1] # class label is at the end of the example line
    
    return dataset.Example(feature_value_dict, class_label)

# Extracts feature from a feature line in ARFF data file
def get_feature_from_line(line):
    line_wo_marker = get_line_without_marker(line, constants.FEATURE_NAME_MARKER)
    line_wo_marker = line_wo_marker.replace("'", "")
    line_wo_marker = re.sub( '\s+', ' ', line_wo_marker).strip()
    tokens = line_wo_marker.split("{")

    name, feature_type, data_type, values = None, None, None, None
        
    # Numeric feature
    if len(tokens) == 1:
        temp_tokens = tokens[0].split(" ")
        name = temp_tokens[0].strip()
        data_type = temp_tokens[1].strip()
        feature_type = constants.NUMERIC_FEATURE
    # Nominal feature
    else:
        name = tokens[0].strip()
        temp_tokens = tokens[1].replace("{", "").replace("}", "").strip().split(constants.VALUE_DELIMITER)
        values = [value.strip() for value in temp_tokens]
        feature_type = constants.NOMINAL_FEATURE
    
    return dataset.Feature(name, type, data_type, values)