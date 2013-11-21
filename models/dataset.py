'''
Created on Oct 5, 2013

@author: excelsior
'''

import constants

"""
    A container object that represents the dataset used in the decision tree learning
    algorithm.
"""
class Dataset(object):
    
    def __init__(self, name, features, output, examples):
        self.name = name
        self.features = features
        self.output = output
        self.examples = examples
    
    def __str__(self):
        return "Name: " + self.name + ", #Features : " + str(len(self.features)) + ", #Labels : " + str(len(self.output.values)) + ", #Examples : " + str(len(self.examples))
        
    def get_dataset_name(self):
        return self.name
    
    def get_features(self):
        return self.features
    
    def get_all_feature_values_count(self):
        count_all_feature_values = 0
        for f in self.features:
            count_all_feature_values = count_all_feature_values + len(f.get_feature_values())
            
        return count_all_feature_values
    def get_output(self):
        return self.output
    
    def get_output_labels(self):
        return self.output.get_output_attribute_values()
    
    def get_examples(self):
        return self.examples
    
    # Number of example instances in this dataset
    def get_examples_count(self):
        return len(self.examples)
    
    # Default class label for the dataset is the first mentioned class label
    def get_default_class_label(self):
        return self.output.values[0]
    
    # Returns the number of examples with a particular class label in this dataset
    def get_count_examples_with_label(self, label):
        count_label_examples = 0
        for example in self.examples:
            if example.class_label == label:
                count_label_examples = count_label_examples + 1
                
        return count_label_examples
    
    # Returns the number of examples in the dataset with specified class label and a 
    # feature value
    def get_count_examples_with_label_and_feature_value(self, label, feature, feature_value):
        count_label_and_feature_value_examples = 0;
        for example in self.examples:
            if example.class_label == label:
                value = example.get_value_for_feature(feature)
                if value == feature_value:
                    count_label_and_feature_value_examples = count_label_and_feature_value_examples + 1
                    
        return count_label_and_feature_value_examples

"""
    Model object to represent the output label object
"""
class Label(object):
    def __init__(self, op_attr_name, op_attr_values):
        self.name = op_attr_name
        self.values = op_attr_values
        
    def get_output_attribute_name(self):
        return self.name
    
    def get_output_attribute_values(self):
        return self.values

"""
    Model object to represent metadata and values of a feature
"""
class Feature(object):
    
    def __init__(self, name, feature_type, data_type, values):
        self.name = name
        self.type = feature_type
        self.data_type = data_type
        self.values = values

    def __str__(self):
        return "Feature : Name=%s, Type=%s, Data Type=%s, Values=%s" % (self.name, self.type, self.data_type, self.values)
    
    def __eq__(self, other): 
        return self.name == other.name    
            
    def __hash__(self):
        return hash(self.name)
                
    def is_nominal_feature(self):
        return self.type == constants.NOMINAL_FEATURE
    
    def get_feature_values(self):
        return self.values
    
"""
    Model object to represent a dataset example
"""
class Example(object):
    
    def __init__(self, feature_val_dict, class_label):
        self.feature_val_dict = feature_val_dict
        self.class_label = class_label

    def __str__(self):
        return "Example : Class Label=%s, Values=%s" % (self.class_label, self.feature_val_dict)
    
    # Returns the value of a specific feature from this example
    def get_value_for_feature(self, feature):
        feature_value = None
        if feature in self.feature_val_dict:
            feature_value = self.feature_val_dict[feature]
          
        return feature_value
    
    # Returns the value of a specific feature from this example
    def get_value_for_feature_name(self, feature_name):
        feature_value = None
        if feature_name in self.feature_val_dict:
            feature_value = self.feature_val_dict[feature_name]
          
        return feature_value    
