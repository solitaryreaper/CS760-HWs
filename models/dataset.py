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
    
    def __init__(self, name, features, output_labels, examples):
        self.name = name
        self.features = features
        self.output_labels = output_labels
        self.examples = examples
    
    def get_dataset_name(self):
        return self.name
    
    def get_features(self):
        return self.features
    
    def get_output_labels(self):
        return self.output_labels
    
    def get_all_examples(self):
        return self.examples
    
    def get_examples_by_values(self, feature_name, feature_value):
        pass
    
    def get_examples_by_output_labels(self, output_label):
        pass

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
            
    def is_nominal_feature(self):
        return self.type == constants.NOMINAL_FEATURE
    

"""
    Model object to represent a dataset example
"""
class Example(object):
    
    def __init__(self, feature_val_dict, class_label):
        self.feature_val_dict = feature_val_dict
        self.class_label = class_label

    def __str__(self):
        return "Example : Class Label=%s, Values=%s" % (self.class_label, self.feature_val_dict)    
