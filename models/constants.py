'''
Created on Oct 5, 2013

@author: excelsior
'''

"""
    Defines all the constants used in decision tree learning modules
"""

# Constants representing various markers in data file
DATASET_NAME_MARKER, FEATURE_NAME_MARKER, CLASS_LABEL_MARKER, DATA_MARKER = "@relation" , "@attribute", "class", "@data"
COMMENT_MARKER = "%"

VALUE_DELIMITER = ","

# Constants representing feature types
NOMINAL_FEATURE, NUMERIC_FEATURE, CLASS_FEATURE = "nominal" , "numeric", "class"

# Constants representing data type of a numeric feature
REAL_DATA_TYPE, NUMERIC_DATA_TYPE = "real" , "numeric"

# Decision tree node types
FEATURE_NODE , CLASS_NODE = "feature", "class"

#Operators
EQUALS, LESS_THAN_OR_EQUAL_TO, GREATER_THAN = "=", "<=", ">"