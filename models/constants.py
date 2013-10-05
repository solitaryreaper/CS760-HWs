'''
Created on Oct 5, 2013

@author: excelsior
'''

"""
    Defines all the constants
"""

# Constants representing various markers in data file
DATASET_NAME_MARKER, FEATURE_NAME_MARKER, CLASS_LABEL_MARKER, DATA_MARKER = "@relation" , "@attribute", "class", "@data"

VALUE_DELIMITER = ","

# Constants representing feature types
NOMINAL_FEATURE, NUMERIC_FEATURE = "nominal" , "numeric"

# Constants representing data type of a numeric feature
REAL_DATA_TYPE, NUMERIC_DATA_TYPE = "real" , "numeric"