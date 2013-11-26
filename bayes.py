'''
Created on Nov 20, 2013

@author: excelsior
'''

'''
    Implements and tests the naive bayes and tree augmented naive bayes algorithm. CS760 HW-3
'''

import sys
from utils import file_reader
from algos import naive_bayes, tan

import logging, time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Program driver
def main(argv):
    assert len(argv) == 3, " Please provide correct arguments as follows : python bayes.py <train file path> <test file path> <n|t>"
    assert(argv[2] == 'n' or argv[2] == 't')

    train_dataset_file_path, test_dataset_file_path = argv[0], argv[1]
    is_naive_bayes_reqd = True if argv[2] == 'n' else False
        
    logger.debug("Loading the training dataset ..")
    train_dataset = file_reader.get_dataset_from_file(train_dataset_file_path)
    
    logger.debug("Loading the test dataset ..")
    test_dataset = file_reader.get_dataset_from_file(test_dataset_file_path)
        
    if is_naive_bayes_reqd:
        start = time.clock()
        logger.debug("Generating the naive bayes model using training dataset ..")
        naive_bayes_model = naive_bayes.Naivebayes(train_dataset)
        
        logger.debug("Evaulating the test dataset on Naive Bayes Model ..")
        naive_bayes.evaluate_naive_bayes_model(naive_bayes_model, test_dataset)
        
        logger.debug("Time taken for running Naive Bayes Model is " + str(time.clock() - start) + " s ")
    else:
        start = time.clock()
        logger.debug("Generating the TAN model using training dataset ..")
        tan_model = tan.TAN(train_dataset)
        
        logger.debug("Evaulating the test dataset on TAN model ..")        
        tan.evaluate_tan_model(tan_model, test_dataset)
        
        logger.debug("Time taken for running TAN Model is " + str(time.clock() - start) + " s ")
    
if __name__ == '__main__':
    main(sys.argv[1:])