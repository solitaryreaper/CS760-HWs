'''
Created on Nov 20, 2013

@author: excelsior
'''

'''
    Tests the naive bayes and tree augmented naive bayes algorithm. CS760 HW-3
'''

import sys
from utils import file_reader
from algos import naive_bayes

# Program driver
def main(argv):
    assert len(argv) == 3, " Please provide correct arguments as follows : python dt-learn.py <train file path> <test file path> <n|t>"
    train_dataset_file_path, test_dataset_file_path = argv[0], argv[1]
    is_naive_bayes_reqd = True if argv[2] == 'b' else False
    
    '''
    print "Arguments : "
    print "Train file " + train_dataset_file_path
    print "Test file " + test_dataset_file_path
    print "Invoke Naive Bayes ? " + str(is_naive_bayes_reqd)
    '''    
        
    # 1) load the training data set
    #print "\nLoading the training dataset .."
    train_dataset = file_reader.get_dataset_from_file(train_dataset_file_path)
    #print str(train_dataset)
    
    # load the test data set
    #print "\nLoading the test dataset .."
    test_dataset = file_reader.get_dataset_from_file(test_dataset_file_path)
        
    # Trigger the appropriate bayesian model based on input parameter
    if is_naive_bayes_reqd:
        #print "\nGenerating the naive bayes model .."
        naive_bayes_model = naive_bayes.Naivebayes(train_dataset)
        #print str(test_dataset)
        
        #print "\nEvaulating the test dataset .."
        naive_bayes.evaluate_naive_bayes_model(naive_bayes_model, test_dataset)
    
if __name__ == '__main__':
    main(sys.argv[1:])