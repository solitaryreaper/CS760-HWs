'''
Created on Oct 8, 2013

@author: excelsior
'''

dtlearn = __import__("dt-learn")

def parse_sample_data_file(file_path):
    dataset = dtlearn.get_dataset_from_file(file_path)
    print "Examples : " + str(len(dataset.examples))
    
if __name__ == '__main__':
    parse_sample_data_file("data/credit-a.arff")