
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.


'''
import numpy as np
import sklearn



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10155856, 'Mackenzie', 'Wilson'), (10157182, 'Nicole', 'Barritt') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    # Read the file elements (separated by commas) into a np array.
    file_as_array = np.genfromtxt(dataset_path, dtype='str', delimiter=',')
    '''
    Array format: (all as strings currently)
             0           1         2         3       ....
    [ 
     0    [ ID#,  Class label,  feature1,  feature2, .... ] 
     1     ...
    ]
    '''
    
    # Store the file's shape as variables to use later.
    num_examples = file_as_array.shape[0]
    num_features = file_as_array.shape[1]
    
    # Create an array to store all file data except the class labels (col 1).
    X = np.zeros((num_examples, num_features-1)) #dtype = float (default)
    X[:,0] = file_as_array.copy()[:,0] #automatically converts to floats
    X[:,1:] = file_as_array[:,2:] 
    
    # Create a 1D array to store all the class labels ('B' or 'M').
    y = file_as_array.copy()[:,1]
    # Use a mask to change to binary, where M=1 (True).
    y[y=='M'] = 1
    y[y=='B'] = 0
    
    
    return X,y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''     
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def random_split(X, y, ratio):
    '''
    Split the given data (and corresponding labels) into training and testing 
     datasets based on the ratio given. 
        
    @param
     X : 2D np.ndarray, X[i,:] is the ith example
     y : 1D np.ndarray, y[i] is the class label of X[i,:]
     ratio : ratio of training:testing data (i.e. 0.8 represents 80% training
             and 20% testing)
    
    @return:
     X_train : 2D array of examples for training
     y_train : 1D array of corresponding labels for training
     X_test : 2D array of examples for testing
     y_test : 1D array of corresponding labels for training
    '''
    assert(ratio>0 and ratio<1)
    assert(X.ndim==2 and y.ndim==1)
    
    # Create small and large arrays of random indexes according to ratio.
    indexes = np.arange(len(y))
    np.random.shuffle(indexes)
    train_idx = indexes[:int(len(indexes)*ratio)]
    test_idx = indexes[int(len(indexes)*ratio):]
    
    X_train = X.copy()[train_idx]
    y_train = y.copy()[train_idx]
    X_test = X.copy()[test_idx]
    y_test = y.copy()[test_idx]
    
    return X_train, y_train, X_test, y_test
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
   # Print the team.
    print(my_team())
    
    # Pre-process the dataset.
    data, labels = prepare_dataset('medical_records.data')
    #print(data)
    #print(labels)
    
    # Split the dataset into the corresponding ratio for crossvalidation. 
    traindata, trainlabel, testdata, testlabel = random_split(data, labels, 0.8)
    print(testlabel)


