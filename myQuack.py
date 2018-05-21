
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.



USEFUL SKLEARN FUNCTIONS:
    neighbors.KNeighborsClassifier
    
    naive_bayes.GaussianNB
    
    model_selection.cross_validate
    
    svm.SVC
    
    tree.DecisionTreeClassifier
    
    model_selection.train_test_split
    
    metrics.classification_report
    metrics.confusion_matrix




'''
import numpy as np
import pandas as pd
from sklearn import naive_bayes, neighbors, tree, svm, model_selection, metrics



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
    
    # Store the file's shape as variables to use later.
    num_examples = file_as_array.shape[0]
    num_features = file_as_array.shape[1]
    
    # Create an array to store all file data except the class labels (col 1).
    X = np.zeros((num_examples, num_features-1)) #dtype = float (default)
    X[:,0] = file_as_array.copy()[:,0] #automatically converts to floats
    X[:,1:] = file_as_array[:,2:] 
    
    # Create a 1D array to store all the class labels ('B' or 'M').
    y = np.zeros_like(file_as_array[:,1], dtype=int)
    for i in range(len(y)):
        # Store a binary 1 for M, 0 for B
        y[i] = (file_as_array[i,1]=='M')
    
    
    return X,y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training, params):
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

def build_DT_classifier(X_training, y_training, params):
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

    knn = neighbors.KNeigborsClassifier()    
    clf = model_selection.GridSearchCV(knn, params, cv=4) 
    clf.fit(X_training, y_training)
    
    return clf #automatically uses the best estimator for predictions
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

    
def split_data(data, label, ratio):
    '''
    Split the given data (and corresponding labels) into training and testing 
     datasets based on the ratio given. Repeatable and non-random.
        
    @param
     data : 2D np.ndarray, X[i,:] is the ith example
     labels : 1D np.ndarray, y[i] is the class label of X[i,:]
     ratio : decimal ratio of training:testing data (i.e. 0.8 represents 80% 
             training and 20% testing)
    
    @return:
     data_train : 2D array of examples for training
     label_train : 1D array of corresponding labels for training
     data_test : 2D array of examples for testing
     label_test : 1D array of corresponding labels for training
    '''
    assert(ratio>0 and ratio<1)
    assert(data.ndim==2 and label.ndim==1)
    
    # Create small and large arrays of data and labels according to ratio.
    data_train = data.copy()[:int(len(data)*ratio)]
    label_train = label.copy()[:int(len(data)*ratio)]
    data_test = data.copy()[int(len(data)*ratio):]
    label_test = label.copy()[int(len(data)*ratio):]
    
    return data_train, label_train, data_test, label_test
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def print_prediction_report(y_true, y_pred, names):
    '''
    Return a bunch of statistics and metrics reporting the performance of a 
     certain classifier model on the given training data. 
     
    @param:
        y_true: A np-array of the target class labels as integers
        y_pred: A np-array of the classifier-predicted class labels as integers
        names: A tuple of the class labels (str), corresponding to (1,0) 
               binary integer class labels.
               
    @return:
        None. Print to console. 
    '''
    
    #classification report
    cr = metrics.classification_report(y_true, y_pred, labels, target_names=names)
    #confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, labels = (1,0))
    display_confusion_matrix(cm, names)

    raise NotImplementedError


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def display_confusion_matrix(cm, labels):
    '''
    Print the confusion matrix values in a nicely formatted table. This is all 
     ugly code so it is separated out and put into this callable function.
    
    @param:
        cm: a 2x2 array
        labels: a tuple of 2 labels corresponding to (1,0) binary markings
        
    @return:
        None. Print to console. 
    '''
    
    assert len(labels)==len(cm)
    assert cm.shape == (2,2)
    
    print('{:14} {:10} {:10} {:3}'.format('PREDICTED:',labels[0], labels[1], 'All'))
    print("\nACTUAL: ")
    print('\n{:14} {:<10} {:<10} {:3}'.format('',"TP:", "FN:", ''))
    print('{:14} {:<10} {:<10} {:<3}'.format(labels[0],cm[0,0], cm[0,1], sum(cm[0])))
    print('\n{:14} {:<10} {:<10} {:3}'.format('',"FP:", "TN:", ''))
    print('{:14} {:<10} {:<10} {:<3}'.format(labels[1],cm[1,0], cm[1,1], sum(cm[1])))
    print('\n{:14} {:<10} {:<10} {:<3}'.format('All',sum(cm[:,0]), sum(cm[:,1]), sum(sum(cm))))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def print_cross_val_report(grid):
    '''
    Print nicely formatted statistics from GridSearchCV results. This includes
     the mean and std statistics for all scores used (2), on both training and
     test data (known as validation dataset).
     
    @param:
        grid: a GridSearchCV object that has been fitted and therefore is 
              available for access through cv_results_
    
    @return:
        None. Print to console. 
    '''

    r = grid.cv_results_
    i = grid.best_index_
    
    #can access values like follows:
    r['mean_test_score'][i]
    r['std_test_score'][i]
    
    raise NotImplementedError    
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
# ------------------- HYPERPARAMETERS TO TEST -------------------------
    # Change these as required. 
    NB_params = {
                'priors': range(0,5)
                } 
    
    DT_params = {
                'max_depth': range(0,5)
                } 
    
    NN_params = {
                'n_neighbours': np.linspace(1,200, 200)
                } 
    
    SVM_params ={
                'gamma': range(0,5)
                } 
# ---------------------------------------------------------------------

    # Store the CLASS LABELS.
    target_names = ("Malignant", "Benign") # corresponding to (1,0) binary values
    
    
    # Print the team.
    print(my_team())
    
    # Pre-process the dataset.
    data, labels = prepare_dataset('medical_records.data')
    
    # Split the dataset into the corresponding ratio for crossvalidation. 
    train_data, train_labels, test_data, test_labels = split_data(data, labels, 0.8)
    
    # Store a list of all parameters necesary for testing and classification
    classifier_list = [["Naive Bayes", build_NB_classifier, NB_params],
                       ["Decision Tree", build_DT_classifier, DT_params],
                       ["Nearest Neighbors", build_NN_classifier, NN_params],
                       ["Support Vector Machine", build_SVM_classifier, SVM_params]]

    # Analyze each classifier. 
    for name, function, params in classifier_list:
        
        # Create appropriate optimized classifier and report validation metrics.
        clf = function(train_data, train_labels, params)
        print_cross_val_report(clf)
        
        # Quantify the classifier's performance on the TRAINING set.
        prediction_train_labels = clf.predict(train_data)
        print_prediction_report(prediction_train_labels, train_labels)
        
        # Quantify the classifier's performance on TEST SET. 
        prediction_labels = clf.predict(test_data)
        print_prediction_report(prediction_labels, test_labels)
        

    '''
    OVERALL STRUCTURE: (this is now INCORRECT)
        
    Print the team.
    
    Prepare dataset. 
    ->function: prepare_dataset()
    
    Split data into TEST(20%) and TRAINING(80%) sets. 
    ->function: split_data()
    
    For each classifier type:
        
        Select a hyperparameter if available/necessary. (manual)
        
        For each value of selected hyperparameter:
            
            Run k-fold cross-validation.
            -> function: cross_validation() or cross_validation_hyperparameter()
            
                Divide data into k-folds 
                -> function: make_k_folds()
                
                For each unique combo of k-1 folds ("training data"):
                    
                    Run experiment. 
                    -> function: run_experiment()
                    
                        Build the classifier with k-1 folds as TRAINING
                        -> function: build_AB_classifier()
                        
                        Test classifier with kth folds as VALIDATION
                        -> function test_classifier()
                
            Average performance data for all folds/experiments.
            Store as performance data for the selected HP-VALUE. 
        
        Build classifier with entire TRAINING set and best HP-VALUE.
            
        Test the classifier with TEST set. 
        -> function test_classifier()
        
        Return the performance metrics for TEST set. 
        Return performance metrics for TRAINING, and VALIDATION data for the 
         "best" HP-VALUE.
    
    '''

