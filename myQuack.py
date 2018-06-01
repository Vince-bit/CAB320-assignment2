'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
Write a main function that calls the different functions to perform the required tasks 
and repeat your experiments.

'''
'''
    Possible metrics: 
        - neg_log_loss: returns neegative "Logarithmic Loss", like accuracy
          but taking into account the uncertainty of the answer (?)
          
        - accuracy: proportion of predictions that exactly match those
          corresponding in the list of true labels
          
        - precision: # correct +ve results, divided by # ALL +ve results returned
            tp / (tp + fp)
            
        - recall: # correct +ve results, divided by # all TRUE +ve results
            tp / (tp + fn)
            
        - f1_score: harmonic average between precision and recall 
            best values = 1, worst values = 0
            
        - support: number of occurences of a given class in list of true labels
        
        - roc_auc: "Area Under Curve"
            best values = 1, value for random assignment = 0.5 
            
'''


import numpy as np
from sklearn import naive_bayes, neighbors, tree, svm, model_selection, metrics
import warnings 



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                              REQUIRED FUNCTIONS
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
    Build a Naive Bayes classifier based on the training set X_training, 
     y_training, optimized for the hyperparameters passed.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
        params: a dict of form:
                     {'parameter name (str)': [parameter range (np.array)]},
                with only one parameter for the purpose of this assignment. 

    @return
        nbc: the naive bayes classifier built in this function
    '''
    print("\nBuilding and optimizing...")

    # Instantiate a GridSearchCV object of NB classifier type.
    nbc = model_selection.GridSearchCV( naive_bayes.GaussianNB(), 
                                        return_train_score=True,
                                        cv=4,
                                        scoring=['accuracy', 'precision', 'roc_auc', 'recall'],
                                        refit='accuracy'
                                        )  
    
    # Fit the data, which will run k-fold cross-validation on X_training. 
    with warnings.catch_warnings(): 
        # Prevent warnings from printing (and ruining all my nice formatting!!)
        warnings.simplefilter("ignore")
        nbc.fit(X_training, y_training)
    
    # Return the GridSearchCV object, which automatically uses the best 
    # estimator for predictions, but also allows access to cv_results_.
    return nbc

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training, params):
    '''  
    Build a Decision Tree classifier based on the training set X_training, 
     y_training, optimized for the hyperparameters passed.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
        params: a dict of form:
                     {'parameter name (str)': [parameter range (np.array)]},
                with only one parameter for the purpose of this assignment. 

    @return
        dtc: the decision tree classifier built in this function
    '''
    print("\nBuilding and optimizing...")
        
    dtc = model_selection.GridSearchCV( tree.DecisionTreeClassifier(), 
                                        params, 
                                        return_train_score=True,
                                        cv=4,
                                        scoring=['accuracy', 'precision', 'roc_auc', 'recall'],
                                        refit='accuracy'
                                        )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  
        dtc.fit(X_training, y_training)
    
    return dtc 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training, params):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, 
    y_training, optimized for the hyperparameters passed. 

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
        params: a dict of form:
                     {'parameter name (str)': [parameter range (np.array)]},
                with only one parameter for the purpose of this assignment. 

    @return
        knn: the k-nearest neighbors classifier built in this function
    '''
    print("\nBuilding and optimizing...")
        
    knn = model_selection.GridSearchCV( neighbors.KNeighborsClassifier(), 
                                        params, 
                                        return_train_score=True,
                                        cv=4,
                                        scoring=['accuracy', 'precision', 'recall', 'roc_auc'],
                                        refit='accuracy'
                                        )  
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        knn.fit(X_training, y_training)
    
    return knn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training, params):
    '''     
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]
        params: a dict of form:
                     {'parameter name (str)': [parameter range (np.array)]},
                with only one parameter for the purpose of this assignment. 

    @return
        svm: the svm classifier built in this function
    '''
    print("\nBuilding and optimizing...")
        
    svc = model_selection.GridSearchCV( svm.LinearSVC(), 
                                        params, 
                                        return_train_score=True,
                                        cv=4,
                                        scoring=['accuracy', 'precision', 'roc_auc', 'recall'],
                                        refit='accuracy'
                                        )  
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svc.fit(X_training, y_training)
    
    return svc

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                            ADDITIONAL FUNCTIONS 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def split_data(data, label, ratio):
    '''
    Split the given data (and corresponding labels) into training and testing 
     datasets based on the ratio given. ** Repeatable and non-random. **
        
    @param
        data: 2D np.ndarray, X[i,:] is the ith example
        labels: 1D np.ndarray, y[i] is the class label of X[i,:]
        ratio: decimal ratio of training:original data (i.e. 0.8 represents 80% 
               training and 20% testing from 100% original data)
    
    @return:
        data_train:  2D array of examples for training
        label_train: 1D array of corresponding labels for training
        data_test: 2D array of examples for testing
        label_test: 1D array of corresponding labels for training
    '''
    #defense 
    assert(ratio>0 and ratio<1)
    assert(data.ndim==2 and label.ndim==1)
    
    # Create small and large arrays of data and labels according to ratio.
    data_train = data.copy()[:int(len(data)*ratio)]
    label_train = label.copy()[:int(len(data)*ratio)]
    data_test = data.copy()[int(len(data)*ratio):]
    label_test = label.copy()[int(len(data)*ratio):]
    
    return data_train, label_train, data_test, label_test
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def print_prediction_report(y_pred, y_true, names):
    '''
    Return a bunch of statistics and metrics reporting the performance of a 
     certain classifier model on the given training data. 
     
    @param:
        y_true: A np-array of the target class labels as integers
        y_pred: A np-array of the classifier-predicted class labels as integers
        names: A tuple of the class labels (str), corresponding to (1,0) 
               binary integer class labels
               
    @return:
        None. Print to console. 
    '''
    labels = [1,0]
    #classification report
    cr = metrics.classification_report(y_true, y_pred, labels, target_names=names)
    #confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, labels = (1,0))
    display_confusion_matrix(cm, names)

    #raise NotImplementedError
    pass

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

    print('\nConfusion Matrix:')    
    print('{:14} {:10} {:10} {:3}'.format('PREDICTED:',labels[0], labels[1], 'All'))
    print("ACTUAL: ")
    print('{:14} {:<3} {:<6} {:<3} {:<6} {:<3}'.format(labels[0],cm[0,0], '(TP)', cm[0,1], '(FN)', sum(cm[0])))
    print('{:14} {:<3} {:<6} {:<3} {:<6} {:<3}'.format(labels[1],cm[1,0], '(FP)', cm[1,1], '(TN)', sum(cm[1])))
    print('{:14} {:<10} {:<10} {:<3}'.format('All',sum(cm[:,0]), sum(cm[:,1]), sum(sum(cm))))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def print_grid_search_report(grid):
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
        
    # Print the parameter optimized and the ideal value found
    print("\n\n- - - CROSS-VALIDATION REPORT - - -")
    print("Based on {}, the best value for hyperparameter '{}' is:\n{}".format(
                                        list(grid.scorer_.keys())[0], 
                                        list(grid.best_params_.keys())[0], 
                                        list(grid.best_params_.values())[0]) )
    
    # For the ideal parameter value, print train and test ("validation") scores
    print('\n{:17} {:18} {:15}'.format('', 'TRAINING', 'VALIDATION'))
    
    print('{:17} {:<5} {} {:<8} {:<5} {} {:<8}'.format('Accuracy:', 
                                       round(r['mean_train_accuracy'][i], 3),
                                       '+/-',
                                       round(r['std_train_accuracy'][i], 3),
                                       round(r['mean_test_accuracy'][i], 3),
                                       '+/-',
                                       round(r['std_test_accuracy'][i], 3)))
    
    print('{:17} {:<5} {} {:<8} {:<5} {} {:<8}'.format('Area Under ROCC:', 
                                       round(r['mean_train_roc_auc'][i], 3),
                                       '+/-',
                                       round(r['std_train_roc_auc'][i], 3),
                                       round(r['mean_test_roc_auc'][i], 3),
                                       '+/-',
                                       round(r['std_test_roc_auc'][i], 3)))
    
    print('{:17} {:<5} {} {:<8} {:<5} {} {:<8}'.format('Precision:', 
                                       round(r['mean_train_precision'][i], 3),
                                       '+/-',
                                       round(r['std_train_precision'][i], 3),
                                       round(r['mean_test_precision'][i], 3),
                                       '+/-',
                                       round(r['std_test_precision'][i], 3)))
    
    print('{:17} {:<5} {} {:<8} {:<5} {} {:<8}'.format('Recall:', 
                                       round(r['mean_train_recall'][i], 3),
                                       '+/-',
                                       round(r['std_train_recall'][i], 3),
                                       round(r['mean_test_recall'][i], 3),
                                       '+/-',
                                       round(r['std_test_recall'][i], 3)))
    
    print('\nMean fit time: {:.8f}'.format(r['mean_fit_time'][i]))
    print('Mean score time: {:.8f}'.format(r['mean_score_time'][i]))

    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def print_introduction(team_array):
    '''
    Print a nice introduction to the code being run, including well-formatted
     team member list.
     
    @param:
        team_array: array of student numbers and names returned form my_team()
        
    @return:
        None. Print to console. 
    '''
    
    print("\n\n***************************************************************")
    print("            CAB320 ASSIGNMENT 2: MACHINE LEARNING              ")
    print("***************************************************************")
    print("\nTEAM MEMBERS:")

    for person in team_array:
        print('{:4} {:4} {:10} {:10}'.format(person[0],':',person[1], person[2]))
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def print_clf_intro(name, params=None):
    '''
    Print a nice introduction to the classifier being tested and used.
     
    @param:
        name: string holding the name of the classifier type
        params: dict holding the name of the hyperparameter to be optimized
        
    @return:
        None. Print to console. 
    '''
    
    print("\n\n\n\n***************************************************************")
    print("* {} CLASSIFIER".format(name))
    if(params is not None):
        print("\nHyperparameter: " + list(params.keys())[0]) # JUST KEY
        print("Values Tested: {} values from {} to {}".format( len( list(params.values())[0] ), 
                                                               min( list(params.values())[0] ), 
                                                               max( list(params.values())[0] ) ) )
    
        

    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                           MAIN FUNCTION
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    
# ------------------- HYPERPARAMETERS TO TEST -------------------------
    # Change these as required. 
    NB_params = None #architecture selection is trivial
    DT_params = {
                'max_depth': np.linspace(1,100,100) #temp
                } 
    
    NN_params = {
                'n_neighbors': np.linspace(1,200, 200, dtype=int)
                } 
    
    SVM_params ={
                'C': np.linspace(0.5,5,91) #temp
                } 
# ---------------------------------------------------------------------


# ------------------- DATA-SPECIFIC VARIABLES -------------------------
    # Change these as required. 
    class_labels = ("Malignant", "Benign") #corresponding to (1,0) binary vals
    path_to_data = 'medical_records.data'
# ---------------------------------------------------------------------



# ----------------- TESTING AND LEARNING BEGINS -----------------------

    # Print the team.
    print_introduction(my_team())
    
    # Pre-process the dataset.
    data, labels = prepare_dataset(path_to_data)
    
    # Split the dataset into the corresponding ratio for crossvalidation. 
    train_data, train_labels, test_data, test_labels = split_data(data,labels,0.8)
    
    # Store a list of all parameters necesary for testing and classification
    classifier_list = [
#                       ["NAIVE BAYES", build_NB_classifier, DT_params],
                       ["DECISION TREE", build_DT_classifier, DT_params],
                       ["NEAREST NEIGHBOURS", build_NN_classifier, NN_params],
                       ["SUPPORT VECTOR MACHINE", build_SVM_classifier, SVM_params]]

    # Analyze each classifier. 
    for name, function, params in classifier_list:
        
        print_clf_intro(name, params)
        
        # Create appropriate optimized classifier and report VALIDATION metrics.
        clf = function(train_data, train_labels, params)
        print_grid_search_report(clf)
        
        # Quantify the classifier's performance on the TRAINING set.
        prediction_train_labels = clf.predict(train_data)
        print("\n\n- - - TRAINING REPORT - - -")
        print_prediction_report(prediction_train_labels, train_labels, class_labels)
        
        # Quantify the classifier's performance on TEST SET. 
        prediction_labels = clf.predict(test_data)
        print("\n\n- - - TEST REPORT - - -")
        print_prediction_report(prediction_labels, test_labels, class_labels)
        








    '''
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

