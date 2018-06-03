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
import matplotlib.pyplot as plt
from sklearn import naive_bayes, neighbors, tree, svm, model_selection, metrics
import warnings 
import time


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


def build_NB_classifier(X_training, y_training):
    '''  
    Build a Naive Bayes classifier based on the training set X_training, 
     y_training, optimized for the hyperparameters passed.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        nbc: the naive bayes classifier built in this function
        results: the dict of scores returned by cross validation, since 
            GridSearchCV would also return this but it cannot be used for 
            NB with no hyperparameter to optimize, and CV must be done before
            fitting takes place (and fitting happens here)
    '''
    
    print_clf_intro("NAIVE BAYES")
    
    # Instantiate a Multinomial NB classifier.
    nbc = naive_bayes.MultinomialNB()
    
    # Perform cross validation and store results. 
    results = model_selection.cross_validate(nbc, X_training, y_training, 
                                             return_train_score=True,
                                             scoring=['accuracy', 
                                                      'precision', 
                                                      'roc_auc', 
                                                      'recall',
                                                      'f1'])
    
    # Fit the data with X-training.
    nbc.fit(X_training, y_training)
    
    # Return the classifier object and CV results. 
    return nbc, results

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    

def build_DT_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, 
     y_training, optimized for the hyperparameters passed.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        dtc: the decision tree classifier built in this function (i.e. a 
            GridSearchCV object that is usable exactly as a clf object, but
            allows access to scores from HP optimization)
    '''
    
    # HYPERPARAMTER TO OPTIMIZE: NAME AND TEST RANGE
    params = { 'max_depth': np.linspace(1,100,100, dtype=int) } 
    print_clf_intro("DECISION TREE", params)   
    
    # Instantiate a GridSearchCV object of Decision Tree classifier type.
    # Pass an int to random_state to ensure repeatability.
    dtc = model_selection.GridSearchCV( tree.DecisionTreeClassifier(random_state=5), 
                                        params, 
                                        return_train_score=True,
    #                                    cv=4, #use default 3 or uncomment
                                        scoring=['accuracy',
                                                 'roc_auc',
                                                 'precision', 
                                                 'recall',
                                                 'f1'],
                                        refit='roc_auc'
                                        )
       
    # Fit the data, which will run k-fold cross-validation on X_training.    
    with warnings.catch_warnings():
        # Prevent warnings from printing (and ruining all my nice formatting!!)
        # Warnings tell us that some precision values are zero, but they are 
        # for parameter values that are BAD and wont be used anyways, so it
        # isnt an issue but we still need to test them in GridSearchCV.        
        warnings.simplefilter("ignore")  
        dtc.fit(X_training, y_training)
    
    # Return the GridSearchCV object, which automatically uses the best 
    # estimator for predictions, but also allows access to cv_results_.
    return dtc 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    

def build_NN_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, 
    y_training, optimized for the hyperparameters passed. 

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        knn: the k-nearest neighbors classifier built in this function (i.e. a 
            GridSearchCV object that is usable exactly as a clf object, but
            allows access to scores from HP optimization)
    '''
    
    # HYPERPARAMETER TO OPTIMIZE: NAME AND TEST RANGE
    params = { 'n_neighbors': np.linspace(1,200, 200, dtype=int) } 
    print_clf_intro("NEAREST NEIGHBOR", params)
      
    # Instantiate a GridSearchCV object of KNN classifier type.
    knn = model_selection.GridSearchCV( neighbors.KNeighborsClassifier(), 
                                        params, 
                                        return_train_score=True,
    #                                    cv=4, #use default 3 or uncomment
                                        scoring=['accuracy', 
                                                 'roc_auc',
                                                 'precision', 
                                                 'recall',
                                                 'f1'],
                                        refit='roc_auc'
                                        )      
    
    # Fit the data, which will run k-fold cross-validation on X_training.
    with warnings.catch_warnings():
        # Prevent warnings from printing (and ruining all my nice formatting!!)
        # Warnings tell us that some precision values are zero, but they are 
        # for parameter values that are BAD and wont be used anyways, so it
        # isnt an issue but we still need to test them in GridSearchCV.
        warnings.simplefilter("ignore")
        knn.fit(X_training, y_training)
    
    # Return the GridSearchCV object, which automatically uses the best 
    # estimator for predictions, but also allows access to cv_results_.
    return knn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    

def build_SVM_classifier(X_training, y_training):
    '''     
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
        X_training: X_training[i,:] is the ith example
        y_training: y_training[i] is the class label of X_training[i,:]

    @return
        svm: the svm classifier built in this function (i.e. a GridSearchCV 
            object that is usable exactly as a clf object, but allows access 
            to scores from HP optimization)
    '''
    
    # HYPERPARAMETER TO OPTIMIZE: NAME AND TEST RANGE
    params ={ 'gamma': np.logspace(-10, 1, 100) } 
    print_clf_intro("SUPPORT VECTOR MACHINE", params)
        
    # Instantiate a GridSearchCV object of SVC classifier type.
    svc = model_selection.GridSearchCV( svm.SVC(), #to allow neg_log_loss 
                                        params, 
                                        return_train_score=True,
    #                                    cv=4, #use default of 3, or uncomment
                                        scoring=['accuracy', 
                                                 'roc_auc',
                                                 'precision', 
                                                 'recall',
                                                 'f1', ],
                                        refit='roc_auc'
                                        )  
        
    # Fit the data, which will run k-fold cross-validation on X_training.
    with warnings.catch_warnings():
        # Prevent warnings from printing (and ruining all my nice formatting!!)
        # Warnings tell us that some precision values are zero, but they are 
        # for parameter values that are BAD and wont be used anyways, so it
        # isnt an issue but we still need to test them in GridSearchCV.
        warnings.simplefilter("ignore")
        svc.fit(X_training, y_training)
    
    # Return the GridSearchCV object, which automatically uses the best 
    # estimator for predictions, but also allows access to cv_results_.
    return svc

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                            ADDITIONAL FUNCTIONS 
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
    
    labels = (1,0)
    
    # Confusion matrix.
    print('\nConfusion Matrix:') 
    cm = metrics.confusion_matrix(y_true, y_pred, labels)
    assert len(names)==len(cm)
    assert cm.shape == (2,2)   
    print('{:14} {:10} {:10} {:3}'.format('PREDICTED:',names[0], names[1], 'All'))
    print("ACTUAL: ")
    print('{:14} {:3} {:3} {:1} {:2} {:3} {:5}'.format(names[0], '(TP)', cm[0,0], '','(FN)', cm[0,1], sum(cm[0])))
    print('{:14} {:3} {:3} {:1} {:2} {:3} {:5}'.format(names[1], '(FP)', cm[1,0], '','(TN)', cm[1,1], sum(cm[1])))
    print('{:14} {:8} {:10} {:5}'.format('All',sum(cm[:,0]), sum(cm[:,1]), sum(sum(cm))))
    
    # Classification report.
    print("\nClassification Report:")
    print(metrics.classification_report(y_true, y_pred, labels, target_names=names))
    
    # Miscellaneous metrics.
    print("\nOverall Metrics:")
    print('{:14} {:.2f}'.format('accuracy:', metrics.accuracy_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('roc_auc:', metrics.roc_auc_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('precision:', metrics.precision_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('recall:', metrics.recall_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('f1:', metrics.f1_score(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('lgrthmc loss:', metrics.log_loss(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('mse:', metrics.mean_squared_error(y_true, y_pred) ))
    print('{:14} {:.2f}'.format('variance:', metrics.explained_variance_score(y_true, y_pred) ))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    
def print_cv_report(r):
    '''
    Print nicely formatted statistics from GridSearchCV results. This includes
     the mean and std statistics for all scores used (2), on both training and
     test data (known as validation dataset).
     
    @param:
        results: a dict of results from running sklearn.model_selection.cross_validate,
                 scored by accuracy, roc_auc, precision and recall. 
    
    @return:
        None. Print to console. 
    '''
    
    score_grid = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    # Print title bar.
    print("\n\n- - - VALIDATION REPORT - - -") 
    
    # Print training and test ("validation") scores on all metrics. 
    print('\n{:12} {:10} {:10}'.format('', 'TRAINING', 'VALIDATION'))
    for metric in score_grid:
        print('{:12} {:8.2f} {:12.2f}'.format(metric + ':', 
                                              np.mean(r['train_%s' % metric]),
                                              np.mean(r['test_%s' % metric] )))
    
    print('\nMean fit time: {:.6f} seconds'.format(np.mean(r['fit_time'])))
    print('Mean score time: {:.6f} seconds'.format(np.mean(r['score_time'])))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def print_grid_search_report(grid):
    '''
    Print nicely formatted statistics from GridSearchCV results. This includes
     the mean and std statistics for all scores used (2), on both training and
     test data (known as validation dataset).
     
    @param:
        grid: a GridSearchCV object scored by accuracy, roc_auc, precision and 
              recall, that has been fitted and therefore is available for 
              access through cv_results_
    
    @return:
        None. Print to console. 
    '''

    r = grid.cv_results_
    i = grid.best_index_
    score_grid = clf.scoring
        
    # Print the parameter optimized and the ideal value found
    print("\n\n- - - VALIDATION REPORT - - -")
    print("Based on validation {} scores, the best value for hyperparameter '{}' is:\n{}".format(
                                        grid.refit, 
                                        list(grid.best_params_.keys())[0], 
                                        list(grid.best_params_.values())[0]) ) 
    
    # For the ideal parameter value, print train and test ("validation") scores
    print('\n{:12} {:10} {:10}'.format('', 'TRAINING', 'VALIDATION'))
    for metric in score_grid:
        print('{:12} {:8.2f} {:12.2f}'.format(metric + ':', 
                                              r['mean_train_%s' % metric][i],
                                              r['mean_test_%s' % metric][i] ))
    
    print('\nMean fit time: {:.6f} seconds'.format(r['mean_fit_time'][i]))
    print('Mean score time: {:.6f} seconds'.format(r['mean_score_time'][i]))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_grid_search_results(clf): 
    # Organize data and labels
    param_name = list(clf.param_grid.keys())[0]
    param_vals = list(clf.param_grid.values())[0]
    metrics = clf.scoring    
    score_grid = []
    for name in metrics:
        score_grid.append(clf.cv_results_['mean_test_%s' % name])

    # Plot the organized data and labels
    p = plt
    idx=0
    for scores in score_grid:
        p.plot(param_vals, scores, '-', label=metrics[idx])
        idx+=1
        
    # Configure plot and show
    p.title("Hyperparameter Optimization by Cross-Validation")
    p.xlabel(param_name + " value")
    if param_name =='gamma': p.xscale('log')
    p.ylabel('average test score')
    p.legend(loc="lower right")
    p.grid(True)
    p.show()

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
        print("Values Tested: {} values from {} to {}".format( 
                                            len( list(params.values())[0] ), 
                                            min( list(params.values())[0] ), 
                                            max( list(params.values())[0] ) ) )
    print("\nWorking...")
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        
        
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                           MAIN FUNCTION
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
if __name__ == "__main__":

# --------------------- Data-specific variables ----------------------
    # Change these as required. 
    class_labels = ("Malignant", "Benign") #corresponding to (1,0) binary vals
    path_to_data = 'medical_records.data'
    test_set_ratio = 0.2
    
    # Store a list of all parameters necesary for testing and classification
    function_list = [
                       build_NB_classifier,
                       build_DT_classifier,
                       build_NN_classifier,
                       build_SVM_classifier
                       ]


# ------------------------ Experiment process ------------------------

    # Print the team.
    print_introduction(my_team())
    
    # Pre-process the dataset.
    data, labels = prepare_dataset(path_to_data)
    
    # Split the dataset into the corresponding ratio for crossvalidation. 
    # Set random_state to a hard-coded number to ensure repeatability.
    train_data,test_data,train_labels,test_labels = model_selection.train_test_split(
            data, labels, test_size=test_set_ratio, random_state=1)  
    
    # Print split information.
    print('\n\nTraining set: {:.1f}% of data, {} samples, {} positive ({:.1f}%)'.format(
        (1-test_set_ratio)*100, len(train_data), sum(train_labels), (sum(train_labels)*100)/len(train_data)))
    print('Test set: {:.1f}% of data, {} samples, {} positive ({:.1f}%)'.format(
        test_set_ratio*100, len(test_data), sum(test_labels), (sum(test_labels)*100)/len(test_data)))

    # Analyze and use each classifier, show results.  
    for function in function_list:
        
        if function is build_NB_classifier: 
            # Indicator for NBC. 
            # Handle this differently, since no optimization is necessary.
            t0 = time.time()
            clf, cv_results = function(train_data, train_labels)
            print_cv_report(cv_results)
        
        else:
            # Create appropriate optimized classifier and report VALIDATION metrics.
            t0 = time.time()
            clf = function(train_data, train_labels)
            print_grid_search_report(clf)
            plot_grid_search_results(clf)
            
        t1 = time.time()   
        print("\nCross-validation, optimization, and fitting took {:.6f} seconds total.".format(t1-t0))

        # Quantify the classifier's performance on the TRAINING set.
        pred_train_labels = clf.predict(train_data)
        t2 = time.time()
        print("\n\n- - - TRAINING REPORT - - -")
        print_prediction_report(pred_train_labels, train_labels, class_labels)
        print("\nPrediction on training set took {:.6f} seconds.".format(t2-t1))
        
        # Quantify the classifier's performance on TEST SET. 
        t3 = time.time()
        pred_labels = clf.predict(test_data)
        t4 = time.time()
        print("\n\n- - - TEST REPORT - - -")
        print_prediction_report(pred_labels, test_labels, class_labels)
        print("\nPrediction on test set took {:.6f} seconds.".format(t4-t3))
