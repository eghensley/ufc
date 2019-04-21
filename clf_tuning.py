#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: eric.hensley@ibm.com
"""

import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'irma':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))
sys.path.insert(3, cur_path)
sys.path.insert(4, os.path.join(cur_path, 'modelling'))

import _config
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.externals import joblib
from sklearn.metrics import log_loss
import random
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from model_feats import features


# Declare folder locations
global data_folder
global tuning_model_folder
global final_model_folder
global tuning_result_folder
global final_result_folder


def ensure_dir(file_path):
    """ Create directory if doesn't exist """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def _weighted_log_loss(y_true, y_pred):
    """
    Calculate custom weighted log-loss score 
    
    Parameters
    ----------
    y_true : pandas Series
        Actual y values
    y_pred : pandas Series
        Predicted y values
        
    Returns
    -------
    numpy array
        Log-loss score weighted to balance classes
        
    """ 
    
    
    # Weights for each class
    test_weights = class_weight.compute_class_weight('balanced',
                                np.unique(y), y)
    # Weights for each sample
    test_weights_dict = {i:j for i,j in zip(np.unique(y), test_weights)}
    # Weighted score
    score = log_loss(y_true, y_pred, sample_weight = [test_weights_dict[i] for i in y_true])
    score *= -1
    return(score)
    
    
def cross_validate(model, folds = 16, njobs = -1, feats = False, verbose = False): 
    """
    Dispatch model testing using cross-validation 
    
    Parameters
    ----------
    model : sklearn classifier
        Classifier model to test
    folds : int
        Number of cross-validation folds
    njobs: int
        Number of parallel jobs to dispatch to available cores
    feats: Boolean
        Flags to test feature selector (Depreciated)
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    float
        Cross-validation score
        
    """ 
    
    
    # Deepcopy model to change parameters
    estimator = deepcopy(model)
    
    # MacOS parallelizes LightGBM tree creation across all available cores, so pass jobs
    # sequentially like for single core job dispatch for sklearn.  Otherwise, the processes
    # overload and take longer to run.
    if feats:
        if sys.platform != 'darwin' and lgb.LGBMClassifier == estimator.steps[-1][1].estimator.__class__:
            njobs = 1     
    else:
        if sys.platform != 'darwin' and lgb.LGBMClassifier in [i[1].__class__ for i in estimator.steps]:
            njobs = 1
            
    # Split data in repeatable manner using random seed from irma._config.py
    splitter = StratifiedKFold(n_splits = folds, random_state = _config.random_seed)
        
    # Break data into CV folds
    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold)
    
    # IF solving using single core (or LightGBM on MacOS), perform cross-validation sequentially
    if njobs == 1:
        cv_scores = []    
        for i, fold in enumerate(all_folds):
            if verbose:
                print('Scoring fold-%s' % (i))
            cv_scores.append(_single_core_solver((estimator, fold, feats)))
    # Pass all jobs through sklearn joblib library to dispatch to all available cores
    else:
        all_jobs = []
        for fold in all_folds:
            all_jobs.append((estimator, fold, feats))
        if verbose:
            cv_scores = joblib.Parallel(n_jobs = njobs, verbose = 25)(joblib.delayed(_single_core_solver) (i) for i in all_jobs)
        else:
            cv_scores = joblib.Parallel(n_jobs = njobs)(joblib.delayed(_single_core_solver) (i) for i in all_jobs)
    
    cv_scores = np.mean(cv_scores)
    return(cv_scores)
    
    
def _single_core_solver(input_vals):
    """
    Dispatch single solving job to an available core
    
    Parameters
    ----------
    input_vals : list
        Job values
        
    Returns
    -------
    float
        Cross-validation score
        
    """ 
    
    
    # Unpack CV parameters from inputs
    est, fold, features = input_vals
    
    # Identify train and test indicies
    train_idx = [j for i,j in enumerate(x.index) if i in fold[0]]
    test_idx = [j for i,j in enumerate(x.index) if i in fold[1]]
    
    # Test effect of feature selection step (depreciated)
    if features:
        score = est.fit_transform(x.loc[train_idx], y.loc[train_idx]).shape[1]
    # Calculate model performance for cross validation fold
    else:
        # Fit model using K-1 folds
        est.fit(x.loc[train_idx], y.loc[train_idx])
        # Predict label of excluded fold
        pred = est.predict_proba(x.loc[test_idx])
        # Weights for each class
        test_weights = class_weight.compute_class_weight('balanced',
                                        np.unique(y.loc[train_idx]),y.loc[train_idx])
        # Weights for each sample
        test_weights_dict = {i:j for i,j in zip(np.unique(y.loc[train_idx]), test_weights)}
        # Weighted logloss score
        score = log_loss(y.loc[test_idx], pred, sample_weight = [test_weights_dict[i] for i in y.loc[test_idx]])
        score *= -1
    return(score)
       
    
def random_search(model, params, trials = 25, folds = 16, verbose = False):  
    """
    Perform random hyperparameter search
    
    Parameters
    ----------
    model : sklearn classifier
        Classifier model to test
    params: dict
        Model parameters
    trials: int
        Number of iterations of random parameter searches to perform
    folds : int
        Number of cross-validation folds
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    dict
        Best parameters and corresponding scores
        
    """ 

    
    def _rand_shuffle(choices):
        """ Randomly select a hyperparameter for search """

        selection = {}
        for key, value in choices.items():
            selection[key] = random.choice(value)
        return(selection)
        
    
    # Initiate data storage dictionary
    search_scores = {}
    
    # Identify current classifier parameters
    current_params = deepcopy(model).steps[-1][1].get_params()
    # Calculate maximum possible parameter combinations
    max_combinations = np.prod([len(i) for i in params.values()])
    
    # Limit trials to maximum combinations if higher
    if trials > max_combinations:
        trials = max_combinations
        
    for i in range(trials):
        # Generate random hyper-parameter combination
        iter_params = _rand_shuffle(params)
        iter_name = '_'.join("{!s}={!r}".format(k,v) for (k,v) in iter_params.items())
        # Generate a new random hyper-parameter combination if previous combination
        # has already been tested
        while iter_name in search_scores.keys():
            iter_params = _rand_shuffle(params)
            iter_name = '_'.join("{!s}={!r}".format(k,v) for (k,v) in iter_params.items()) 
        
        # Set model parameters to selected combination
        search_scores[iter_name] = {}
        for key, value in iter_params.items():
            estimator = deepcopy(model)
            estimator.set_params(**{key: value})
            search_scores[iter_name][key] = value
            if verbose:
                try:
                    print('%s: %.3f' % (key, value))
                except TypeError:
                    print('%s: %s' % (key, value))
                    
        # Skip testing if new set of hyper-parameters is the same as the 
        # current parameter combination 
        if estimator.steps[-1][1].get_params() == current_params:
            if verbose:
                print('Parameters already in default.')
            iter_score = -np.inf
        # Calculate model performance with new hyper-parameters using 
        # cross-validation
        else:
            try:
                iter_score = cross_validate(estimator)
            except:
                iter_score = -np.inf
        if verbose:
            print('- score: %.5f' % (iter_score))
        # Add scores to storage dictionary
        search_scores[iter_name]['score'] = iter_score
        
    # Identify and return best hyper-parameter combination
    best_combination = max(search_scores, key =lambda x: search_scores[x]['score'])
    return_results = {'score': search_scores[best_combination]['score']}
    return_results['parameters'] = {}
    best_params = dict(search_scores[best_combination])
    best_params.pop('score')
    return_results['parameters'] = best_params
    return(return_results)
    
    
def test_scaler(clf, verbose = True, prev_score = False, prev_scaler = False, skip = False):  
    """
    Perform test of different scalers on model performance
    
    Parameters
    ----------
    clf : sklearn classifier
        Classifier model to test
    verbose: Boolean
        Flag to print output
    prev_score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    prev_scaler : Boolean/sklearn scaler
        Flag if scaler has already been test, default False but pass current scaler 
    skip: Boolean
        Flag to skip Robust Scaler (takes forever with SVC)
        
    Returns
    -------
    sklearn scaler, float
        Best scaler and corresponding score
        
    """ 
    
    
    print('Searching for best scaler.')
    # Initiate data storage dictionary
    scores = {}
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    
    # Test performance of prepending scalers to sklearn pipeline
    for scale, name in zip([StandardScaler(), MinMaxScaler(), RobustScaler()], ['Standard', 'MinMax', 'Robust']):
        if skip and name == 'Robust':
            continue
        # Skip testing of scaler already included in current pipeline
        if prev_scaler and prev_score and prev_scaler.__class__ == scale.__class__:
            if verbose:
                print('%s Already Included' % (name))
                print('Score: %.5f' % (prev_score))
            continue
        # Calculate cross-validated pipeline performance using scaler 
        score = cross_validate(model.set_params(**{'scale': scale}))
        if verbose:
            print('%s Score: %.5f' % (name, np.mean(score)))
        # Add scores to storage dictionary
        scores[name] = {'scale': scale}
        scores[name]['score'] = score
        
    # Identify and return best scaler       
    best_scale = max(scores, key = lambda x: scores[x]['score'])
    if prev_score:
        # If original scaler performs worse than new, use new
        if scores[best_scale]['score'] > prev_score:
            print('Using %s.' % (best_scale))
            return(scores[best_scale]['scale'], scores[best_scale]['score'])
        # If new scaler performs worse than previous, keep original
        else:
            print('Keeping original.')
            return(prev_scaler, prev_score)
    else:
        print('Using %s.' % (best_scale))        
        return(scores[best_scale]['scale'], scores[best_scale]['score'])
        

def test_solver(clf, prev_score, verbose = False):
    """
    Perform test of different solvers on model performance
    
    Parameters
    ----------
    clf : sklearn classifier
        Classifier model to test
    prev_score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    str, float
        Best solver algorithm and corresponding score
        
    """ 
    
    
    print('Searching for best solver.')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    comp_model = deepcopy(model.steps[-1][1])
    # Initiate data storage dictionary        
    scores = {}

    # Test performance of using different solver algorithms    
    for solve in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
        model.set_params(**{'clf__solver': solve})
        # Skip testing of solver already tested
        if comp_model.get_params() == model.steps[-1][1].get_params():
            if verbose:
                print('Sovler Already Included... skipping')
            continue
        # Calculate cross-validated model performance using algorithm
        score = cross_validate(model)
        if verbose:
            print('%s Score: %.5f' % (solve, np.mean(score)))
        # Add scores to storage dictionary
        scores[solve] = score
        
    # Identify and return best scaler               
    best_solver = max(scores, key = lambda x: scores[x])
    # If original scaler performs worse than new, use new
    if scores[best_solver] > prev_score:
        print('Using %s.' % (best_solver))
        return(model.set_params(**{'clf__solver': best_solver}), scores[best_solver]) 
    # If new scaler performs worse than previous, keep original
    else:
        print('No Improvement, Using Default')
        return(clf, prev_score) 
 
    
def test_degree(clf, prev_score, verbose = False):
    """
    Perform test of different degrees on model performance
    
    Parameters
    ----------
    clf : sklearn classifier
        Classifier model to test
    prev_score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    int, float
        Best degree and corresponding score

    """ 

        
    print('Searching for best degree.')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    comp_model = deepcopy(model.steps[-1][1])
    # Initiate data storage dictionary
    scores = {}
    for deg in [2,3,4]:
        model.set_params(**{'clf__degree': deg})
        # Skip testing of degree already tested
        if comp_model.get_params() == model.steps[-1][1].get_params():
            if verbose:
                print('Degree Already Included... skipping')
            continue
        # Calculate cross-validated model performance using degree
        score = cross_validate(model)
        if verbose:
            print('%s Score: %.5f' % (deg, np.mean(score)))
        # Add scores to storage dictionary
        scores[deg] = score
        
    # Identify and return best degree                       
    best_solver = max(scores, key = lambda x: scores[x])
    # If original degree performs worse than new, use new
    if scores[best_solver] > prev_score:
        print('Using %s.' % (best_solver))
        return(model.set_params(**{'clf__degree': best_solver}), scores[best_solver]) 
    # If new degree performs worse than previous, keep original
    else:
        print('No Improvement, Using Default')
        return(clf, prev_score) 
               
    
def lgb_find_lr(_model, score, lr_ = .1, step = 'clf', verbose = True):
    """
    Iterively find the best learning rate for the gradient boosting model.
    
    Parameters
    ----------
    _model : LightGBM classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    lr_: float
        Current learning rate
    step: str
        Step in the sklearn pipeline of the classifier being tuned
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    LightGBM classifier, float
        Model with best learning rate parameter and corresponding score

    """ 
    
    
    def check_lr(_clf, _lr, target, _verbose = False):
        """
        Test best number of trees for the given learning rate
        
        Parameters
        ----------
        _clf : LightGBM classifier
            Classifier model to test
        _lr: float
            Learning rate to test
        target: str
            Step in the sklearn pipeline of the classifier being tuned
        verbose: Boolean
            Flag to print output
            
        Returns
        -------
        dict
            Scores for each number of trees at the learning rate
            
        """ 
    
    
        # Deepcopy model to change parameters
        clf = deepcopy(_clf)
        if _verbose:
            print('Learning Rate: %.5f' % (_lr))
        # Initiate data storage dictionary
        scores = {}
        
        # Calculate score using 100 trees at current learning rate
        _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): 100}))
        scores[100] = np.mean(_score)
    
        # Iteritively decrease trees until improvement ends
        for tree in [90, 75, 60, 50]:
            _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): tree}))
            if len(list(scores.values())) > 0 and np.mean(_score) < list(scores.values())[-1]:
                break
            if _verbose:
                print('Trees: %s' % (tree))
                print('  Score: %.5f' % (np.mean(_score)))
            scores[tree] = np.mean(_score)
        # Iteritively increase trees until improvement ends
        for tree in [110, 125, 135, 150]:
            _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): tree}))
            if len(list(scores.values())) > 0 and np.mean(_score) < list(scores.values())[-1]:
                break
            if _verbose:
                print('Trees: %s' % (tree))
                print('  Score: %.5f' % (np.mean(_score)))
            scores[tree] = np.mean(_score)
        return (scores)

    print('Searching for best learning rate')
    # Deepcopy model to change parameters
    model = deepcopy(_model)
    
    # Initiate set with current learning rate, new score and previous score
    # throughout the optimization process
    checkpoint = (lr_, -np.inf, score) 
    
    # Optimize learning rate to find the optimal learning rate for 100 trees.
    # There is an inverse relationship between number of trees and learning rate,
    # so increase learning rate if optimal trees is less than 100 and vice versa
    # until best trees converges at 100.
    # 
    # New learning rate = learning rate * (best trees / 100)
    # eg. if 150 trees is optimal for current learning rate, multiply by 1.5
    #
    # To avoid getting stuck in a loop of the optimal learning rate falling between
    # two values, once improvement ends decrease sensitivity by 50% until improvement
    while lr_:
        # Calculate scores across different number of trees for learning rate
        score = check_lr(model, lr_, step, _verbose = verbose)
        # Identify best number of trees
        best_trees, best_score = [[i,j] for i,j in score.items() if j == max(score.values())][0]
        # Identify score for learning rate using 100 trees
        model_score = score[100]
        
        if verbose:
            print('Prev Score: %.5f' % (checkpoint[2]))
            print('New Score: %.5f' % (model_score))

        # IF new learning rate results in worse performance, reduce sensitivity by 50%
        if checkpoint[2] > model_score:
            if verbose:
                print('---- Neg Progress ----')
            lr_ = checkpoint[0] + ((lr_ - checkpoint[0]) * .5)
            continue
        else:
            checkpoint = (lr_, best_score, model_score)   
            
        # IF 100 trees is optimal at current learning rate,
        # set best learning rate and return 
        if best_trees == 100:
            print('---- Best Learning Rate')
            print('Learning Rate: %.5f' % (lr_))
            print('Score: %.5f' % (best_score))      
            return(model.set_params(**{'%s__learning_rate' % (step):lr_, '%s__n_estimators' % (step): 100}), best_score)
        # Adjust learning rate based on number of trees
        else:
            lr_ *= best_trees/100


def lgb_dart_find_lr(_model, score, lr_ = .1, step = 'clf', verbose = True):
    """
    Iterively find the best learning rate for the gradient boosting model using 
    DART algorithm.
    Find the optimal learning rate for 100 trees.

    Parameters
    ----------
    _model : LightGBM classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    lr_: float
        Current learning rate
    step: str
        Step in the sklearn pipeline of the classifier being tuned
    verbose: Boolean
        Flag to print output
        
    Returns
    -------
    LightGBM classifier, float
        Model with best learning rate parameter and corresponding score

    """ 
    
    
    def check_lr(_clf, _lr, target, _verbose = False):
        """
        Test best number of trees for the given learning rate
        
        Parameters
        ----------
        _clf : LightGBM classifier
            Classifier model to test
        _lr: float
            Learning rate to test
        target: str
            Step in the sklearn pipeline of the classifier being tuned
        verbose: Boolean
            Flag to print output
            
        Returns
        -------
        dict
            Scores for each number of trees at the learning rate
            
        """ 
        
        
        # Deepcopy model to change parameters
        clf = deepcopy(_clf)
        if _verbose:
            print('Learning Rate: %.5f' % (_lr))
        # Initiate data storage dictionary
        scores = {}

        # Calculate score using 100 trees at current learning rate        
        _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): 100}))
        scores[100] = np.mean(_score)

        # Iteritively decrease trees until improvement ends    
        for tree in [91, 80, 67, 50]:
            _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): tree}))
            if len(list(scores.values())) > 0 and np.mean(_score) < list(scores.values())[-1]:
                break
            if _verbose:
                print('Trees: %s' % (tree))
                print('  Score: %.5f' % (np.mean(_score)))
            scores[tree] = np.mean(_score)
        # Iteritively increase trees until improvement ends
        for tree in [110, 125, 150, 200]:
            _score = cross_validate(clf.set_params(**{'%s__learning_rate' % (target):_lr, '%s__n_estimators' % (target): tree}))
            if len(list(scores.values())) > 0 and np.mean(_score) < list(scores.values())[-1]:
                break
            if _verbose:
                print('Trees: %s' % (tree))
                print('  Score: %.5f' % (np.mean(_score)))
            scores[tree] = np.mean(_score)
        return (scores)
    
    
    print('Searching for best learning rate')
    # Deepcopy model to change parameters
    model = deepcopy(_model)
    # Initiate set with current learning rate, new score and previous score
    # throughout the optimization process
    checkpoint = (lr_, -np.inf, score)  
    
    # Optimize learning rate to find the optimal learning rate for 100 trees.
    # There is an inverse relationship between number of trees and learning rate,
    # so increase learning rate if optimal trees is less than 100 and vice versa
    # until best trees converges at 100.
    # 
    # New learning rate = learning rate * (best trees / 100)
    # eg. if 150 trees is optimal for current learning rate, multiply by 1.5
    #
    # To avoid getting stuck in a loop of the optimal learning rate falling between
    # two values, once improvement ends decrease sensitivity by 50% until improvement
    while lr_:
        # Calculate scores across different number of trees for learning rate
        score = check_lr(model, lr_, step, _verbose = verbose)
        # Identify best number of trees
        best_trees, best_score = [[i,j] for i,j in score.items() if j == max(score.values())][0]
        # Identify score for learning rate using 100 trees
        model_score = score[100]
        
        if verbose:
            print('Prev Score: %.5f' % (checkpoint[2]))
            print('New Score: %.5f' % (model_score))

        # IF new learning rate results in worse performance, reduce sensitivity by 50%
        if checkpoint[2] > model_score:
            if verbose:
                print('---- Neg Progress ----')
            lr_ = checkpoint[0] + ((lr_ - checkpoint[0]) * .5)
            continue
        else:
            checkpoint = (lr_, best_score, model_score)   

        # IF 100 trees is optimal at current learning rate,
        # set best learning rate and return     
        if best_trees == 100:
            print('---- Best Learning Rate')
            print('Learning Rate: %.5f' % (lr_))
            print('Score: %.5f' % (best_score))      
            return(model.set_params(**{'%s__learning_rate' % (step):lr_, '%s__n_estimators' % (step): 100}), best_score)
        # Adjust learning rate based on number of trees
        else:
            lr_ *= best_trees/100
    
    
def lgb_tree_params(use_model, score, folds = 16, iter_ = 300, step = 'clf', comp_score = None, clf = None):
    """
    Iterively find the best hyper-parameters for the gradient boosting model
    
    Parameters
    ----------
    use_model : LightGBM classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    folds: int
        Number of cross-validation folds to test on
    iter_: int
        Number of iterations of random parameter searches to perform
    step: str
        Step in the sklearn pipeline of the classifier being tuned
    comp_score: float default None
        Score of previous test         
    clf: LightGBM classifier
        Classifier to tune (depreciated)
        
    Returns
    -------
    LightGBM classifier, float
        Model with best hyper-parameters and corresponding score

    """ 
    

    print('Setting tree parameters')
    # Deepcopy model to change parameters    
    model = deepcopy(use_model)
    # Initiate grid of possible parameters and values to search
    param_dist = {'%s__min_child_samples' % (step): [int(i) for i in range(2, 50)],
                  '%s__num_leaves' % (step): [int(i) for i in range(10, 300)],
                  '%s__subsample' % (step): [i/1000 for i in range(600, 1000)],
                  '%s__max_bin' % (step): [int(i) for i in range(100, 500)]}
    
    # Perform random hyper-parameter search to find best combination
    if clf is None:
        results = random_search(model, param_dist, trials = iter_, folds = folds)
    else:
        results = random_search(deepcopy(clf), param_dist, trials = iter_, folds = folds)  
        
    print('Iteration LogLoss: %.5f' % (results['score']))
    # IF a comparison score is provided, return Boolean of if improvement occured,
    # model, and score
    if comp_score is None:
        if results['score'] < score:
            return(True, use_model, score)
        else:
            return(False, model.set_params(**results['parameters']), results['score'])
    # Return model and score
    else:
        if results['score'] < comp_score:
            return(clf, score)
        else:
            return(clf.set_params(**results['parameters']), results['score'])


def knn_hyper_parameter_tuning(folds = 16, iter_ = 300):
    """
    Iterively find the best hyper-parameters for the KNN model
    
    Parameters
    ----------
    folds: int
        Number of cross-validation folds to test on
    iter_: int
        Number of iterations of random parameter searches to perform

    Returns
    -------
    Boolean, sklearn KNN classifier, float
        Flag if performance imporved, model with best hyper-parameters and corresponding score

    """ 
    
    
    print('Searching hyper parameters')
    # Deepcopy model to change parameters
    model = deepcopy(knn_clf)
    # Initiate grid of possible parameters and values to search
    param_dist = {'clf__weights':  ['uniform', 'distance'],
                     'clf__algorithm': ['ball_tree', 'kd_tree'],
                     'clf__leaf_size': [int(i) for i in range(10, 200)],
                     'clf__n_neighbors': [int(i) for i in range(5, 750)]}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(model, param_dist, trials = iter_, folds = folds)
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < knn_checkpoint_score:
        return(True, knn_clf, knn_checkpoint_score)
    else:
        return(False, model.set_params(**results['parameters']), results['score'])


def svc_hyper_parameter_tuning(model, score, folds = 16, iter_ = 36):
    """
    Iterively find the best hyper-parameters for the SVC model
    
    Parameters
    ----------
    model : sklearn SVC classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    folds: int
        Number of cross-validation folds to test on
    iter_: int
        Number of iterations of random parameter searches to perform
        
    Returns
    -------
    sklearn SVC classifier, float
        Model with best hyper-parameters and corresponding score

    """ 
    

    # Deepcopy model to change parameters    
    clf = deepcopy(model)
    print('Searching hyper parameters')
    # Initiate grid of possible parameters and values to search
    param_dist = {'clf__C':  np.logspace(-3, 2, 6),
                     'clf__gamma': np.logspace(-3, 2, 6)}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(clf, param_dist, trials = iter_, folds = folds)
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])
        
        
def C_parameter_tuning(clf, score, folds = 16, iter_ = 5):
    """
    Iterively find the best C for the sklearn model
    
    Parameters
    ----------
    clf : sklearn classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    folds: int
        Number of cross-validation folds to test on
    iter_: int
        Number of iterations of random parameter searches to perform
        
    Returns
    -------
    sklearn classifier, float
        Model with best C parameter and corresponding score

    """ 
    
    
    print('Searching for optimal C parameter')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(model, {'clf__C':  np.logspace(-2, 2, 5)}, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(model.set_params(**results['parameters']), results['score'])


def lgb_drop_lr(use_model, score, lr_drop, trees_drop, dropped_score_val, folds = 16):
    """
    Iterively find the best combination of lower learning rate and higher trees
    for a gradient boosting model
    
    Parameters
    ----------
    use_model : LightGBM classifier
        Classifier model to test
    score : Boolean/float
        Flag if scaler has already been scored, default False but pass float 
    lr_drop: float
        Current learning rate to start from
    trees_drop: int
        Current number of trees to start from
    dropped_score_val: float
        Current model performance score to improve from
    folds: int
        Number of cross-validation folds to test on
        
    Returns
    -------
    LightGBM classifier, float
        Model with best hyper-parameters and corresponding score

    """ 
    
    
    def drop_lr(use_clf, l_drop, trees, folds = 16):
        """
        Test best number of trees for the given learning rate
        
        Parameters
        ----------
        use_clf : LightGBM classifier
            Classifier model to test
        l_drop: float
            Starting learning rate
        trees: int
            Starting number of trees
        folds: int
            Number of cross-validation folds to test on
            
        Returns
        -------
        float, int
            Best score, best number of trees
            
        """ 
        
    
        # Deepcopy model to change parameters
        clf = deepcopy(use_clf)
        prev_score = -np.inf
        prev_trees = 0
        
        # Increase trees by multiples until improvement ends
        for growth in np.linspace(1.5, 15.5, 29):
            num_trees = int(trees * growth)
            # Calculate performance for trees at current learning rate
            lr_score = cross_validate(clf.set_params(**{'clf__learning_rate':l_drop, 'clf__n_estimators': num_trees}))
            
            if np.mean(lr_score) > prev_score and growth < 15.5:
                print('%s trees IMPROVEMENT, continuing'  % (num_trees))
                prev_score = np.mean(lr_score)
                prev_trees = num_trees
            else:
                print('%s trees NO IMPROVEMENT'  % (num_trees))
                return prev_score, prev_trees
           
            
    # Deepcopy model to change parameters    
    model = deepcopy(use_model)            
    improvement = 0
    
    # Decrease learning rate and find optimal trees until improvement ends
    while improvement >= 0 and trees_drop <= 10000:
        # Calculate optimal trees for learning rate
        drop_scores, drop_trees = drop_lr(model, lr_drop/2, trees_drop, folds = folds)
        print('Previous best score of: %s' % (dropped_score_val))
        print('Max test score of: %s' % (drop_scores)) 
        print('Best test trees: %s' % (drop_trees))
        improvement = drop_scores - dropped_score_val
        if improvement >= 0:
            lr_drop /= 2
            trees_drop = drop_trees
            print('Continuing Search')
            print('Trees: %s'%(trees_drop))
            print('LR: %s' % (lr_drop))
            dropped_score_val = drop_scores
        else:
            print('Optimized Trees/LR Found')
            print('---- Trees: %s'%(trees_drop))
            print('---- LR: %s'%(lr_drop))
            
    return(model.set_params(**{'clf__learning_rate':lr_drop, 'clf__n_estimators': trees_drop}), dropped_score_val)


def _linsvc(dim, stage):
    """
    Automated iteritive process to tune, train and store a sklearn linear SVC 
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global linsvc_clf
    global linsvc_checkpoint_score

    # Skip if model is not specified for tuning
    if 'linSVC' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 2:
        return
    
    print('    - linSVC -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'linSVC_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'linSVC_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        linsvc_clf = joblib.load(os.path.join(tuning_model_folder, 'linSVC_%s_%s.pkl' % (dim, stage - 1)))
        linsvc_checkpoint_score = _ret_scores(dim, 'linSVC', stage - 1)
    # Initiate pipeline and score
    else:
        linsvc_clf = Pipeline([('scale',None), ('clf',SVC(random_state = 1108, class_weight = 'balanced', kernel = 'linear', probability = True))])
        linsvc_checkpoint_score = -np.inf
        
    # Determine optimal scaler
    if stage == 0:
        scale, linsvc_checkpoint_score = test_scaler(linsvc_clf, skip = True) 
        linsvc_clf = linsvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'linSVC', linsvc_clf, linsvc_checkpoint_score)
        return
    
    # Determine optimal C value
    elif stage == 1:
        linsvc_clf, linsvc_checkpoint_score = C_parameter_tuning(linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, dim, 'linSVC', linsvc_clf, linsvc_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters
    elif stage == 2:        
        scale, linsvc_checkpoint_score = test_scaler(linsvc_clf, prev_score = linsvc_checkpoint_score, prev_scaler = linsvc_clf.steps[0][1], skip = True) 
        linsvc_clf = linsvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'linSVC', linsvc_clf, linsvc_checkpoint_score, final = True)
        return
    
    
def _rbfsvc(dim, stage):
    """
    Automated iteritive process to tune, train and store a sklearn rbf SVC 
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global rbfsvc_clf
    global rbfsvc_checkpoint_score

    # Skip if model is not specified for tuning
    if 'rbfSVC' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 2:
        return
    
    print('    - rbfSVC -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'rbfSVC_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'rbfSVC_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        rbfsvc_clf = joblib.load(os.path.join(tuning_model_folder, 'rbfSVC_%s_%s.pkl' % (dim, stage - 1)))
        rbfsvc_checkpoint_score = _ret_scores(dim, 'rbfSVC', stage - 1)
    # Initiate pipeline and score
    else:
        rbfsvc_clf = Pipeline([('scale',None), ('clf',SVC(random_state = 1108, class_weight = 'balanced', kernel = 'rbf', probability = True))])
        rbfsvc_checkpoint_score = -np.inf
        
    # Determine optimal scaler
    if stage == 0:
        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf) 
        rbfsvc_clf = rbfsvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'rbfSVC', rbfsvc_clf, rbfsvc_checkpoint_score)
        return
            
    # Determine optimal hyper-parameters (C, gamma)  
    elif stage == 1:
        rbfsvc_clf, rbfsvc_checkpoint_score = svc_hyper_parameter_tuning(rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, dim, 'rbfSVC', rbfsvc_clf, rbfsvc_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters
    elif stage == 2:
        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf, prev_score = rbfsvc_checkpoint_score, prev_scaler = rbfsvc_clf.steps[0][1]) 
        rbfsvc_clf = rbfsvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'rbfSVC', rbfsvc_clf, rbfsvc_checkpoint_score, final = True)
        return


def _polysvc(dim, stage):
    """
    Automated iteritive process to tune, train and store a sklearn polynomial SVC 
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global polysvc_clf
    global polysvc_checkpoint_score

    # Skip if model is not specified for tuning
    if 'polySVC' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 3:
        return
    
    print('    - polySVC -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'polySVC_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder)  or 'polySVC_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        polysvc_clf = joblib.load(os.path.join(tuning_model_folder, 'polySVC_%s_%s.pkl' % (dim, stage - 1)))
        polysvc_checkpoint_score = _ret_scores(dim, 'polySVC', stage - 1)
    # Initiate pipeline and score
    else:
        polysvc_clf = Pipeline([('scale',None), ('clf',SVC(random_state = 1108, class_weight = 'balanced', kernel = 'poly', probability = True))])
        polysvc_checkpoint_score = -np.inf
        
    # Determine optimal scaler
    if stage == 0:
        scale, polysvc_checkpoint_score = test_scaler(polysvc_clf, skip = True) 
        polysvc_clf = polysvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'polySVC', polysvc_clf, polysvc_checkpoint_score)
        return

    # Determine optimal polynomial degree
    elif stage == 1:
        polysvc_clf, polysvc_checkpoint_score = test_degree(polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, dim, 'polySVC', polysvc_clf, polysvc_checkpoint_score)
        return

    # Determine optimal hyper-parameters (C, gamma)      
    elif stage == 2:
        polysvc_clf, polysvc_checkpoint_score = svc_hyper_parameter_tuning(polysvc_clf, polysvc_checkpoint_score, iter_ = 12)
        _save_model(stage, dim, 'polySVC', polysvc_clf, polysvc_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters            
    elif stage == 3:
        scale, polysvc_checkpoint_score = test_scaler(polysvc_clf, prev_score = polysvc_checkpoint_score, prev_scaler = polysvc_clf.steps[0][1], skip = True) 
        polysvc_clf = polysvc_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'polySVC', polysvc_clf, polysvc_checkpoint_score, final = True)
        return
    
    
def _log(dim, stage):
    """
    Automated iteritive process to tune, train and store a sklearn logistic regression
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    

    # Declare global classifier and score
    global log_clf
    global log_checkpoint_score

    # Skip if model is not specified for tuning
    if 'logRegression' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 4:
        return
    
    print('    - logRegression -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'logRegression_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'logRegression_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        log_clf = joblib.load(os.path.join(tuning_model_folder, 'logRegression_%s_%s.pkl' % (dim, stage - 1)))
        log_checkpoint_score = _ret_scores(dim, 'logRegression', stage - 1)
    # Initiate pipeline and score
    else:
        log_clf = Pipeline([('scale',None), ('clf',LogisticRegression(max_iter = 1000, random_state = 1108, class_weight = 'balanced'))])
        log_checkpoint_score = -np.inf
        
    # Determine optimal scaler
    if stage == 0:
        scale, log_checkpoint_score = test_scaler(log_clf) 
        log_clf = log_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'logRegression', log_clf, log_checkpoint_score)
        return
    
    # Determine optimal solver algorithm
    elif stage == 1:
        log_clf, log_checkpoint_score = test_solver(log_clf, log_checkpoint_score)
        _save_model(stage, dim, 'logRegression', log_clf, log_checkpoint_score)
        return

    # Retest to determine optimal scaler with new solver    
    elif stage == 2:
        scale, log_checkpoint_score = test_scaler(log_clf, prev_score = log_checkpoint_score, prev_scaler = log_clf.steps[0][1]) 
        log_clf = log_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'logRegression', log_clf, log_checkpoint_score)
        return
    
    # Determine optimal C value
    elif stage == 3:
        log_clf, log_checkpoint_score = C_parameter_tuning(log_clf, log_checkpoint_score)
        _save_model(stage, dim, 'logRegression', log_clf, log_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters
    elif stage == 4:
        log_clf, log_checkpoint_score = test_solver(log_clf, log_checkpoint_score)
        _save_model(stage, dim, 'logRegression', log_clf, log_checkpoint_score, final = True)
        return


def _knn(dim, stage):
    """
    Automated iteritive process to tune, train and store a sklearn K-nearest neighbor
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global knn_clf
    global knn_checkpoint_score

    # Skip if model is not specified for tuning
    if 'KNN' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 2:
        return
    
    print('    - KNN -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'KNN_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'KNN_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        knn_clf = joblib.load(os.path.join(tuning_model_folder, 'KNN_%s_%s.pkl' % (dim, stage - 1)))
        knn_checkpoint_score = _ret_scores(dim, 'KNN', stage - 1)
    # Initiate pipeline and score
    else:
        knn_clf = Pipeline([('scale',None), ('clf',KNeighborsClassifier())])
        knn_checkpoint_score = -np.inf
        
    # Determine optimal scaler
    if stage == 0:
        scale, knn_checkpoint_score = test_scaler(knn_clf) 
        knn_clf = knn_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'KNN', knn_clf, knn_checkpoint_score)
        return

    # Determine optimal hyper-parameters (leaves, weights, neighbors, algorithm)  
    # Retest in 25 trial increments until new combination is better than default
    elif stage == 1:   
        continue_, knn_clf, knn_checkpoint_score = knn_hyper_parameter_tuning()
        while continue_:
            continue_, knn_clf, knn_checkpoint_score = knn_hyper_parameter_tuning(iter_ = 25) 
        _save_model(stage, dim, 'KNN', knn_clf, knn_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters
    elif stage == 2:
        scale, knn_checkpoint_score = test_scaler(knn_clf, prev_score = knn_checkpoint_score, prev_scaler = knn_clf.steps[0][1]) 
        knn_clf = knn_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'KNN', knn_clf, knn_checkpoint_score, final = True)
        return    
   
     
def _lgb(dim, stage): 
    """
    Automated iteritive process to tune, train and store a LightGBM gradient boosting 
    model for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global lgb_clf
    global lgb_checkpoint_score
    
    # Skip if model is not specified for tuning
    if 'LightGBM' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 4:
        return
    
    print('    - LightGBM -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'LightGBM_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'LightGBM_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        lgb_clf = joblib.load(os.path.join(tuning_model_folder, 'LightGBM_%s_%s.pkl' % (dim, stage - 1))) 
        lgb_checkpoint_score = _ret_scores(dim, 'LightGBM', stage - 1)
    # Initiate pipeline and score
    else:
        lgb_clf = Pipeline([('scale',None), ('clf',lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True))])
        lgb_checkpoint_score = -np.inf
    
    # Determine optimal scaler
    if stage == 0:
        scale, lgb_checkpoint_score = test_scaler(lgb_clf) 
        lgb_clf = lgb_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'LightGBM', lgb_clf, lgb_checkpoint_score)
        return

    # Determine optimal learning rate for 100 trees
    elif stage == 1:
        lgb_clf, lgb_checkpoint_score = lgb_find_lr(lgb_clf, lgb_checkpoint_score) 
        _save_model(stage, dim, 'LightGBM', lgb_clf, lgb_checkpoint_score)
        return

    # Determine optimal hyper-parameters (leaves, min_child, subsample, max_bin)
    # Retest in 25 trial increments until new combination is better than default
    elif stage == 2:        
        continue_tree, lgb_clf, lgb_checkpoint_score = lgb_tree_params(lgb_clf, lgb_checkpoint_score)
        while continue_tree:
            continue_tree, lgb_clf, lgb_checkpoint_score = lgb_tree_params(lgb_clf, lgb_checkpoint_score, iter_=25)
        _save_model(stage, dim, 'LightGBM', lgb_clf, lgb_checkpoint_score)
        return

    # Retest to determine optimal scaler with new hyperparameters
    if stage == 3:
        scale, lgb_checkpoint_score = test_scaler(lgb_clf, prev_score = lgb_checkpoint_score, prev_scaler = lgb_clf.steps[0][1])
        lgb_clf = lgb_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'LightGBM', lgb_clf, lgb_checkpoint_score)
        return
    
    # Optimize decreased learning rate with increased trees
    elif stage == 4:
        lgb_clf, lgb_checkpoint_score = lgb_drop_lr(lgb_clf, lgb_checkpoint_score,
                                                    lgb_clf.get_params()['clf__learning_rate'], 
                                                    lgb_clf.get_params()['clf__n_estimators'], 
                                                    lgb_checkpoint_score)
        _save_model(stage, dim, 'LightGBM', lgb_clf, lgb_checkpoint_score, final = True)
        print('Tuned LightGBM %s Model:' % (dim))
        return
    
    
def _dartlgb(dim, stage): 
    """
    Automated iteritive process to tune, train and store a LightGBM gradient boosting
    model using DART algorithm for a specific target.
    
    Parameters
    ----------
    dim : str
        Target to predict (i.e. loan step 1)
    stage : int
        Which step of tuning to perform

    """ 
    
    
    # Declare global classifier and score
    global lgb_dart_clf
    global lgb_dart_checkpoint_score
    
    # Skip if model is not specified for tuning
    if 'DartLightGBM' not in _config.base_tuning_models:
        return
    # Skip if stage is past model maximum
    if stage > 4:
        return
    
    print('    - DartLightGBM -    ')
    
    # Skip if stage has already been completed and the model stored
    if 'DartLightGBM_%s_%s.pkl' % (dim, stage) in os.listdir(tuning_model_folder) or 'DartLightGBM_%s_%s.pkl' % (dim, stage) in os.listdir(final_model_folder):
        return
    # IF stages have already been completed, load the latest model and score
    elif stage != 0:
        lgb_dart_clf = joblib.load(os.path.join(tuning_model_folder, 'DartLightGBM_%s_%s.pkl' % (dim, stage - 1))) 
        lgb_dart_checkpoint_score = _ret_scores(dim, 'DartLightGBM', stage - 1)
    # Initiate pipeline and score
    else:
        lgb_dart_clf = Pipeline([('scale',None), ('clf',lgb.LGBMClassifier(random_state = 1108, learning_rate = .25, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True, boosting_type = 'dart'))])
        lgb_dart_checkpoint_score = -np.inf
    
    # Determine optimal scaler
    if stage == 0:
        scale, lgb_dart_checkpoint_score = test_scaler(lgb_dart_clf) 
        lgb_dart_clf = lgb_dart_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'DartLightGBM', lgb_dart_clf, lgb_dart_checkpoint_score)
        return
    
    # Determine optimal learning rate for 100 trees
    elif stage == 1:
        lgb_dart_clf, lgb_dart_checkpoint_score = lgb_dart_find_lr(lgb_dart_clf, lgb_dart_checkpoint_score, lr_ = .25) 
        _save_model(stage, dim, 'DartLightGBM', lgb_dart_clf, lgb_dart_checkpoint_score)
        return

    # Determine optimal hyper-parameters (leaves, min_child, subsample, max_bin)
    # Retest in 25 trial increments until new combination is better than default
    elif stage == 2:
        continue_tree, lgb_dart_clf, lgb_dart_checkpoint_score = lgb_tree_params(lgb_dart_clf, lgb_dart_checkpoint_score)
        while continue_tree:
            continue_tree, lgb_dart_clf, lgb_dart_checkpoint_score = lgb_tree_params(lgb_dart_clf, lgb_dart_checkpoint_score, iter_=25)
        _save_model(stage, dim, 'DartLightGBM', lgb_dart_clf, lgb_dart_checkpoint_score)
        return
    
    # Retest to determine optimal scaler with new hyperparameters
    elif stage == 3:
        scale, lgb_dart_checkpoint_score = test_scaler(lgb_dart_clf, prev_score = lgb_dart_checkpoint_score, prev_scaler = lgb_dart_clf.steps[0][1]) 
        lgb_dart_clf = lgb_dart_clf.set_params(**{'scale': scale})
        _save_model(stage, dim, 'DartLightGBM', lgb_dart_clf, lgb_dart_checkpoint_score)
        return
    
    # Optimize decreased learning rate with increased trees
    elif stage == 4:
        lgb_dart_clf, lgb_dart_checkpoint_score = lgb_drop_lr(lgb_dart_clf, lgb_dart_checkpoint_score,
                                                    lgb_dart_clf.get_params()['clf__learning_rate'], 
                                                    lgb_dart_clf.get_params()['clf__n_estimators'], 
                                                    lgb_dart_checkpoint_score)
        _save_model(stage, dim, 'DartLightGBM', lgb_dart_clf, lgb_dart_checkpoint_score, final = True)
        print('Tuned DartLightGBM %s Model:' % (dim))
        return
    
    
def _save_scores(dimen, mod, res, stg, final = False):
    """
    Save model results to text files
    
    Parameters
    ----------
    dimen: str
        Target to predict (i.e. loan step 1)
    mod: str
        Model name (i.e. LogRegression)
    res: float
        Model score
    stg: int
        Stage of model tuning
    final: Boolean
        Flag to save results to final result folder instead of tuning folder
    
    """ 
    
    
    if final:
        f = open(os.path.join(final_result_folder, '%s_tuning.txt' % (dimen)), 'a')
        f.write('%s: %s, \n'%(mod, res))
        f.close()
    else:
        f = open(os.path.join(tuning_result_folder, '%s_%s.txt' % (dimen, mod)), 'a')
        f.write('%s:%s, \n'%(stg, res))
        f.close()     


def _ret_scores(dimen, mod, stg):  
    """
    Load previously stored model result from prior tuning stage
    
    Parameters
    ----------
    dimen: str
        Target to predict (i.e. loan step 1)
    mod: str
        Model name (i.e. LogRegression)
    stg: int
        Stage of model tuning
    
    """ 
    
    
    f = open(os.path.join(tuning_result_folder, '%s_%s.txt' % (dimen, mod)), 'r')
    _scores = f.read()
    _score = float(_scores.split(',')[-2].split(':')[1])
    return(float(_score))
    
    
def _save_model(stage, dim, name, model, checkpoint, final = False):
    """
    Save model to disk after stage tuning
    
    Parameters
    ----------
    stage: int
        Stage of model tuning
    dim: str
        Target to predict (i.e. loan step 1)
    name: str
        Model name (i.e. LogRegression)
    model: sklearn classifier
        Classifier to save
    checkpoint: float
        Model score
    final: Boolean
        Flag to save results to final result folder instead of tuning folder
    
    """ 
    
    
    print('Storing Stage %s %s %s Model' % (stage, dim, name))
    if final:
        joblib.dump(model, os.path.join(final_model_folder, '%s_%s_%s.pkl' % (name, dim, stage)))     
    else:
        joblib.dump(model, os.path.join(tuning_model_folder, '%s_%s_%s.pkl' % (name, dim, stage)))     
    _save_scores(dim, name, checkpoint, stage, final) 


def stage_dispatch(dimen, stage):
    """
    Dispatch model tuning to automated pipelines
    
    Parameters
    ----------
    dimen: str
        Target to predict (i.e. loan)
    stage: int
        Stage of model tuning

    Returns
    -------
    int
        stage + 1

    """ 
    
    
    _dartlgb(dimen, stage)
    _lgb(dimen, stage)
    _log(dimen, stage)
    _polysvc(dim, stage)
    _rbfsvc(dim, stage)
    _linsvc(dim, stage)
    _knn(dim, stage)
    return(stage + 1)
            
    
def tune(dimension, _step): 
    """
    Perform automated model tuning
    
    Parameters
    ----------
    dimension: str
        Target to predict (i.e. loan)
    _step: int
        Stage of model tuning

    """ 

    
    # Declare global x and y values
    global x
    global y
    
    # Ensure required folders exist
    for loc in [tuning_model_folder, final_model_folder, tuning_result_folder, final_result_folder]:
        ensure_dir(loc)
        
    # Load training data
    data = pd.read_csv(os.path.join(data_folder, '%s_%s_data.csv' % (dimension, _step)))
    data.set_index('rel_id', inplace = True)
    
    # Load identified significant features
    x_feats = features[dimension][_step]
    
    # Split X and Y data
    y = data['label']
    x = data[x_feats]
    
    # Convert all X data to float
    for feat in x_feats:
        x[feat] = x[feat].astype(float)
        
    # Drop full data set
    data = None
    
    # Dispatch model tuning pipelines
    stage = 0
    while stage < 10:
        print('   -- Stage %s --    ' % (stage))
        stage = stage_dispatch(dimension, stage)


if __name__ == '__main__':
    all_dims = _config.base_tuning_aspects
    peice = _config.step
    for dim in all_dims:
        print('   -- Tuning %s %s --    ' % (dim.upper(), peice))
        
        data_folder = os.path.join(cur_path, 'modelling', 'data')
        tuning_model_folder = os.path.join(cur_path, 'modelling', 'models', dim, 'clf', '%s_tuning' % (peice))
        final_model_folder = os.path.join(cur_path, 'modelling', 'models', dim, 'clf', '%s_final' % (peice))
        tuning_result_folder = os.path.join(cur_path, 'modelling', 'results', dim, 'clf', '%s_tuning' % (peice))
        final_result_folder = os.path.join(cur_path, 'modelling', 'results', dim, 'clf', '%s_final' % (peice))

        tune(dim, peice)
