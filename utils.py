import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
#sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))
#sys.path.insert(3, cur_path)
#sys.path.insert(4, os.path.join(cur_path, 'modelling'))

import _config
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np

#import lightgbm as lgb
import importlib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight
from sklearn.externals import joblib
from sklearn.metrics import log_loss, mean_squared_error
import random
from copy import deepcopy
from joblib import dump, load
import json
from random import sample 
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from progress_bar import progress
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import lightgbm
from sklearn.linear_model import LogisticRegression, Lasso

def ensure_dir(file_path):
    """ Create directory if doesn't exist """

#    directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    
def cross_validate(x,y,est,scaler, only_scores = False, njobs = -1, verbose = False): 
#    x,y,est,scaler, only_scores, njobs, verbose = X,Y,model,scale, True, -1, True
    if len(y.unique()) == 2:
        splitter = StratifiedKFold(n_splits = 8, random_state = 53)
    else:
        splitter = KFold(n_splits = 8, random_state = 53)
        
    if est.__class__ == lightgbm.sklearn.LGBMClassifier:
        njobs = 1

    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold)
    
    jobs = []
    for train, test in all_folds:
        jobs.append([scaler.fit_transform(x.iloc[train]), scaler.fit_transform(x.iloc[test]), y.iloc[train], y.iloc[test], est])
    
    if njobs == 1:
        cv_results = []
        for job in jobs:
            if only_scores:
                cv_results.append(_single_core_scorer(job))
            else:
                cv_results.append(_single_core_solver(job))
    else:
        if verbose:
            cv_results = joblib.Parallel(n_jobs = njobs, verbose = 25)(joblib.delayed(_single_core_scorer) (i) for i in jobs)
        else:
            cv_results = joblib.Parallel(n_jobs = njobs)(joblib.delayed(_single_core_scorer) (i) for i in jobs)
    
    results = np.mean(cv_results)
    return(results)
        
        
def _save_scores(dimen, mod, res, stg, final = False):
#    dimen, mod, res, stg, final = dim, name, checkpoint, stage, final
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
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'final', 'results')
    else:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'tuning', 'results')
    ensure_dir(result_folder)
    
    if os.path.isfile(os.path.join(result_folder, '%s.json' % (mod))):
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'r') as fp:
            scores = json.load(fp)
        scores[stg] = res
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'w') as fp:
            json.dump(scores, fp)
    else:
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'w') as fp:
            json.dump({stg: res}, fp)
            
  
def _save_feats(dimen, mod, feats, stg, final = False):
#    dimen, mod, feats, stg, final = dim, name, features, stage, final

    if final:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'final', 'features')
    else:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'tuning', 'features')
    ensure_dir(result_folder)
    
    if os.path.isfile(os.path.join(result_folder, '%s.json' % (mod))):
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'r') as fp:
            scores = json.load(fp)
        scores[stg] = feats
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'w') as fp:
            json.dump(scores, fp)
    else:
        with open(os.path.join(result_folder, '%s.json' % (mod)), 'w') as fp:
            json.dump({stg: feats}, fp)
            
            
def _save_model(stage, dim, name, model, scale, checkpoint, features, final = False):
#    stage, dim, name, model, scale, checkpoint, features, final = stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, False
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
        model_folder = os.path.join(cur_path, 'modelling', dim, 'final', 'models', name)
    else:
        model_folder = os.path.join(cur_path, 'modelling', dim, 'tuning', 'models', name)
        
    ensure_dir(model_folder)
    dump(model, os.path.join(model_folder, '%s.pkl' % (stage)))    

    if final:
        scale_folder = os.path.join(cur_path, 'modelling', dim, 'final', 'scalers', name)
    else:
        scale_folder = os.path.join(cur_path, 'modelling', dim, 'tuning', 'scalers', name)
        
    ensure_dir(scale_folder)
    dump(scale, os.path.join(scale_folder, '%s.pkl' % (stage)))  

    _save_scores(dim, name, checkpoint, stage, final) 
    _save_feats(dim, name, features, stage, final) 

#def _single_core_solver(input_vals):
##   trainx, testx, trainy, testy, model, metric = job
#    trainx, testx, trainy, testy, model, metric = input_vals
#
#    test_weights = class_weight.compute_class_weight('balanced',
#                                np.unique(trainy),trainy)    
#    test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}    
#
#    model.fit(trainx, trainy)   
#    
#     
#    if metric == 'logloss':
#        pred = model.predict_proba(testx)
#        score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy])
#        score *= -1
#    elif metric == 'accuracy':
#        pred = model.predict(testx)
#        score = accuracy_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
#    elif metric == 'f1':
#        pred = model.predict(testx)
#        score = f1_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
#    elif metric == 'recall':
#        pred = model.predict(testx)
#        score = recall_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
#    elif metric == 'prec':
#        pred = model.predict(testx)
#        score = precision_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
#        
#    return(score)


def _single_core_scorer(input_vals):
#   trainx, testx, trainy, testy, model = job
    trainx, testx, trainy, testy, model = input_vals
    if len(trainy.unique()) == 2:
        obj = 'class'
    else:
        obj = 'reg'
        
    if obj == 'class':
        test_weights = class_weight.compute_class_weight('balanced',
                                    np.unique(trainy),trainy)    
        test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}    
        
    model.fit(trainx, trainy)   
    
    if obj == 'class':    
        pred = model.predict_proba(testx)
    else:
        pred = model.predict(testx)
    #pred_bin = [0 if i[0] > .5 else 1 for i in pred]
    if obj == 'class':
        score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy]) * -1
    else:
        score = mean_squared_error(testy, pred) * -1
    return(score)
    

def test_scaler(clf, x, y, verbose = False, prev_score = False, prev_scaler = False, skip = False, prog = True):  
#   clf, x, y, verbose, prev_score, prev_scaler, skip, prog = lgb_clf, X, Y, True, False, False, False, True
    if prog:
        print('Searching for best scaler.')
    # Initiate data storage dictionary
    scores = {}
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    
    # Test performance of prepending scalers to sklearn pipeline
    total_scales = 3
    cur_scale = 0
    for scale, name in zip([StandardScaler(), MinMaxScaler(), RobustScaler()], ['Standard', 'MinMax', 'Robust']):
        if not verbose and prog:
            progress(cur_scale, total_scales)
        if skip and name == 'Robust':
            continue
        
        # Skip testing of scaler already included in current pipeline
        if prev_scaler and prev_score and prev_scaler.__class__ == scale.__class__:
            if verbose:
                print('%s Already Included' % (name))
                print('Score: %.5f' % (prev_score))
            continue

        scale_score = cross_validate(x,y,model,scale,only_scores=True, verbose = verbose)
    
        if verbose:
            print('%s Score: %.5f' % (name, scale_score))
        # Add scores to storage dictionary
        scores[name] = {'scale': scale}
        scores[name]['score'] = scale_score   
        cur_scale += 1
        
    # Identify and return best scaler       
    best_scale = max(scores, key = lambda x: scores[x]['score'])
    if prev_score:
        # If original scaler performs worse than new, use new
        if scores[best_scale]['score'] > prev_score:
            if prog:
                print('Using %s.' % (best_scale))
            return(scores[best_scale]['scale'], scores[best_scale]['score'])
        # If new scaler performs worse than previous, keep original
        else:
            if prog:
                print('Keeping original.')
            return(prev_scaler, prev_score)
    else:
        if prog:
            print('Using %s.' % (best_scale))        
        return(scores[best_scale]['scale'], scores[best_scale]['score'])


def test_solver(x, y, clf, scaler, prev_score, verbose = False): 
#    x, y, clf, scaler, prev_score, verbose = X[features], Y, log_clf, scale, log_checkpoint_score, False
    print('Searching for best solver.')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    # Initiate data storage dictionary        
    scores = {}
    # Test performance of using different solver algorithms    
    all_solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    cur_solver = 0
    for solve in all_solvers:
        if not verbose:
            progress(cur_solver, len(all_solvers))    
        model.set_params(**{'solver': solve})
        score = test_scaler(model, x, y, verbose = verbose, prev_score = False, prev_scaler = False, skip = False, prog = False)

#        # Skip testing of solver already tested
#        if clf.get_params() == model.get_params():
#            if verbose:
#                print('Sovler Already Included... skipping')
#            continue
        # Calculate cross-validated model performance using algorithm
#        score = cross_validate(x,y, model, scaler, only_scores=True, verbose = verbose)
        if verbose:
            print('%s Score: %.5f' % (solve, score[1]))
        # Add scores to storage dictionary
        scores[solve] = {'score': score[1], 'scale': score[0]}
        cur_solver += 1
        
    # Identify and return best scaler               
    best_solver = max(scores, key = lambda x: scores[x]['score'])
    # If original scaler performs worse than new, use new
    if scores[best_solver]['score'] > prev_score:
        print('Using %s.' % (best_solver))
        return(model.set_params(**{'solver': best_solver}), scores[best_solver]['score'], scores[best_solver]['scale']) 
    # If new scaler performs worse than previous, keep original
    else:
        print('No Improvement, Using Default')
        return(clf, prev_score, scaler)         
        
        
def stage_init(name, dimension):
#    name, dimension = name, dimension
    final_folder = os.path.join(cur_path, 'modelling', dimension, 'final', 'models', name)
    if os.path.isdir(final_folder):
        return(np.nan, False, False, False, False)
    else:
        model_folder = os.path.join(cur_path, 'modelling', dimension, 'tuning', 'models', name)
        scaler_folder = os.path.join(cur_path, 'modelling', dimension, 'tuning', 'scalers', name)
        if os.path.isdir(model_folder):
            stored_models = os.listdir(model_folder)
            prev_stage = max([int(i.replace('.pkl', '')) for i in stored_models])
            mod = load(os.path.join(model_folder, '%s.pkl' % (prev_stage)))
            scale = load(os.path.join(scaler_folder, '%s.pkl' % (prev_stage)))
            feats_folder = os.path.join(cur_path, 'modelling', dimension, 'tuning', 'features')
            with open(os.path.join(feats_folder, '%s.json' % (name)), 'r') as fp:
                feats = json.load(fp)[str(prev_stage)]
            results_folder = os.path.join(cur_path, 'modelling', dimension, 'tuning', 'results')
            with open(os.path.join(results_folder, '%s.json' % (name)), 'r') as fp:
                result = json.load(fp)[str(prev_stage)]
            return(prev_stage + 1, mod, scale, feats, result)  
        else:
            return(0, False, False, False, False)
        

def init_feat_selection(x, y, model, thresh = '2*mean'):
#    x, y, model, thresh = X, Y, rbfsvc_clf, '2*mean'
    print('Searching for best features.')
#    n_cols = len(list(x))
    log_sfm = SelectFromModel(model, threshold =  thresh)
    log_sfm.fit(x, y)
    sig_feats = [i for i,j in zip(list(x), log_sfm.get_support()) if j]
    return(sig_feats)


def feat_selection(x, y, scale, model, prev_score, _iter = 24, njobs = -1, verbose = False):
#    x, y, scale, model, prev_score, _iter, njobs, verbose = X[features], Y, scale, rbfsvc_clf, rbfsvc_checkpoint_score, 24, -1, False

    print('Searching for best features.')

    scaleX = scale.fit_transform(x)
    scaleX = pd.DataFrame(scaleX)
    scaleX.columns = list(x)
    n_cols = len(list(scaleX))
    
    def _rfe(inputs):
        _model, _n_col, _x, _y = inputs
        selector = RFE(_model, _n_col, step=1)
        selector.fit(_x, _y)
        return([i for i,j in zip(list(_x), selector.support_) if j])
        
    if _iter > n_cols:
        _iter = n_cols
        
    rfe_jobs = []
    for n_col in sample(range(n_cols), _iter):
        if n_col == 0:
            continue
        rfe_jobs.append([model, n_col, scaleX, y])
    
    feat_options = joblib.Parallel(n_jobs = njobs, verbose = 25)(joblib.delayed(_rfe) (i) for i in rfe_jobs)

    cur_iter = 0
    sig_feats = {}
    for feats in feat_options:
        if not verbose:
            progress(cur_iter, _iter)    
        feat_score = cross_validate(x[feats],y,model,scale,only_scores=True, verbose = verbose)
        
        sig_feats[cur_iter] = {'features': feats,
                             'score': feat_score}
        cur_iter += 1

    best_combination = max(sig_feats, key =lambda x: sig_feats[x]['score'])
    if sig_feats[best_combination]['score'] > prev_score:
        return(sig_feats[best_combination]['score'], sig_feats[best_combination]['features'])
    else:
        return(prev_score, list(x))
        
        
def feat_selection_2(x, y, scale, model, prev_score, _iter = 24, njobs = -1, verbose = False):
#    x, y, scale, model, prev_score, _iter, njobs, verbose = X[features], Y, scale, rbfsvr_reg, rbfsvr_checkpoint_score, 24, -1, False
    print('Searching for best features.')

    scaleX = scale.fit_transform(x)
    scaleX = pd.DataFrame(scaleX)
    scaleX.columns = list(x)
    n_cols = len(list(scaleX))
    
    def _rfe(inputs):
        _model, _n_col, _x, _y = inputs
        if len(_y.unique()) == 2:
            rfe_model = LogisticRegression(random_state = 1108)
        else:
            rfe_model = Lasso(random_state = 1108)
        selector = RFE(rfe_model, _n_col, step=1)
        selector.fit(_x, _y)
        return([i for i,j in zip(list(_x), selector.support_) if j])
        
    if _iter > n_cols:
        _iter = n_cols
        
    rfe_jobs = []
    for n_col in sample(range(n_cols), _iter):
        if n_col == 0:
            continue
        rfe_jobs.append([model, n_col, scaleX, y])
    
    feat_options = joblib.Parallel(n_jobs = njobs, verbose = 25)(joblib.delayed(_rfe) (i) for i in rfe_jobs)

    cur_iter = 0
    sig_feats = {}
    for feats in feat_options:
        if not verbose:
            progress(cur_iter, _iter)    
        feat_score = cross_validate(x[feats],y,model,scale,only_scores=True, verbose = verbose)
        
        sig_feats[cur_iter] = {'features': feats,
                             'score': feat_score}
        cur_iter += 1

    best_combination = max(sig_feats, key =lambda x: sig_feats[x]['score'])
    if sig_feats[best_combination]['score'] > prev_score:
        return(sig_feats[best_combination]['score'], sig_feats[best_combination]['features'])
    else:
        return(prev_score, list(x))
        
        
def random_search(x, y, model, params, _scale, trials = 25, verbose = False):  
#    x, y, model, params, _scale, trials, verbose = x, y, model, {'C':  np.logspace(-2, 2, 5)}, scale, iter_, False
    def _rand_shuffle(choices):
        """ Randomly select a hyperparameter for search """

        selection = {}
        for key, value in choices.items():
            selection[key] = random.choice(value)
        return(selection)
        
    
    # Initiate data storage dictionary
    search_scores = {}
    
    # Identify current classifier parameters
    current_params = deepcopy(model).get_params()
    # Calculate maximum possible parameter combinations
    max_combinations = np.prod([len(i) for i in params.values()])
    
    # Limit trials to maximum combinations if higher
    if trials > max_combinations:
        trials = max_combinations
    
    
    for i in range(trials):
        progress(i, trials)
        
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
#            estimator = deepcopy(model)
            estimator = model.set_params(**{key: value})
            search_scores[iter_name][key] = value
            if verbose:
                try:
                    print('%s: %.3f' % (key, value))
                except TypeError:
                    print('%s: %s' % (key, value))
                    
        # Skip testing if new set of hyper-parameters is the same as the 
        # current parameter combination 
        if estimator.get_params() == current_params:
            if verbose:
                print('Parameters already in default.')
            iter_score = -np.inf
        # Calculate model performance with new hyper-parameters using 
        # cross-validation
        else:
            try:
                iter_score = cross_validate(x,y,estimator,_scale,only_scores=True, verbose = verbose)
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
    

def drop_lr(_x, _y, _scale, _clf, _param_search, _verbose = False):  
#    _x, _y, _scale, _clf, _param_search, _verbose = x, y, scale, model, param_search[tree_iter-1], True
    clf = deepcopy(_clf)
    _lr = _param_search['lr']/2
    _trees = _param_search['best_trees']
#    _iter_growth = int(_param_search['best_trees']/4)
    if _verbose:
        print('Learning Rate: %.5f' % (_lr))  
    _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr}), _scale,only_scores=True, verbose = _verbose)
    scores = {_trees: _score}

    for growth in [.9, .8, .7, .6, .5, .4, .3, .2]:
        _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr, 'n_estimators': int(_trees * growth)}), _scale,only_scores=True, verbose = _verbose)
        if _score < max(scores.values()):
            break
        if _verbose:
            print('Trees: %s' % (int(_trees * growth)))
            print('  Score: %.5f' % (_score))
        scores[int(_trees * growth)] = _score
    for growth in [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]:
        _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr, 'n_estimators': int(_trees * growth)}), _scale,only_scores=True, verbose = _verbose)
        if _score < max(scores.values()):
            break
        if _verbose:
            print('Trees: %s' % (int(_trees * growth)))
            print('  Score: %.5f' % (_score))
        scores[int(_trees * growth)] = _score
    return(scores)
    
        
def lgb_drop_lr(_model, x, y, scale, start_score, verbose = True):
#    _model, x, y, scale, start_score, verbose = lgb_clf, X[features], Y, scale, lgb_checkpoint_score, True
    model = deepcopy(_model)            

    init_score = check_lr(x, y, scale, model, model.get_params()['learning_rate'], _verbose = verbose)
    init_best_trees, init_best_score = [[i,j] for i,j in init_score.items() if j == max(init_score.values())][0]
    tree_iter = 0
    param_search = {tree_iter: {'lr':model.get_params()['learning_rate'], 'best_trees': init_best_trees, 'best_score': init_best_score} }
    improvement = init_best_score - start_score
    model.set_params(**{'n_estimators': init_best_trees})

    # Decrease learning rate and find optimal trees until improvement ends
    while improvement >= 0:
        # Calculate optimal trees for learning rate
        tree_iter += 1
        cur_lr = param_search[tree_iter-1]['lr']/2
#        cur_trees = param_search[tree_iter-1]['best_trees']
        drop_score = drop_lr(x, y, scale, model, param_search[tree_iter-1], _verbose = verbose)
#        drop_score = {i['trees']: i['score'] for i in drop_score.values()}
        drop_best_trees, drop_best_score = [[i,j] for i,j in drop_score.items() if j == max(drop_score.values())][0]
        print('Best Results: Trees = %i, Score = %.3f' % (drop_best_trees, drop_best_score))
        improvement = drop_best_score - param_search[tree_iter-1]['best_score']
        if improvement > 0:
            param_search[tree_iter] = {'lr':cur_lr, 'best_trees': drop_best_trees, 'best_score': drop_best_score}
    return(model.set_params(**{'learning_rate':param_search[max(param_search.keys())]['lr'], 'n_estimators': param_search[max(param_search.keys())]['best_trees']}), param_search[max(param_search.keys())]['best_score'])


def check_lr(_x, _y, _scale, _clf, _lr, _verbose = False):  
#    _x, _y, _scale, _clf, _lr, _verbose = x, y, scale, model, lr_, True
    clf = deepcopy(_clf)
    if _verbose:
        print('Learning Rate: %.5f' % (_lr))
    scores = {}
    
    _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr, 'n_estimators': 100}), _scale,only_scores=True, verbose = _verbose)
    scores[100] = _score

    for tree in [90, 75, 60, 50]:
        _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr, 'n_estimators': tree}), _scale,only_scores=True, verbose = _verbose)
        if _score < max(scores.values()):
            break
        if _verbose:
            print('Trees: %s' % (tree))
            print('  Score: %.5f' % (_score))
        scores[tree] = _score
    for tree in [110, 125, 135, 150]:
        _score = cross_validate(_x, _y, clf.set_params(**{'learning_rate':_lr, 'n_estimators': tree}), _scale,only_scores=True, verbose = _verbose)
        if _score < max(scores.values()):
            break
        if _verbose:
            print('Trees: %s' % (tree))
            print('  Score: %.5f' % (_score))
        scores[tree] = _score
    return (scores)
    
    
def rf_trees(x, y, scale, clf, prev_score, verbose = True):  
#    x, y, scale, clf, prev_score, verbose = X, Y, scale, rf_reg, rf_checkpoint_score, True
    print('Setting best trees for Random Forest.')
    model = deepcopy(clf)
    scores = {}
    
    start_score = cross_validate(x, y, model, scale, only_scores=True, verbose = verbose)
    scores[100] = start_score
    new_trees = 100

    add_trees = True
    while add_trees:
        new_trees *= 1.5

        new_score = cross_validate(x, y, model.set_params(**{'n_estimators':int(new_trees)}), scale, only_scores=True, verbose = verbose)
        if verbose:
            print('Trees: %i' % (new_trees))
            print('   Score: %.3f' % (new_score))
        
        if new_score > scores[new_trees/1.5]:
            scores[new_trees] = new_score
        else:
            add_trees = False
            
    if scores[new_trees/1.5] > prev_score:
        return(model.set_params(**{'n_estimators':int(new_trees)}), scores[new_trees/1.5])
    else:
        return(clf, prev_score)


def lgb_find_lr(_model, x, y, scale, start_score, lr_ = .1, verbose = True):
#    _model, x, y, scale, start_score, lr_, verbose = lgb_reg, X[features], Y, scale, lgbr_checkpoint_score, .1, True
    print('Searching for best learning rate')
    # Deepcopy model to change parameters
    model = deepcopy(_model)
    init_score = check_lr(x, y, scale, model, lr_, _verbose = verbose)
    init_best_trees, init_best_score = [[i,j] for i,j in init_score.items() if j == max(init_score.values())][0]
    tree_iter = 0
    param_search = {tree_iter: {'lr':lr_, 'best_trees': init_best_trees, '100_score':start_score, 'best_score': init_best_score, 'sensitivity': 1} }

    search_trees = False
    if param_search[0]['best_score'] > param_search[0]['100_score']:
        search_trees = True
        
    while search_trees:
        tree_iter += 1
        new_lr = param_search[tree_iter-1]['lr'] * ((100 + (param_search[tree_iter-1]['best_trees']-100)*param_search[tree_iter-1]['sensitivity'])/100)
        score = check_lr(x, y, scale, model, new_lr, _verbose = verbose)
        best_trees, best_score = [[i,j] for i,j in score.items() if j == max(score.values())][0]
        print('Best Results: Trees = %i, Score = %.3f, Sensitivity = %.2f' % (best_trees, best_score, param_search[tree_iter-1]['sensitivity']))
        param_search[tree_iter] = {'lr':new_lr, 'best_trees': best_trees, '100_score':score[100], 'best_score': best_score, 'sensitivity': param_search[tree_iter-1]['sensitivity']} 
        if param_search[tree_iter]['100_score'] < param_search[tree_iter-1]['100_score']:
            tree_iter += 1
            param_search[tree_iter] = deepcopy(param_search[tree_iter-2])
            param_search[tree_iter]['sensitivity'] *= .75
        if param_search[tree_iter]['best_trees'] == 100:
            search_trees = False
        if tree_iter == 200:
            search_trees = False    

    return(model.set_params(**{'learning_rate':param_search[tree_iter]['lr']}), param_search[tree_iter]['best_score'])


def knn_hyper_parameter_tuning(x, y, clf, scale, score, iter_ = 1000):
#    x,y,clf,scale,score, iter_ = X[features], Y, knn_clf, scale, knn_checkpoint_score, 10
    # Deepcopy model to change parameters    
    model = deepcopy(clf)
    print('Searching hyper parameters')
    # Initiate grid of possible parameters and values to search
    param_dist = {'weights':  ['uniform', 'distance'],
                     'algorithm': ['ball_tree', 'kd_tree'],
                     'leaf_size': [int(i) for i in range(10, 100)],
                     'n_neighbors': [int(i) for i in range(2, 100)]}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])   
    

def svc_hyper_parameter_tuning(x, y, clf, scale, score, iter_ = 25):
#    x,y,clf,scale,score, iter_ = X[features], Y, rbfsvc_clf, scale, rbfsvc_checkpoint_score, 25
    # Deepcopy model to change parameters    
    model = deepcopy(clf)
    print('Searching hyper parameters')
    # Initiate grid of possible parameters and values to search
    param_dist = {'C':  np.logspace(-2, 2, 5),
                     'gamma': np.logspace(-2, 2, 5)}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])
        
        
def C_parameter_tuning(x, y, clf, scale, score, iter_ = 5):
#    x, y, clf, scale, score, iter_ = X[features], Y, log_clf, scale, log_checkpoint_score, 5
    print('Searching for optimal C parameter')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, {'C':  np.logspace(-2, 2, 5)}, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(model.set_params(**results['parameters']), results['score'])


def alpha_parameter_tuning(x, y, clf, scale, score, iter_ = 5):
#    x, y, clf, scale, score, iter_ = X[features], Y, log_clf, scale, log_checkpoint_score, 5
    print('Searching for optimal alpha parameter')
    # Deepcopy model to change parameters
    model = deepcopy(clf)
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, {'alpha':  np.logspace(-2, 2, 5)}, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(model.set_params(**results['parameters']), results['score'])
     
        
def forest_params(x, y, clf, scale, score, iter_ = 500):
#    x,y,clf,scale,score, iter_ = X[features], Y, lgb_clf, scale, lgb_checkpoint_score, 25

    print('Setting tree parameters')
    # Deepcopy model to change parameters    
    model = deepcopy(clf)
    # Initiate grid of possible parameters and values to search
    param_dist = {'max_features': [None, 'sqrt', 'log2', .2,.3,.4,.5,.6,.7,.8,.9],
                  'max_depth': [3,5,8,10,12,15,20,25,30,None],
                  'criterion': ['gini', 'entropy'],
                  'min_samples_split': [int(i) for i in range(2, 50)]}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
        
    print('Iteration LogLoss: %.5f' % (results['score']))
    # IF a comparison score is provided, return Boolean of if improvement occured,
    # model, and score
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])           

        
def lgb_tree_params(x, y, clf, scale, score, iter_ = 250):
#    x,y,clf,scale,score, iter_ = X[features], Y, lgb_clf, scale, lgb_checkpoint_score, 25

    print('Setting tree parameters')
    # Deepcopy model to change parameters    
    model = deepcopy(clf)
    # Initiate grid of possible parameters and values to search
    param_dist = {'min_child_samples': [int(i) for i in range(2, 50)],
                  'num_leaves': [int(i) for i in range(10, 300)],
                  'subsample': [i/1000 for i in range(600, 1000)],
                  'max_bin': [int(i) for i in range(100, 500)]}
    
    # Perform random hyper-parameter search to find best combination
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
        
    print('Iteration LogLoss: %.5f' % (results['score']))
    # IF a comparison score is provided, return Boolean of if improvement occured,
    # model, and score
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])           
