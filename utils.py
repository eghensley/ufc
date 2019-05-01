import os, sys
import platform

try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()
while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))
#sys.path.insert(3, cur_path)
#sys.path.insert(4, os.path.join(cur_path, 'modelling'))

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np

#import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight
#from sklearn.externals import joblib
#import joblib
from sklearn.metrics import log_loss, mean_squared_error#, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import random
from copy import deepcopy
from joblib import dump, load, Parallel, delayed
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
#    directory = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    
def _single_core_solver(input_vals):
#   trainx, testx, trainy, testy, model = job
    trainx, testx, trainy, testy, model = input_vals
    if len(trainy.unique()) == 2:
        obj = 'class'
    else:
        obj = 'reg'        
    model.fit(trainx, trainy)       
    if obj == 'class':    
        pred = model.predict_proba(testx)
        pred = [i[1] for i in pred]
    else:
        pred = model.predict(testx)
    #pred_bin = [0 if i[0] > .5 else 1 for i in pred]
    return(pd.DataFrame(pred, testy.index))
    
    
def cross_validate(x,y,est,scaler, only_scores = True, njobs = -1, verbose = False): 
#    x,y,est,scaler, only_scores, njobs, verbose = x,Y,model,scale, False, -1, True
    if len(y.unique()) == 2:
        splitter = StratifiedKFold(n_splits = 8, random_state = 53)
    else:
        splitter = KFold(n_splits = 8, random_state = 53)        
    if est.__class__ == lightgbm.sklearn.LGBMClassifier or est.__class__ == lightgbm.sklearn.LGBMRegressor:
        njobs = 1        
    if platform.system() == 'Darwin':
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
                cv_results.append(_single_core_eval(job)) 
            else:
                cv_results.append(_single_core_solver(job))
    else:
        if only_scores:
            if verbose:
                cv_results = Parallel(n_jobs = njobs, verbose = 25)(delayed(_single_core_eval) (i) for i in jobs)
            else:
                cv_results = Parallel(n_jobs = njobs)(delayed(_single_core_eval) (i) for i in jobs)
        else:
            if verbose:
                cv_results = Parallel(n_jobs = njobs, verbose = 25)(delayed(_single_core_solver) (i) for i in jobs)
            else:
                cv_results = Parallel(n_jobs = njobs)(delayed(_single_core_solver) (i) for i in jobs)            
    if only_scores:
        results = np.mean(cv_results)
    else:
        results = pd.DataFrame()
        for df in cv_results:
            results = results.append(df)
    return(results)
        
        
def _save_scores(dimen, mod, res, stg, final = False):
#    dimen, mod, res, stg, final = dim, name, checkpoint, stage, final
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
            

def _save_meta_scores(dimen, _preddim, mod, res, stg, final = False):
#    dimen, mod, res, stg, final = dim, name, checkpoint, stage, final
    if final:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'meta', _preddim, 'final', 'results')
    else:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'meta', _preddim, 'tuning', 'results')
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
            

def _save_meta_feats(dimen, _preddim, mod, feats, stg, final = False):
#    dimen, mod, feats, stg, final = dim, name, features, stage, final
    if final:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'meta', _preddim, 'final', 'features')
    else:
        result_folder = os.path.join(cur_path, 'modelling', dimen, 'meta', _preddim, 'tuning', 'features')
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
            
            
def _save_meta_model(preddim, stage, dim, name, model, scale, checkpoint, features, final = False):
#    stage, dim, name, model, scale, checkpoint, features, final = stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, False
    print('Storing Stage %s %s %s %s Model' % (stage, dim, preddim, name))
    if final:
        model_folder = os.path.join(cur_path, 'modelling', dim, 'meta', preddim, 'final', 'models', name)
    else:
        model_folder = os.path.join(cur_path, 'modelling', dim, 'meta', preddim, 'tuning', 'models', name)        
    ensure_dir(model_folder)
    dump(model, os.path.join(model_folder, '%s.pkl' % (stage)))    
    if final:
        scale_folder = os.path.join(cur_path, 'modelling', dim, 'meta', preddim, 'final', 'scalers', name)
    else:
        scale_folder = os.path.join(cur_path, 'modelling', dim, 'meta', preddim, 'tuning', 'scalers', name)        
    ensure_dir(scale_folder)
    dump(scale, os.path.join(scale_folder, '%s.pkl' % (stage)))  
    _save_meta_scores(dim, preddim, name, checkpoint, stage, final) 
    _save_meta_feats(dim, preddim, name, features, stage, final) 

            
def _save_model(stage, dim, name, model, scale, checkpoint, features, final = False):
#    stage, dim, name, model, scale, checkpoint, features, final = stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, False
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

    
def _single_core_eval(input_vals):
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
    if obj == 'class':
        score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy]) * -1
    else:
        score = mean_squared_error(testy, pred) * -1
    return(score)
    

def test_scaler(clf, x, y, verbose = False, prev_score = False, prev_scaler = False, skip = False, prog = True):  
#   clf, x, y, verbose, prev_score, prev_scaler, skip, prog = lgb_clf, X, Y, True, False, False, False, True
    if prog:
        print('Searching for best scaler.')
    scores = {}
    model = deepcopy(clf)    
    total_scales = 3
    cur_scale = 0
    for scale, name in zip([StandardScaler(), MinMaxScaler(), RobustScaler()], ['Standard', 'MinMax', 'Robust']):
        if not verbose and prog:
            progress(cur_scale, total_scales)
        if skip and name == 'Robust':
            continue        
        if prev_scaler and prev_score and prev_scaler.__class__ == scale.__class__:
            if verbose:
                print('%s Already Included' % (name))
                print('Score: %.5f' % (prev_score))
            continue
        scale_score = cross_validate(x,y,model,scale,only_scores=True, verbose = verbose)    
        if verbose:
            print('%s Score: %.5f' % (name, scale_score))
        scores[name] = {'scale': scale}
        scores[name]['score'] = scale_score   
        cur_scale += 1        
    best_scale = max(scores, key = lambda x: scores[x]['score'])
    if prev_score:
        if scores[best_scale]['score'] > prev_score:
            if prog:
                print('Using %s.' % (best_scale))
            return(scores[best_scale]['scale'], scores[best_scale]['score'])
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
    model = deepcopy(clf)
    scores = {}
    all_solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    cur_solver = 0
    for solve in all_solvers:
        if not verbose:
            progress(cur_solver, len(all_solvers))    
        model.set_params(**{'solver': solve})
        score = test_scaler(model, x, y, verbose = verbose, prev_score = False, prev_scaler = False, skip = False, prog = False)
        if verbose:
            print('%s Score: %.5f' % (solve, score[1]))
        scores[solve] = {'score': score[1], 'scale': score[0]}
        cur_solver += 1        
    best_solver = max(scores, key = lambda x: scores[x]['score'])
    if scores[best_solver]['score'] > prev_score:
        print('Using %s.' % (best_solver))
        return(model.set_params(**{'solver': best_solver}), scores[best_solver]['score'], scores[best_solver]['scale']) 
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
        

def stage_meta_init(preddim, name, dimension):
#    name, dimension = name, dimension
    final_folder = os.path.join(cur_path, 'modelling', dimension, 'meta', preddim, 'final', 'models', name)
    if os.path.isdir(final_folder):
        return(np.nan, False, False, False, False)
    else:
        model_folder = os.path.join(cur_path, 'modelling', dimension, 'meta', preddim, 'tuning', 'models', name)
        scaler_folder = os.path.join(cur_path, 'modelling', dimension, 'meta', preddim, 'tuning', 'scalers', name)
        if os.path.isdir(model_folder):
            stored_models = os.listdir(model_folder)
            prev_stage = max([int(i.replace('.pkl', '')) for i in stored_models])
            mod = load(os.path.join(model_folder, '%s.pkl' % (prev_stage)))
            scale = load(os.path.join(scaler_folder, '%s.pkl' % (prev_stage)))
            feats_folder = os.path.join(cur_path, 'modelling', dimension, 'meta', preddim, 'tuning', 'features')
            with open(os.path.join(feats_folder, '%s.json' % (name)), 'r') as fp:
                feats = json.load(fp)[str(prev_stage)]
            results_folder = os.path.join(cur_path, 'modelling', dimension, 'meta', preddim, 'tuning', 'results')
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


def feat_selection(x, y, scale, model, prev_score, _iter = 50, njobs = -1, verbose = False):
#    x, y, scale, model, prev_score, _iter, njobs, verbose = X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False
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
    if model.__class__ == lightgbm.sklearn.LGBMClassifier or model.__class__ == lightgbm.sklearn.LGBMRegressor:
        njobs = 1        
    if platform.system() == 'Darwin':
        njobs = 1         
    if njobs == 1:
        feat_options = []
        for i,(job) in enumerate(rfe_jobs):
            progress(i, _iter)
            feat_options.append(_rfe(job)) 
    else:
        feat_options = Parallel(n_jobs = njobs, verbose = 25)(delayed(_rfe) (i) for i in rfe_jobs)
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
        
        
def feat_selection_2(x, y, scale, model, prev_score, _iter = 50, njobs = -1, verbose = False):
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
    if model.__class__ == lightgbm.sklearn.LGBMClassifier or model.__class__ == lightgbm.sklearn.LGBMRegressor:
        njobs = 1        
    if platform.system() == 'Darwin':
        njobs = 1         
    if njobs == 1:
        feat_options = []
        for i,(job) in enumerate(rfe_jobs):
            progress(i, _iter)
            feat_options.append(_rfe(job)) 
    else:
        feat_options = Parallel(n_jobs = njobs, verbose = 25)(delayed(_rfe) (i) for i in rfe_jobs)
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
    search_scores = {}
    current_params = deepcopy(model).get_params()
    max_combinations = np.prod([len(i) for i in params.values()])
    if trials > max_combinations:
        trials = max_combinations    
    for i in range(trials):
        progress(i, trials)
        iter_params = _rand_shuffle(params)
        iter_name = '_'.join("{!s}={!r}".format(k,v) for (k,v) in iter_params.items())
        while iter_name in search_scores.keys():
            iter_params = _rand_shuffle(params)
            iter_name = '_'.join("{!s}={!r}".format(k,v) for (k,v) in iter_params.items())         
        search_scores[iter_name] = {}
        for key, value in iter_params.items():
            estimator = deepcopy(model)
            estimator = estimator.set_params(**{key: value})
            search_scores[iter_name][key] = value
            if verbose:
                try:
                    print('%s: %.3f' % (key, value))
                except TypeError:
                    print('%s: %s' % (key, value))
        if estimator.get_params() == current_params:
            if verbose:
                print('Parameters already in default.')
            iter_score = -np.inf
        else:
            try:
                iter_score = cross_validate(x,y,estimator,_scale,only_scores=True, verbose = verbose)
            except:
                iter_score = -np.inf
        if verbose:
            print('- score: %.5f' % (iter_score))
        search_scores[iter_name]['score'] = iter_score
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
    while improvement >= 0:
        tree_iter += 1
        cur_lr = param_search[tree_iter-1]['lr']/2
        drop_score = drop_lr(x, y, scale, model, param_search[tree_iter-1], _verbose = verbose)
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
    model = deepcopy(clf)
    print('Searching hyper parameters')
    param_dist = {'weights':  ['uniform', 'distance'],
                     'algorithm': ['ball_tree', 'kd_tree'],
                     'leaf_size': [int(i) for i in range(10, 100)],
                     'n_neighbors': [int(i) for i in range(2, 100)]}
    
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])   
    

def svc_hyper_parameter_tuning(x, y, clf, scale, score, iter_ = 100):
#    x,y,clf,scale,score, iter_ = X[features], Y, rbfsvc_clf, scale, rbfsvc_checkpoint_score, 25
    model = deepcopy(clf)
    print('Searching hyper parameters')
    param_dist = {'C':  np.logspace(-2, 2, 10),
                     'gamma': np.logspace(-2, 2, 10)}    
    results = random_search(x, y, model, param_dist, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])
        
        
def C_parameter_tuning(x, y, clf, scale, score, iter_ = 25):
#    x, y, clf, scale, score, iter_ = X[features], Y, log_clf, scale, log_checkpoint_score, 5
    print('Searching for optimal C parameter')
    model = deepcopy(clf)    
    results = random_search(x, y, model, {'C':  np.logspace(-2, 2, 25)}, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    if results['score'] < score:
        return(clf, score)
    else:
        return(model.set_params(**results['parameters']), results['score'])


def alpha_parameter_tuning(x, y, clf, scale, score, iter_ = 25):
#    x, y, clf, scale, score, iter_ = X[features], Y, log_clf, scale, log_checkpoint_score, 5
    print('Searching for optimal alpha parameter')
    model = deepcopy(clf)    
    results = random_search(x, y, model, {'alpha':  np.logspace(-2, 2, 25)}, scale, trials = iter_)  
    print('Hyperparameter iteration LogLoss: %.5f' % (results['score']))
    if results['score'] < score:
        return(clf, score)
    else:
        return(model.set_params(**results['parameters']), results['score'])
     
        
def forest_params(x, y, clf, scale, score, iter_ = 1000):
#    x,y,clf,scale,score, iter_ = X[features], Y, lgb_clf, scale, lgb_checkpoint_score, 25
    print('Setting tree parameters')
    model = deepcopy(clf)
    param_dist = {'max_features': [None, 'sqrt', 'log2', .2,.3,.4,.5,.6,.7,.8,.9],
                  'max_depth': [3,5,8,10,12,15,20,25,30,None],
                  'criterion': ['gini', 'entropy'],
                  'min_samples_split': [int(i) for i in range(2, 50)]}    
    results = random_search(x, y, model, param_dist, scale, trials = iter_)          
    print('Iteration LogLoss: %.5f' % (results['score']))
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])           

        
def lgb_tree_params(x, y, clf, scale, score, iter_ = 1000):
#    x,y,clf,scale,score, iter_ = X[features], Y, lgb_clf, scale, lgb_checkpoint_score, 25
    print('Setting tree parameters')
    model = deepcopy(clf)
    param_dist = {'min_child_samples': [int(i) for i in range(2, 50)],
                  'num_leaves': [int(i) for i in range(10, 300)],
                  'subsample': [i/1000 for i in range(600, 1000)],
                  'max_bin': [int(i) for i in range(100, 500)]}    
    results = random_search(x, y, model, param_dist, scale, trials = iter_)          
    print('Iteration LogLoss: %.5f' % (results['score']))
    if results['score'] < score:
        return(clf, score)
    else:
        return(clf.set_params(**results['parameters']), results['score'])           
