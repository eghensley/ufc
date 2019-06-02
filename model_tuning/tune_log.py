import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))
sys.path.insert(3, cur_path)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import _save_model, stage_init, test_solver, test_scaler,\
     feat_selection, C_parameter_tuning,\
     svc_hyper_parameter_tuning, lgb_find_lr, lgb_tree_params,\
     lgb_drop_lr, forest_params, rf_trees, feat_selection_2,\
     knn_hyper_parameter_tuning, pipe_init, pca_tune
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier     
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier






EXTENSION = '_only_avg'
if EXTENSION == '':
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data_validation.csv'))
elif EXTENSION == '_only_avg':
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', 'only_avg', 'pred_data_winner_est_training.csv'))
pred_data_winner.set_index('bout_id', inplace = True)
pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
pred_data_winner.drop('fight_date', axis = 1, inplace = True)
#pred_data_winner = pred_data_winner.reset_index().sort_values('bout_id').set_index('bout_id')

X = pred_data_winner[[i for i in list(pred_data_winner) if i != 'winner']]
Y = pred_data_winner['winner'].apply(lambda x: x if x == 1 else 0)


def tune_log():
    name = 'LogRegression'
    dimension = 'winner'
    
    stage, log_clf, log_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        log_clf = LogisticRegression(max_iter = 1000, random_state = 1108, class_weight = 'balanced', solver = 'lbfgs')
        log_clf, log_checkpoint_score = pipe_init(X,Y,log_clf)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)
    
    elif stage == 1:
        log_clf, log_checkpoint_score = test_scaler(log_clf, log_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        log_clf, log_checkpoint_score = feat_selection(X, Y, log_clf, log_checkpoint_score)#, _iter = 5)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        log_clf, log_checkpoint_score = test_scaler(log_clf, log_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        log_clf, log_checkpoint_score = pca_tune(X,Y,log_clf, log_checkpoint_score)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        log_clf, log_checkpoint_score = feat_selection(X, Y, log_clf, log_checkpoint_score)#, _iter = 5)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        log_clf, log_checkpoint_score = test_scaler(log_clf, log_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)
        
    elif stage == 7:
        log_clf, log_checkpoint_score = C_parameter_tuning(X, Y, log_clf, log_checkpoint_score)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        log_clf, log_checkpoint_score = pca_tune(X,Y,log_clf, log_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        log_clf, log_checkpoint_score = feat_selection(X, Y, log_clf, log_checkpoint_score)#, _iter = 5)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        log_clf, log_checkpoint_score = test_scaler(log_clf, log_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)
        
    elif stage == 11:
        log_clf, log_checkpoint_score = test_solver(X, Y, log_clf, log_checkpoint_score) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)
    
    elif stage == 12:
        log_clf, log_checkpoint_score = pca_tune(X,Y,log_clf, log_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 13: 
        log_clf, log_checkpoint_score = feat_selection(X, Y, log_clf, log_checkpoint_score)#, _iter = 5)
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 14:
        log_clf, log_checkpoint_score = test_scaler(log_clf, log_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, log_clf, log_checkpoint_score, final = True, extension = EXTENSION)
        
        
def tune_linsvc():
    name = 'LinSVC'
    dimension = 'winner'
    
    stage, linsvc_clf, linsvc_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
        
    if stage == 0:
        linsvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'linear', probability = True)
        linsvc_clf, linsvc_checkpoint_score = pipe_init(X,Y,linsvc_clf)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)
    
    elif stage == 1:
        linsvc_clf, linsvc_checkpoint_score = test_scaler(linsvc_clf, linsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        linsvc_clf, linsvc_checkpoint_score = feat_selection(X, Y, linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        linsvc_clf, linsvc_checkpoint_score = test_scaler(linsvc_clf, linsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        linsvc_clf, linsvc_checkpoint_score = pca_tune(X,Y,linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        linsvc_clf, linsvc_checkpoint_score = feat_selection(X, Y, linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        linsvc_clf, linsvc_checkpoint_score = test_scaler(linsvc_clf, linsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)
                
    elif stage == 7:
        linsvc_clf, linsvc_checkpoint_score = C_parameter_tuning(X, Y, linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)
 
    elif stage == 8:
        linsvc_clf, linsvc_checkpoint_score = pca_tune(X,Y,linsvc_clf, linsvc_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        linsvc_clf, linsvc_checkpoint_score = feat_selection(X, Y, linsvc_clf, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        linsvc_clf, linsvc_checkpoint_score = test_scaler(linsvc_clf, linsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, linsvc_clf, linsvc_checkpoint_score, final = True, extension = EXTENSION)
        
        
def tune_rbfsvc():
    name = 'RbfSVC'
    dimension = 'winner'
    
    stage, rbfsvc_clf, rbfsvc_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        rbfsvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'rbf', probability = True)
        rbfsvc_clf, rbfsvc_checkpoint_score = pipe_init(X,Y,rbfsvc_clf)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        rbfsvc_clf, rbfsvc_checkpoint_score= test_scaler(rbfsvc_clf, rbfsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        rbfsvc_clf, rbfsvc_checkpoint_score = feat_selection_2(X, Y, rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        rbfsvc_clf, rbfsvc_checkpoint_score= test_scaler(rbfsvc_clf, rbfsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        rbfsvc_clf, rbfsvc_checkpoint_score = pca_tune(X,Y,rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        rbfsvc_clf, rbfsvc_checkpoint_score = feat_selection_2(X, Y, rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        rbfsvc_clf, rbfsvc_checkpoint_score= test_scaler(rbfsvc_clf, rbfsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)
                
    elif stage == 7:
        rbfsvc_clf, rbfsvc_checkpoint_score = svc_hyper_parameter_tuning(X, Y, rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        rbfsvc_clf, rbfsvc_checkpoint_score = pca_tune(X,Y,rbfsvc_clf, rbfsvc_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        rbfsvc_clf, rbfsvc_checkpoint_score = feat_selection_2(X, Y, rbfsvc_clf, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        rbfsvc_clf, rbfsvc_checkpoint_score= test_scaler(rbfsvc_clf, rbfsvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, rbfsvc_checkpoint_score, final = True, extension = EXTENSION)
                           

def tune_polysvc():
    name = 'PolySVC'
    dimension = 'winner'
    
    stage, polysvc_clf, polysvc_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        polysvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'poly', probability = True)
        polysvc_clf, polysvc_checkpoint_score = pipe_init(X,Y,polysvc_clf)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        polysvc_clf, polysvc_checkpoint_score = test_scaler(polysvc_clf, polysvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        polysvc_clf, polysvc_checkpoint_score = feat_selection_2(X, Y, polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        polysvc_clf, polysvc_checkpoint_score = test_scaler(polysvc_clf, polysvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        polysvc_clf, polysvc_checkpoint_score = pca_tune(X,Y,polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        polysvc_clf, polysvc_checkpoint_score = feat_selection_2(X, Y, polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        polysvc_clf, polysvc_checkpoint_score = test_scaler(polysvc_clf, polysvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)
                
    elif stage == 7:
        polysvc_clf, polysvc_checkpoint_score = svc_hyper_parameter_tuning(X, Y, polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        polysvc_clf, polysvc_checkpoint_score = pca_tune(X,Y,polysvc_clf, polysvc_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        polysvc_clf, polysvc_checkpoint_score = feat_selection_2(X, Y, polysvc_clf, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        polysvc_clf, polysvc_checkpoint_score = test_scaler(polysvc_clf, polysvc_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, polysvc_clf, polysvc_checkpoint_score, final = True, extension = EXTENSION)
             
        
def tune_lgb():
    name = 'LightGBM'
    dimension = 'winner'
    
    stage, lgb_clf, lgb_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        lgb_clf = lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True)
        lgb_clf, lgb_checkpoint_score = pipe_init(X,Y,lgb_clf)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        lgb_clf, lgb_checkpoint_score = test_scaler(lgb_clf, lgb_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        lgb_clf, lgb_checkpoint_score = feat_selection(X, Y, lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        lgb_clf, lgb_checkpoint_score = test_scaler(lgb_clf, lgb_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        lgb_clf, lgb_checkpoint_score = pca_tune(X,Y,lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        lgb_clf, lgb_checkpoint_score = feat_selection(X, Y, lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        lgb_clf, lgb_checkpoint_score = test_scaler(lgb_clf, lgb_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 7:
        lgb_clf, lgb_checkpoint_score = lgb_find_lr(lgb_clf, X, Y, lgb_checkpoint_score) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        lgb_clf, lgb_checkpoint_score = pca_tune(X,Y,lgb_clf, lgb_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        lgb_clf, lgb_checkpoint_score = feat_selection(X, Y, lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        lgb_clf, lgb_checkpoint_score = test_scaler(lgb_clf, lgb_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)
                                
    elif stage == 11: 
        lgb_clf, lgb_checkpoint_score = lgb_tree_params(X, Y, lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 12:
        lgb_clf, lgb_checkpoint_score = pca_tune(X,Y,lgb_clf, lgb_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 13: 
        lgb_clf, lgb_checkpoint_score = feat_selection(X, Y, lgb_clf, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 14:
        lgb_clf, lgb_checkpoint_score = test_scaler(lgb_clf, lgb_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = False, extension = EXTENSION)
                       
    elif stage == 15: 
        lgb_clf, lgb_checkpoint_score = lgb_drop_lr(lgb_clf, X, Y, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, lgb_checkpoint_score, final = True, extension = EXTENSION)


def tune_dart():
    name = 'DartGBM'
    dimension = 'winner'
    
    stage, dart_clf, dart_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        dart_clf = lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True)
        dart_clf, dart_checkpoint_score = pipe_init(X,Y,dart_clf)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        dart_clf, dart_checkpoint_score = test_scaler(dart_clf, dart_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        dart_clf, dart_checkpoint_score = feat_selection(X, Y, dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        dart_clf, dart_checkpoint_score = test_scaler(dart_clf, dart_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        dart_clf, dart_checkpoint_score = pca_tune(X,Y,dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        dart_clf, dart_checkpoint_score = feat_selection(X, Y, dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        dart_clf, dart_checkpoint_score = test_scaler(dart_clf, dart_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)
                
    elif stage == 7:
        dart_clf, dart_checkpoint_score = lgb_find_lr(dart_clf, X, Y, dart_checkpoint_score) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        dart_clf, dart_checkpoint_score = pca_tune(X,Y,dart_clf, dart_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        dart_clf, dart_checkpoint_score = feat_selection(X, Y, dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        dart_clf, dart_checkpoint_score = test_scaler(dart_clf, dart_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)
                                                
    elif stage == 11: 
        dart_clf, dart_checkpoint_score = lgb_tree_params(X, Y, dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 12:
        dart_clf, dart_checkpoint_score = pca_tune(X,Y,dart_clf, dart_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 13: 
        dart_clf, dart_checkpoint_score = feat_selection(X, Y, dart_clf, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 14:
        dart_clf, dart_checkpoint_score = test_scaler(dart_clf, dart_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = False, extension = EXTENSION)
                               
    elif stage == 15: 
        dart_clf, dart_checkpoint_score = lgb_drop_lr(dart_clf, X, Y, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, dart_checkpoint_score, final = True, extension = EXTENSION)


def tune_rf():
    name = 'RFclass'
    dimension = 'winner'
    
    stage, rf_clf, rf_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        rf_clf = RandomForestClassifier(random_state = 1108, n_estimators = 100)
        rf_clf, rf_checkpoint_score = pipe_init(X,Y,rf_clf)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        rf_clf, rf_checkpoint_score = test_scaler(rf_clf, rf_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        rf_clf, rf_checkpoint_score = feat_selection(X, Y, rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        rf_clf, rf_checkpoint_score = test_scaler(rf_clf, rf_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        rf_clf, rf_checkpoint_score = pca_tune(X,Y,rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 5: 
        rf_clf, rf_checkpoint_score = feat_selection(X, Y, rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6:
        rf_clf, rf_checkpoint_score = test_scaler(rf_clf, rf_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)
            
    elif stage == 7:
        rf_clf, rf_checkpoint_score = forest_params(X, Y, rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        rf_clf, rf_checkpoint_score = pca_tune(X,Y,rf_clf, rf_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 9: 
        rf_clf, rf_checkpoint_score = feat_selection(X, Y, rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 10:
        rf_clf, rf_checkpoint_score = test_scaler(rf_clf, rf_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 11:
        rf_clf, rf_checkpoint_score = rf_trees(X, Y, rf_clf, rf_checkpoint_score)
        _save_model(stage, 'winner', name, rf_clf, rf_checkpoint_score, final = True, extension = EXTENSION)


def tune_knn():
    name = 'KNN'
    dimension = 'winner'
    
    stage, knn_clf, knn_checkpoint_score = stage_init(name, dimension, extension = EXTENSION)
    
    if stage == 0:
        knn_clf = KNeighborsClassifier()
        knn_clf, knn_checkpoint_score = pipe_init(X,Y,knn_clf)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 1:
        knn_clf, knn_checkpoint_score = test_scaler(knn_clf, knn_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 2: 
        knn_clf, knn_checkpoint_score = feat_selection_2(X, Y, knn_clf, knn_checkpoint_score)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 3:
        knn_clf, knn_checkpoint_score = test_scaler(knn_clf, knn_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 4:
        knn_clf, knn_checkpoint_score = pca_tune(X,Y,knn_clf, knn_checkpoint_score)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)
    
    elif stage == 5:
        knn_clf, knn_checkpoint_score = knn_hyper_parameter_tuning(X, Y, knn_clf, knn_checkpoint_score)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 6: 
        knn_clf, knn_checkpoint_score = feat_selection_2(X, Y, knn_clf, knn_checkpoint_score)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 7:
        knn_clf, knn_checkpoint_score = test_scaler(knn_clf, knn_checkpoint_score, X, Y) 
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = False, extension = EXTENSION)

    elif stage == 8:
        knn_clf, knn_checkpoint_score = pca_tune(X,Y,knn_clf, knn_checkpoint_score, iter_ = 10)
        _save_model(stage, 'winner', name, knn_clf, knn_checkpoint_score, final = True, extension = EXTENSION)
        

if __name__ == '__main__':
    for i in range(20):
        tune_lgb()
#        tune_log()
#        tune_linsvc()
#        tune_rbfsvc()
        tune_dart()
#        tune_rf()
        tune_polysvc()
#        tune_knn()

