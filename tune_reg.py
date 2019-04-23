import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))

import imp
import pandas as pd
import numpy as np
from utils import _save_model, stage_init, test_solver, test_scaler,\
     init_feat_selection, feat_selection, C_parameter_tuning,\
     svc_hyper_parameter_tuning, lgb_find_lr, lgb_tree_params,\
     lgb_drop_lr
from sklearn.svm import SVR
import lightgbm as lgb
     
pred_data_length = pd.read_csv(os.path.join(cur_path, 'pred_data_length.csv'))
pred_data_length.drop('Unnamed: 0', inplace = True, axis = 1)
pred_data_length.set_index('bout_id', inplace = True)
pred_data_length.drop('fighter_id', axis = 1, inplace = True)
pred_data_length.drop('opponent_id', axis = 1, inplace = True)
pred_data_length.drop('fight_date', axis = 1, inplace = True)

X = pred_data_length[[i for i in list(pred_data_length) if i != 'length']]
Y = pred_data_length['length']

        
def tune_linsvr():
    name = 'LinSVR'
    dimension = 'length'
    
    stage, linsvr_reg, scale, features, linsvr_checkpoint_score = stage_init(name, dimension)
        
    if stage == 0:
        linsvr_reg = SVR(kernel = 'linear')
        linsvr_checkpoint_score = -np.inf
        features = init_feat_selection(X, Y, linsvr_reg)
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
        
    elif stage == 1: 
        linsvr_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvr_reg, linsvr_checkpoint_score, 10, -1, False)
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
            
    elif stage == 3:
        linsvr_reg, linsvr_checkpoint_score = C_parameter_tuning(X[features], Y, linsvr_reg, scale, linsvr_checkpoint_score)
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
                
    elif stage == 5:
        linsvr_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvr_reg, linsvr_checkpoint_score, 10, -1, False)
        _save_model(stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = True)
       
#        
#def tune_rbfsvr():
#    name = 'RbfSVR'
#    dimension = 'length'
#    
#    stage, rbfsvr_reg, scale, features, rbfsvr_checkpoint_score = stage_init(name, dimension)
#    
#    if stage == 0:
#        rbfsvc_clf = SVR(random_state = 1108, kernel = 'rbf')
#        rbfsvc_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvc_clf)
#        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf, X[features], Y) 
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
#        
#    elif stage == 1: 
#        rbfsvc_checkpoint_score, features = feat_selection(X[features], Y, scale, rbfsvc_clf, rbfsvc_checkpoint_score, 10, -1, False)
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
#    
#    elif stage == 2:
#        scale, linsvc_checkpoint_score = test_scaler(rbfsvc_clf, X[features], Y) 
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
#            
#    elif stage == 3:
#        rbfsvc_clf, rbfsvc_checkpoint_score = svc_hyper_parameter_tuning(X[features], Y, rbfsvc_clf, scale, rbfsvc_checkpoint_score)
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
#    
#    elif stage == 4:
#        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf, X[features], Y) 
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
#
#    elif stage == 5:
#        rbfsvc_checkpoint_score, features = feat_selection(X[features], Y, scale, rbfsvc_clf, rbfsvc_checkpoint_score, 10, -1, False)
#        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = True)
#                       

def tune_lgr():
    name = 'LightGBR'
    dimension = 'length'
    
    stage, lgb_reg, scale, features, lgbr_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        lgb_reg = lgb.LGBMRegressor(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1)
        lgbr_checkpoint_score = -np.inf    
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X, Y) 
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 3:
        lgb_reg, lgbr_checkpoint_score = lgb_find_lr(lgb_reg, X[features], Y, scale, lgbr_checkpoint_score) 
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 5: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 6: 
        lgb_reg, lgbr_checkpoint_score = lgb_tree_params(X[features], Y, lgb_reg, scale, lgbr_checkpoint_score, iter_ = 1000)
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 8: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 9: 
        lgb_reg, lgbr_checkpoint_score = lgb_drop_lr(lgb_reg, X[features], Y, scale, lgbr_checkpoint_score)
        _save_model(stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = True)


def tune_dartr():
    name = 'DartrGBM'
    dimension = 'length'
    
    stage, dart_reg, scale, features, dartr_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        dart_reg = lgb.LGBMRegressor(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1)
        dartr_checkpoint_score = -np.inf    
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X, Y) 
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 3:
        dart_reg, dartr_checkpoint_score = lgb_find_lr(dart_reg, X[features], Y, scale, dartr_checkpoint_score) 
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 5: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 6: 
        dart_reg, dartr_checkpoint_score = lgb_tree_params(X[features], Y, dart_reg, scale, dartr_checkpoint_score, iter_ = 1000)
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 8: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 9: 
        dart_reg, dartr_checkpoint_score = lgb_drop_lr(dart_reg, X[features], Y, scale, dartr_checkpoint_score)
        _save_model(stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = True)

if __name__ == '__main__':
    for i in range(6):
        tune_lgr()
        tune_linsvr()
#        tune_rbfsvc()
        tune_dartr()
        
 