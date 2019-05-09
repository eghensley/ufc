import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))

#import imp
import pandas as pd
import numpy as np
from utils import _save_meta_model, stage_meta_init, test_scaler,\
     init_feat_selection, feat_selection, C_parameter_tuning,\
     svc_hyper_parameter_tuning, lgb_find_lr, lgb_tree_params,\
     lgb_drop_lr, alpha_parameter_tuning, forest_params, rf_trees,\
     feat_selection_2, poly_hyper_parameter_tuning
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

pred_cols = ['KNN',
             'LogRegression',
             'LinSVC',
             'RFclass',
             'RbfSVC',
             'PolySVC',
             'DartGBM',
             'LightGBM']
winner_res_data = pd.read_csv(os.path.join(cur_path, 'data', 'meta', 'meta_winner.csv'))
winner_res_data.set_index('Unnamed: 0', inplace = True)
X = winner_res_data[[i for i in list(winner_res_data) if i not in pred_cols]]


        
def tune_linsvr():
    name = 'LinSVR'
    dimension = 'winner'
  
    stage, linsvr_reg, scale, features, linsvr_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
        
    if stage == 0:
        linsvr_reg = SVR(kernel = 'linear')
        linsvr_checkpoint_score = -np.inf
        features = init_feat_selection(X, Y, linsvr_reg)
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
        
    elif stage == 1: 
        linsvr_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvr_reg, linsvr_checkpoint_score, 10, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
            
    elif stage == 3:
        linsvr_reg, linsvr_checkpoint_score = C_parameter_tuning(X[features], Y, linsvr_reg, scale, linsvr_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, linsvr_checkpoint_score = test_scaler(linsvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = False)
                
    elif stage == 5:
        linsvr_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvr_reg, linsvr_checkpoint_score, 10, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, linsvr_reg, scale, linsvr_checkpoint_score, features, final = True)
       

def tune_rbfsvr():
    name = 'RbfSVR'
    dimension = 'winner'
    
    stage, rbfsvr_reg, scale, features, rbfsvr_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        rbfsvr_reg = SVR(kernel = 'rbf')
        rbfsvr_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvr_reg)
        scale, rbfsvr_checkpoint_score = test_scaler(rbfsvr_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, list(X), final = False)
        
    elif stage == 1: 
        rbfsvr_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rbfsvr_reg, rbfsvr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, linsvc_checkpoint_score = test_scaler(rbfsvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, features, final = False)
            
    elif stage == 3:
        rbfsvr_reg, rbfsvr_checkpoint_score = svc_hyper_parameter_tuning(X[features], Y, rbfsvr_reg, scale, rbfsvr_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, rbfsvr_checkpoint_score = test_scaler(rbfsvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, features, final = False)

    elif stage == 5:
        rbfsvr_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rbfsvr_reg, rbfsvr_checkpoint_score, 10, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, rbfsvr_reg, scale, rbfsvr_checkpoint_score, features, final = True)
                       
      

def tune_polysvr():
    name = 'PolySVR'
    dimension = 'winner'
    
    stage, polysvr_reg, scale, features, polysvr_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        polysvr_reg = SVR(kernel = 'poly')
        polysvr_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvc_clf)
        scale, polysvr_checkpoint_score = test_scaler(polysvr_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, list(X), final = False)
        
    elif stage == 1: 
        polysvr_checkpoint_score, features = feat_selection_2(X[features], Y, scale, polysvr_reg, polysvr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, polysvr_checkpoint_score = test_scaler(polysvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, features, final = False)
            
    elif stage == 3:
        polysvr_reg, polysvr_checkpoint_score = poly_hyper_parameter_tuning(X[features], Y, polysvr_reg, scale, polysvr_checkpoint_score, iter_ = 50)
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, polysvr_checkpoint_score = test_scaler(polysvr_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, features, final = False)

    elif stage == 5:
        polysvr_checkpoint_score, features = feat_selection_2(X[features], Y, scale, polysvr_reg, polysvr_checkpoint_score, 10, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, polysvr_reg, scale, polysvr_checkpoint_score, features, final = True)


def tune_lasso():
    name = 'LassoRegression'
    dimension = 'winner'
    
    stage, lasso_reg, scale, features, lasso_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        lasso_reg = Lasso(max_iter = 1000, random_state = 1108)
        lasso_checkpoint_score = -np.inf
        scale, lasso_checkpoint_score = test_scaler(lasso_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, list(X), final = False)
    
    elif stage == 1: 
        lasso_checkpoint_score, features = feat_selection(X[features], Y, scale, lasso_reg, lasso_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, features, final = False)
 
    elif stage == 2:
        scale, lasso_checkpoint_score = test_scaler(lasso_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, features, final = False)

    elif stage == 3:
        lasso_reg, lasso_checkpoint_score = alpha_parameter_tuning(X[features], Y, lasso_reg, scale, lasso_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, features, final = False)

    elif stage == 4: 
        lasso_checkpoint_score, features = feat_selection(X[features], Y, scale, lasso_reg, lasso_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, features, final = False)
    
    elif stage == 5:
        scale, lasso_checkpoint_score = test_scaler(lasso_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lasso_reg, scale, lasso_checkpoint_score, features, final = True)



def tune_rf():
    name = 'RFreg'
    dimension = 'winner'
    
    stage, rf_reg, scale, features, rf_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        rf_reg = RandomForestRegressor(random_state = 1108, n_estimators = 100)
        rf_checkpoint_score = -np.inf    
        scale, rf_checkpoint_score = test_scaler(rf_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        rf_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rf_reg, rf_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, rf_checkpoint_score = test_scaler(rf_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = False)
          
    elif stage == 3: 
        rf_reg, rf_checkpoint_score = forest_params(X[features], Y, rf_reg, scale, rf_checkpoint_score, iter_ = 1000)
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = False)

    elif stage == 4:
        scale, rf_checkpoint_score = test_scaler(rf_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = False)

    elif stage == 5: 
        rf_checkpoint_score, features = feat_selection(X[features], Y, scale, rf_reg, rf_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = False)
     
    elif stage == 6:
        rf_reg, rf_checkpoint_score = rf_trees(X, Y, scale, rf_reg, rf_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, rf_reg, scale, rf_checkpoint_score, features, final = True)
        
        
def tune_lgr():
    name = 'LightGBR'
    dimension = 'winner'
    
    stage, lgb_reg, scale, features, lgbr_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        lgb_reg = lgb.LGBMRegressor(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1)
        lgbr_checkpoint_score = -np.inf    
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 3:
        lgb_reg, lgbr_checkpoint_score = lgb_find_lr(lgb_reg, X[features], Y, scale, lgbr_checkpoint_score) 
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 5: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 6: 
        lgb_reg, lgbr_checkpoint_score = lgb_tree_params(X[features], Y, lgb_reg, scale, lgbr_checkpoint_score, iter_ = 1000)
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, lgbr_checkpoint_score = test_scaler(lgb_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 8: 
        lgbr_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_reg, lgbr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = False)

    elif stage == 9: 
        lgb_reg, lgbr_checkpoint_score = lgb_drop_lr(lgb_reg, X[features], Y, scale, lgbr_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, lgb_reg, scale, lgbr_checkpoint_score, features, final = True)


def tune_dartr():
    name = 'DartrGBM'
    dimension = 'winner'
    
    stage, dart_reg, scale, features, dartr_checkpoint_score = stage_meta_init(meta_dimension, name, dimension)
    
    if stage == 0:
        dart_reg = lgb.LGBMRegressor(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1)
        dartr_checkpoint_score = -np.inf    
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X, Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 3:
        dart_reg, dartr_checkpoint_score = lgb_find_lr(dart_reg, X[features], Y, scale, dartr_checkpoint_score) 
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 5: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 6: 
        dart_reg, dartr_checkpoint_score = lgb_tree_params(X[features], Y, dart_reg, scale, dartr_checkpoint_score, iter_ = 1000)
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, dartr_checkpoint_score = test_scaler(dart_reg, X[features], Y) 
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 8: 
        dartr_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_reg, dartr_checkpoint_score, 24, -1, False)
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = False)

    elif stage == 9: 
        dart_reg, dartr_checkpoint_score = lgb_drop_lr(dart_reg, X[features], Y, scale, dartr_checkpoint_score)
        _save_meta_model(meta_dimension, stage, dimension, name, dart_reg, scale, dartr_checkpoint_score, features, final = True)

if __name__ == '__main__':
    for meta_dimension in pred_cols:
        Y = winner_res_data[meta_dimension]
        for i in range(10):
#            tune_lgr()
#            tune_linsvr()
#            tune_dartr()
            tune_lasso()
#            tune_rf()
#            tune_rbfsvr()
#            tune_polysvr()
