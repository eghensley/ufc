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
from sklearn.linear_model import LogisticRegression
from utils import _save_model, stage_init, test_solver, test_scaler,\
     init_feat_selection, feat_selection, C_parameter_tuning,\
     svc_hyper_parameter_tuning, lgb_find_lr, lgb_tree_params,\
     lgb_drop_lr, forest_params, rf_trees, feat_selection_2,\
     knn_hyper_parameter_tuning
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier     
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

pred_data_winner = pd.read_csv(os.path.join(cur_path, 'pred_data_winner.csv'))
pred_data_winner.drop('Unnamed: 0', inplace = True, axis = 1)
pred_data_winner.set_index('bout_id', inplace = True)
pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
pred_data_winner.drop('fight_date', axis = 1, inplace = True)

X = pred_data_winner[[i for i in list(pred_data_winner) if i != 'winner']]
Y = pred_data_winner['winner'].apply(lambda x: x if x == 1 else 0)


def tune_log():
    name = 'LogRegression'
    dimension = 'winner'
    
    stage, log_clf, scale, features, log_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        log_clf = LogisticRegression(max_iter = 1000, random_state = 1108, class_weight = 'balanced')
        log_checkpoint_score = -np.inf
        scale, log_checkpoint_score = test_scaler(log_clf, X, Y) 
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, list(X), final = False)
    
    elif stage == 1: 
        log_checkpoint_score, features = feat_selection(X[features], Y, scale, log_clf, log_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = False)
    
    elif stage == 2:
        log_clf, log_checkpoint_score, scale = test_solver(X[features], Y, log_clf, scale, log_checkpoint_score) 
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = False)
    
    elif stage == 3: 
        log_checkpoint_score, features = feat_selection(X[features], Y, scale, log_clf, log_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = False)

    elif stage == 4:
        log_clf, log_checkpoint_score = C_parameter_tuning(X[features], Y, log_clf, scale, log_checkpoint_score)
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = False)

    elif stage == 5: 
        log_checkpoint_score, features = feat_selection(X[features], Y, scale, log_clf, log_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = False)
    
    elif stage == 6:
        log_clf, log_checkpoint_score, scale = test_solver(X[features], Y, log_clf, scale, log_checkpoint_score) 
        _save_model(stage, 'winner', name, log_clf, scale, log_checkpoint_score, features, final = True)
    
        
def tune_linsvc():
    name = 'LinSVC'
    dimension = 'winner'
    
    stage, linsvc_clf, scale, features, linsvc_checkpoint_score = stage_init(name, dimension)
        
    if stage == 0:
        linsvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'linear', probability = True)
        linsvc_checkpoint_score = -np.inf
        features = init_feat_selection(X, Y, linsvc_clf)
        scale, linsvc_checkpoint_score = test_scaler(linsvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = False)
        
    elif stage == 1: 
        linsvc_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvc_clf, linsvc_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, linsvc_checkpoint_score = test_scaler(linsvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = False)
            
    elif stage == 3:
        linsvc_clf, linsvc_checkpoint_score = C_parameter_tuning(X[features], Y, linsvc_clf, scale, linsvc_checkpoint_score)
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, linsvc_checkpoint_score = test_scaler(linsvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = False)
                
    elif stage == 5:
        linsvc_checkpoint_score, features = feat_selection(X[features], Y, scale, linsvc_clf, linsvc_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, linsvc_clf, scale, linsvc_checkpoint_score, features, final = True)
       
        
def tune_rbfsvc():
    name = 'RbfSVC'
    dimension = 'winner'
    
    stage, rbfsvc_clf, scale, features, rbfsvc_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        rbfsvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'rbf', probability = True)
        rbfsvc_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvc_clf)
        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf, X, Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, list(X), final = False)
        
    elif stage == 1: 
        rbfsvc_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rbfsvc_clf, rbfsvc_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, linsvc_checkpoint_score = test_scaler(rbfsvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
            
    elif stage == 3:
        rbfsvc_clf, rbfsvc_checkpoint_score = svc_hyper_parameter_tuning(X[features], Y, rbfsvc_clf, scale, rbfsvc_checkpoint_score)
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, rbfsvc_checkpoint_score = test_scaler(rbfsvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = False)

    elif stage == 5:
        rbfsvc_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rbfsvc_clf, rbfsvc_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, rbfsvc_clf, scale, rbfsvc_checkpoint_score, features, final = True)
                       

def tune_polysvc():
    name = 'PolySVC'
    dimension = 'winner'
    
    stage, polysvc_clf, scale, features, polysvc_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        polysvc_clf = SVC(random_state = 1108, class_weight = 'balanced', kernel = 'poly', probability = True)
        polysvc_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvc_clf)
        scale, polysvc_checkpoint_score = test_scaler(polysvc_clf, X, Y) 
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, list(X), final = False)
        
    elif stage == 1: 
        polysvc_checkpoint_score, features = feat_selection_2(X[features], Y, scale, polysvc_clf, polysvc_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, polysvc_checkpoint_score = test_scaler(polysvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, features, final = False)
            
    elif stage == 3:
        polysvc_clf, polysvc_checkpoint_score = svc_hyper_parameter_tuning(X[features], Y, polysvc_clf, scale, polysvc_checkpoint_score)
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, polysvc_checkpoint_score = test_scaler(polysvc_clf, X[features], Y) 
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, features, final = False)

    elif stage == 5:
        polysvc_checkpoint_score, features = feat_selection_2(X[features], Y, scale, polysvc_clf, polysvc_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, polysvc_clf, scale, polysvc_checkpoint_score, features, final = True)
             
        
def tune_lgb():
    name = 'LightGBM'
    dimension = 'winner'
    
    stage, lgb_clf, scale, features, lgb_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        lgb_clf = lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True)
        lgb_checkpoint_score = -np.inf    
        scale, lgb_checkpoint_score = test_scaler(lgb_clf, X, Y) 
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        lgb_checkpoint_score, features = feat_selection_2(X[features], Y, scale, lgb_clf, lgb_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, lgb_checkpoint_score = test_scaler(lgb_clf, X[features], Y) 
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 3:
        lgb_clf, lgb_checkpoint_score = lgb_find_lr(lgb_clf, X[features], Y, scale, lgb_checkpoint_score) 
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, lgb_checkpoint_score = test_scaler(lgb_clf, X[features], Y) 
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 5: 
        lgb_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_clf, lgb_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 6: 
        lgb_clf, lgb_checkpoint_score = lgb_tree_params(X[features], Y, lgb_clf, scale, lgb_checkpoint_score, iter_ = 1000)
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, lgb_checkpoint_score = test_scaler(lgb_clf, X[features], Y) 
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 8: 
        lgb_checkpoint_score, features = feat_selection(X[features], Y, scale, lgb_clf, lgb_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = False)

    elif stage == 9: 
        lgb_clf, lgb_checkpoint_score = lgb_drop_lr(lgb_clf, X[features], Y, scale, lgb_checkpoint_score)
        _save_model(stage, 'winner', name, lgb_clf, scale, lgb_checkpoint_score, features, final = True)


def tune_dart():
    name = 'DartGBM'
    dimension = 'winner'
    
    stage, dart_clf, scale, features, dart_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        dart_clf = lgb.LGBMClassifier(random_state = 1108, n_estimators = 100, subsample = .8, verbose=-1, is_unbalance = True)
        dart_checkpoint_score = -np.inf    
        scale, dart_checkpoint_score = test_scaler(dart_clf, X, Y) 
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        dart_checkpoint_score, features = feat_selection_2(X[features], Y, scale, dart_clf, dart_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, dart_checkpoint_score = test_scaler(dart_clf, X[features], Y) 
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 3:
        dart_clf, dart_checkpoint_score = lgb_find_lr(dart_clf, X[features], Y, scale, dart_checkpoint_score) 
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)
                                
    elif stage == 4:
        scale, dart_checkpoint_score = test_scaler(dart_clf, X[features], Y) 
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 5: 
        dart_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_clf, dart_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 6: 
        dart_clf, dart_checkpoint_score = lgb_tree_params(X[features], Y, dart_clf, scale, dart_checkpoint_score, iter_ = 1000)
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 7:
        scale, dart_checkpoint_score = test_scaler(dart_clf, X[features], Y) 
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 8: 
        dart_checkpoint_score, features = feat_selection(X[features], Y, scale, dart_clf, dart_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = False)

    elif stage == 9: 
        dart_clf, dart_checkpoint_score = lgb_drop_lr(dart_clf, X[features], Y, scale, dart_checkpoint_score)
        _save_model(stage, 'winner', name, dart_clf, scale, dart_checkpoint_score, features, final = True)



def tune_rf():
    name = 'RFclass'
    dimension = 'winner'
    
    stage, rf_clf, scale, features, rf_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        rf_clf = RandomForestClassifier(random_state = 1108, n_estimators = 100)
        rf_checkpoint_score = -np.inf    
        scale, rf_checkpoint_score = test_scaler(rf_clf, X, Y) 
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, list(X), final = False)

    elif stage == 1: 
        rf_checkpoint_score, features = feat_selection_2(X[features], Y, scale, rf_clf, rf_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, features, final = False)

    elif stage == 2:
        scale, rf_checkpoint_score = test_scaler(rf_clf, X[features], Y) 
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, features, final = False)
          
    elif stage == 3: 
        rf_clf, rf_checkpoint_score = forest_params(X[features], Y, rf_clf, scale, rf_checkpoint_score, iter_ = 1000)
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, features, final = False)

    elif stage == 4:
        scale, rf_checkpoint_score = test_scaler(rf_clf, X[features], Y) 
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, features, final = False)

    elif stage == 5: 
        rf_checkpoint_score, features = feat_selection(X[features], Y, scale, rf_clf, rf_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, rf_clf, scale, rf_checkpoint_score, features, final = False)

    elif stage == 6:
        rf_clf, rf_checkpoint_score = rf_trees(X, Y, scale, rf_clf, rf_checkpoint_score)
        _save_model(stage, dimension, name, rf_clf, scale, rf_checkpoint_score, features, final = True)
        


def tune_knn():
    name = 'KNN'
    dimension = 'winner'
    
    stage, knn_clf, scale, features, knn_checkpoint_score = stage_init(name, dimension)
    
    if stage == 0:
        knn_clf = KNeighborsClassifier()
        knn_checkpoint_score = -np.inf
#        features = init_feat_selection(X, Y, rbfsvc_clf)
        scale, knn_checkpoint_score = test_scaler(knn_clf, X, Y) 
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, list(X), final = False)
        
    elif stage == 1: 
        knn_checkpoint_score, features = feat_selection_2(X[features], Y, scale, knn_clf, knn_checkpoint_score, 24, -1, False)
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, features, final = False)
    
    elif stage == 2:
        scale, knn_checkpoint_score = test_scaler(knn_clf, X[features], Y) 
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, features, final = False)
            
    elif stage == 3:
        knn_clf, knn_checkpoint_score = knn_hyper_parameter_tuning(X[features], Y, knn_clf, scale, knn_checkpoint_score)
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, features, final = False)
    
    elif stage == 4:
        scale, knn_checkpoint_score = test_scaler(knn_clf, X[features], Y) 
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, features, final = False)

    elif stage == 5:
        knn_checkpoint_score, features = feat_selection_2(X[features], Y, scale, knn_clf, knn_checkpoint_score, 10, -1, False)
        _save_model(stage, 'winner', name, knn_clf, scale, knn_checkpoint_score, features, final = True)


     
if __name__ == '__main__':
    for i in range(10):
        tune_lgb()
        tune_log()
        tune_linsvc()
        tune_rbfsvc()
        tune_dart()
        tune_rf()
        tune_polysvc()
        tune_knn()

