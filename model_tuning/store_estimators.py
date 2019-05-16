import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))

from sklearn.pipeline import Pipeline
import json
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from model_tuning.meta_scoring_data import pull_val_data
import pandas as pd
from _connections import db_connection
from pop_psql import pg_query
from sklearn.metrics import log_loss

PSQL = db_connection('psql')


class FeatureSelector(object):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X):
        return X.loc[:,self.cols ] 
    def fit(self, X, y=None):
        return self


def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)


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
    
    
def cross_validate(x,y,est, verbose = False): 
#    x,y,est,scaler, only_scores, njobs, verbose = x,Y,model,scale, False, -1, True
    if len(y.unique()) == 2:
        splitter = StratifiedKFold(n_splits = 16, random_state = 86)
    else:
        splitter = KFold(n_splits = 16, random_state = 86)   

    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold)    
    jobs = []
    for train, test in all_folds:
        jobs.append([x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test], est])    
    cv_results = []
    for job in jobs:
        cv_results.append(_single_core_solver(job))
    results = pd.DataFrame()
    for df in cv_results:
        results = results.append(df)
    return(results)
    
    
domain = 'winner'
def pull_eval_data(domain):
#    
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', '%s_data_validation.csv' % (domain)))
#    pred_data_winner.set_index('bout_id', inplace = True)
#    pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
#    pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
#    pred_data_winner.drop('fight_date', axis = 1, inplace = True)
    pred_data_winner = pred_data_winner.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
    X = pred_data_winner[[i for i in list(pred_data_winner) if i != domain]]
    if domain == 'length':
        Y = pred_data_winner['length']/300
    elif domain == 'winner':
        Y = pred_data_winner[domain].apply(lambda x: x if x == 1 else 0)
    Y.sort_values()
    return(X, Y)
    

def pull_eval_test_data(domain):
#    
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', '%s_data_test.csv' % (domain)))
#    pred_data_winner.set_index('bout_id', inplace = True)
#    pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
#    pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
#    pred_data_winner.drop('fight_date', axis = 1, inplace = True)
    pred_data_winner = pred_data_winner.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
    X = pred_data_winner[[i for i in list(pred_data_winner) if i != domain]]
    if domain == 'length':
        Y = pred_data_winner['length']/300
    elif domain == 'winner':
        Y = pred_data_winner[domain].apply(lambda x: x if x == 1 else 0)
    Y.sort_values()
    return(X, Y)
    
    
def eval_estimators_2(domain):
    X, Y = pull_eval_data(domain)
    X_test, Y_test = pull_eval_test_data(domain)
    
    pred_df = pd.DataFrame(Y_test)

    final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'models')
    for mod_name in os.listdir(final_model_folder):
        if mod_name == '.DS_Store':
            continue
        model_path = os.listdir(os.path.join(final_model_folder, mod_name))
        model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
        model.fit(X,Y)
        
        mod_preds = model.predict_proba(X_test)
        
        mod_preds = pd.DataFrame([i[0] for i in mod_preds], X_test.index)
        mod_preds.rename(columns = {0: mod_name}, inplace = True)
        pred_df = pred_df.join(mod_preds)
    
    vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
    vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
    vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)

    pred_df = pred_df.join(vegas_preds).dropna()
    pred_df.to_csv(os.path.join(cur_path, 'model_test.csv'))

#    pred_df.drop('win_prob', axis = 1, inplace = True)
    list(pred_df)
    mod_scores = {}
    for mod in list(pred_df):
        if mod == 'winner':
            continue
        mod_scores[mod] = log_loss(pred_df['winner'], pred_df[mod])
        
       
    pred_cols = [i for i in list(pred_df) if i not in ['winner', 'VEGAS']]
    mod_errors = ()  
    for idx in pred_df.index:
        mod_errors[idx] = {}
        row = pred_df.loc[idx]
        for mod in pred_cols:
            if domain == 'winner':
                row_score = logloss(row[domain], row[mod])
            elif domain == 'length':
                row_score = abs(row[domain] - row[mod])
            mod_errors[idx][mod] = row_score 
    mod_errors = pd.DataFrame.from_dict(mod_errors).T
    
    
def eval_estimators(domain):
    X, Y = pull_eval_data(domain)
    pred_df = pd.DataFrame(Y)

    final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'models')
    for mod_name in os.listdir(final_model_folder):
        if mod_name == '.DS_Store':
            continue
        model_path = os.listdir(os.path.join(final_model_folder, mod_name))
        model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
        mod_preds = cross_validate(X,Y,model)
        mod_preds.rename(columns = {0: mod_name}, inplace = True)
        pred_df = pred_df.join(mod_preds)
    
    vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
    vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
    vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)

    pred_df = pred_df.join(vegas_preds).dropna()
    pred_df.to_csv(os.path.join(cur_path, 'model_preds.csv'))

#    pred_df.drop('win_prob', axis = 1, inplace = True)
    list(pred_df)
    mod_scores = {}
    for mod in list(pred_df):
        if mod == 'winner':
            continue
        mod_scores[mod] = log_loss(pred_df['winner'], pred_df[mod])
        
       
    pred_cols = [i for i in list(pred_df) if i not in ['winner', 'VEGAS']]
    mod_errors = ()  
    for idx in pred_df.index:
        mod_errors[idx] = {}
        row = pred_df.loc[idx]
        for mod in pred_cols:
            if domain == 'winner':
                row_score = logloss(row[domain], row[mod])
            elif domain == 'length':
                row_score = abs(row[domain] - row[mod])
            mod_errors[idx][mod] = row_score 
    mod_errors = pd.DataFrame.from_dict(mod_errors).T
    
    
    
def store_estimators(domain):
    X, Y = pull_test_data(domain)
    final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'models')
    for mod_name in os.listdir(final_model_folder):
        if mod_name == '.DS_Store':
            continue
        model_path = os.listdir(os.path.join(final_model_folder, mod_name))
        model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
        feats_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'features')
        with open(os.path.join(feats_folder, '%s.json' % (mod_name)), 'r') as fp:
            feats = json.load(fp)
            feats = feats[max(feats.keys())]
        feat_selector = FeatureSelector(feats)
        scale_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'scalers', mod_name)
        scale_path = os.path.join(scale_folder, os.listdir(os.path.join(scale_folder))[0])
        scale = load(scale_path)
        pipe = Pipeline([('feat_selection', feat_selector), ('scaler', scale), ('clf', model)])
        pipe.fit(X, Y)
        dump(pipe, os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'predictors', '%s.pkl' % (mod_name)))
    

    

#if __name__ == '__main__':
#    domain = 'winner'
#    store_estimators(domain)
#    store_error_estimators(domain)