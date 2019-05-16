import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))

import pandas as pd
from joblib import load
import numpy as np
from model_validation.store_estimators import FeatureSelector
from sklearn.metrics import log_loss, mean_absolute_error
from model_tuning.meta_scoring_data import logloss
from _connections import db_connection
from pop_psql import pg_query

def pull_test_data(domain):
#    PSQL = db_connection('psql')

    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', '%s_data_test.csv' % (domain)))
    
#    pred_data_winner.loc[pred_data_winner['bout_id'] == '0bc7e4852f75a06d']['fighter_id']
    pred_data_winner.set_index('bout_id', inplace = True)
    pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
    pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
    pred_data_winner.drop('fight_date', axis = 1, inplace = True)
    
    PRED_X = pred_data_winner[[i for i in list(pred_data_winner) if i != domain]]

#    bouts = pg_query(PSQL.client, "select b.bout_id, weight_desc from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id join ufc.weights w on b.weight_id = w.weight_id")
#    bouts.columns = ['bout_id', 'weight_id']
#    weights = pd.get_dummies(bouts['weight_id'])
#    weights['index'] = bouts['bout_id']
#    weights.drop_duplicates(inplace = True)
#    weights.set_index('index', inplace = True) 
#    META_X = PRED_X.join(weights)

    
    if domain == 'length':
        Y = pred_data_winner['length']/300
    elif domain == 'winner':
        Y = pred_data_winner[domain].apply(lambda x: x if x == 1 else 0)
    
    return(PRED_X, Y)
    
    
def pull_models(domain):
    all_models = {}
    for mod in os.listdir(os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'predictors')):
        if mod == '.DS_Store':
            continue
        
        if domain == 'winner' and mod == 'PolySVC.pkl':
            continue
        
        if domain == 'length' and mod in ['LinSVR.pkl', 'PolySVR.pkl', 'RbfSVR.pkl', 'LassoRegression.pkl', 'RFreg.pkl', 'LightGBR.pkl']:
            continue
        all_models[mod.split('.')[0]] = {}
        est = load(os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'predictors', mod))
        meta = load(os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'meta', mod))
        all_models[mod.split('.')[0]]['est'] = est
        all_models[mod.split('.')[0]]['meta'] = meta
    return(all_models)
    


def eval_models(domain):
    PRED_X, META_X, Y = pull_test_data(domain)
    all_models = pull_models(domain)
    results = {}
    for k,v in all_models.items():
        results[k] = {}
    
        if domain == 'winner':
            res_pred = v['est'].predict_proba(PRED_X)
        elif domain == 'length':
            res_pred = v['est'].predict(PRED_X)

    #    v['est'].predict(PRED_X)
        err_pred = v['meta'].predict(META_X)
        
        if domain == 'winner':
            results[k]['est'] = log_loss(Y, res_pred) * -1
        elif domain == 'length':
            results[k]['est'] = mean_absolute_error(Y, res_pred)
                    
        err_res = []
        for bout, row, act, err in zip(PRED_X.index, res_pred, Y.values, err_pred):
            if domain == 'winner':
                row_score = logloss(act, row[1])
            elif domain == 'length':
                row_score = abs(act - row)
            err_res.append(abs(err - row_score))
            
        results[k]['meta'] = np.mean(err_res) * -1
        
    results = pd.DataFrame.from_dict(results).T
    return(results)




def comb_preds(domain):

    PRED_X, META_X, Y = pull_test_data(domain)
    all_models = pull_models(domain)
    
    results = pd.DataFrame()
    for k,v in all_models.items():
    
        if domain == 'winner':
            res_pred = v['est'].predict_proba(PRED_X)
            res_pred = [i[1] for i in res_pred]
        elif domain == 'length':
            res_pred = v['est'].predict(PRED_X)

    #    v['est'].predict(PRED_X)
        err_pred = v['meta'].predict(META_X)
        
        mod_res = pd.DataFrame([res_pred, err_pred]).T
        mod_res.index = PRED_X.index
        mod_res.columns = ['%s_pred' % (k), '%s_err' % (k)]
        
        if len(results) == 0:
            results = mod_res
        else:
            results = results.join(mod_res)
                   
    pred_cols = [i for i in list(results) if 'pred' in i]
    err_cols = [i for i in list(results) if 'err' in i]
    
    predictions = results[pred_cols].mean(axis = 1)
    predicted_errors = results[err_cols].mean(axis = 1)

    return(predictions, predicted_errors)
    
    

def score_comb(domain):  
    
    predictions, predicted_errors = comb_preds(domain)

    PRED_X, META_X, Y = pull_test_data(domain)
    
    if domain == 'winner':
        score = log_loss(Y, predictions)
    elif domain == 'length':
        score = mean_absolute_error(Y, predictions)
        
    return(score)
    
    
if __name__ == '__main__':
    domain = 'winner'
    
    results = eval_models(domain)
    predictions, predicted_errors = comb_preds(domain)
    score = score_comb(domain)

