import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))


from _connections import db_connection
import pandas as pd
from pop_psql import pg_query
import json
from joblib import load
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

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
    
    
def cross_validate(x,y,est,scaler, verbose = False): 
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
        jobs.append([scaler.fit_transform(x.iloc[train]), scaler.fit_transform(x.iloc[test]), y.iloc[train], y.iloc[test], est])    
    cv_results = []
    for job in jobs:
        cv_results.append(_single_core_solver(job))
    results = pd.DataFrame()
    for df in cv_results:
        results = results.append(df)
    return(results)



def pull_val_data(domain):
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', '%s_data_validation.csv' % (domain)))
    pred_data_winner.set_index('bout_id', inplace = True)
    pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
    pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
    pred_data_winner.drop('fight_date', axis = 1, inplace = True)
    
    X = pred_data_winner[[i for i in list(pred_data_winner) if i != domain]]
    
    if domain == 'length':
        Y = pred_data_winner['length']/300
    elif domain == 'winner':
        Y = pred_data_winner[domain].apply(lambda x: x if x == 1 else 0)
    
    return(X, Y)


def store_meta_res(domain):
#    domain = 'length'
    X, Y = pull_val_data(domain)
    
    pred_df = pd.DataFrame(Y)
#    res_df = pd.DataFrame()
    final_model_folder = os.path.join(cur_path, 'modelling', domain, 'final', 'models')
    for mod_name in os.listdir(final_model_folder):
        if mod_name == '.DS_Store':
            continue
        model_path = os.listdir(os.path.join(final_model_folder, mod_name))
        model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
        feats_folder = os.path.join(cur_path, 'modelling', domain, 'final', 'features')
        with open(os.path.join(feats_folder, '%s.json' % (mod_name)), 'r') as fp:
            feats = json.load(fp)
            feats = feats[max(feats.keys())]
        scale_folder = os.path.join(cur_path, 'modelling', domain, 'final', 'scalers', mod_name)
        scale_path = os.path.join(scale_folder, os.listdir(os.path.join(scale_folder))[0])
        scale = load(scale_path)
        mod_preds = cross_validate(X[feats],Y,model,scale)
        mod_preds.rename(columns = {0: mod_name}, inplace = True)
        pred_df = pred_df.join(mod_preds)
        
    pred_cols = [i for i in list(pred_df) if i != domain]
    
    mod_scores = {}
    for idx in pred_df.index:
        mod_scores[idx] = {}
        row = pred_df.loc[idx]
        for mod in pred_cols:
            if domain == 'winner':
                row_score = logloss(row[domain], row[mod])
            elif domain == 'length':
                row_score = abs(row[domain] - row[mod])
            mod_scores[idx][mod] = row_score 
    mod_scores = pd.DataFrame.from_dict(mod_scores).T
        
    
    meta_data = mod_scores.join(X)
    PSQL = db_connection('psql')
    bouts = pg_query(PSQL.client, "select b.bout_id, weight_desc from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id join ufc.weights w on b.weight_id = w.weight_id")
    bouts.columns = ['bout_id', 'weight_id']
    weights = pd.get_dummies(bouts['weight_id'])
    weights['index'] = bouts['bout_id']
    weights.drop_duplicates(inplace = True)
    weights.set_index('index', inplace = True) 
    meta_data = meta_data.join(weights)
    meta_data.to_csv(os.path.join(cur_path, 'data', 'meta', 'meta_%s.csv' % (domain)))