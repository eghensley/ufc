import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))


import pandas as pd
from joblib import load
import json
from _connections import db_connection
from pop_psql import pg_query


pred_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'pred_data_TEST.csv'))
pred_data.drop('Unnamed: 0', axis = 1, inplace = True)
pred_data.set_index('bout_id', inplace = True)

hold_cols = ['fighter_id', 'fight_date', 'opponent_id']

meta = pred_data[hold_cols]
X = pred_data[[i for i in list(pred_data) if i not in hold_cols]]

PSQL = db_connection('psql')
bouts = pg_query(PSQL.client, "select b.bout_id, weight_desc from ufc.bouts b join ufc.fights f on f.fight_id = b.fight_id join ufc.weights w on b.weight_id = w.weight_id")
bouts.columns = ['bout_id', 'weight_id']
weights = pd.get_dummies(bouts['weight_id'])
weights['index'] = bouts['bout_id']
weights.drop_duplicates(inplace = True)
weights.set_index('index', inplace = True) 
meta_X = X.join(weights)

length_preds = {}
length_error_preds = {}
for mod in os.listdir(os.path.join(cur_path, 'fit_models', 'length')):
    length_model = load(os.path.join(cur_path, 'fit_models', 'length', mod))
    feats_folder = os.path.join(cur_path, 'modelling', 'length', 'final', 'features')
    with open(os.path.join(feats_folder, '%s.json' % (mod.split('.')[0])), 'r') as fp:
        feats = json.load(fp)
        feats = feats[max(feats.keys())]
    preds_length = length_model.predict(X[feats])
    error_model = load(os.path.join(cur_path, 'error_preds', 'length', '%s.pkl' % (mod.split('.')[0])))
    errors_length = error_model.predict(meta_X)

    for bout, predd, error in zip(X.index, preds_length, errors_length):        
        if bout not in length_preds.keys():
            length_preds[bout] = {}
        if bout not in length_error_preds.keys():
            length_error_preds[bout] = {}
        length_preds[bout][mod.split('.')[0]] = predd
        length_error_preds[bout][mod.split('.')[0]] = (error - .693) * -1

length_error_preds = pd.DataFrame.from_dict(length_error_preds).T
length_preds = pd.DataFrame.from_dict(length_preds).T

combined_length_preds = {}
for i in length_error_preds.index:
    sum_error = length_error_preds.loc[i].sum()
    adj_preds = []
    for mod in list(length_error_preds):
        adj_preds.append((length_error_preds.loc[i, mod]/sum_error) * length_preds.loc[i, mod])
    combined_length_preds[i] = sum(adj_preds)

combined_length_preds = pd.DataFrame.from_dict(combined_length_preds, orient= 'index')    
combined_length_preds.columns = ['predicted_length']

winner_preds = {}
winner_error_preds = {}
for mod in os.listdir(os.path.join(cur_path, 'fit_models', 'winner')):
    winner_model = load(os.path.join(cur_path, 'fit_models', 'winner', mod))
    feats_folder = os.path.join(cur_path, 'modelling', 'winner', 'final', 'features')
    with open(os.path.join(feats_folder, '%s.json' % (mod.split('.')[0])), 'r') as fp:
        feats = json.load(fp)
        feats = feats[max(feats.keys())]
    preds_winner = winner_model.predict_proba(X[feats])
    error_model = load(os.path.join(cur_path, 'error_preds', 'winner', '%s.pkl' % (mod.split('.')[0])))
    errors_winner = error_model.predict(meta_X)

    for bout, predd, error in zip(X.index, preds_winner, errors_winner):
        if bout not in winner_preds.keys():
            winner_preds[bout] = {}
        if bout not in winner_error_preds.keys():
            winner_error_preds[bout] = {}
        winner_preds[bout][mod.split('.')[0]] = predd[0]
        winner_error_preds[bout][mod.split('.')[0]] = (error - .693) * -1

winner_error_preds = pd.DataFrame.from_dict(winner_error_preds).T
winner_preds = pd.DataFrame.from_dict(winner_preds).T

combined_winner_preds = {}
for i in winner_error_preds.index:
    sum_error = winner_error_preds.loc[i].sum()
    adj_preds = []
    for mod in list(winner_error_preds):
        adj_preds.append((winner_error_preds.loc[i, mod]/sum_error) * winner_preds.loc[i, mod])
    combined_winner_preds[i] = sum(adj_preds)

combined_winner_preds = pd.DataFrame.from_dict(combined_winner_preds, orient= 'index')    
combined_winner_preds.columns = ['predicted_winner']


fighters = pg_query(PSQL.client, "select fighter_id, name from ufc.fighters")
fighters = {k:v for k,v in fighters.values}

predictions = meta.join(combined_winner_preds).join(combined_length_preds)

predictions['fighter_id'] = predictions['fighter_id'].apply(lambda x: fighters[x])
predictions['opponent_id'] = predictions['opponent_id'].apply(lambda x: fighters[x])

