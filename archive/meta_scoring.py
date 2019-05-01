import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))


import requests
from lxml import html
from _connections import db_connection
from pg_tables import create_tables
import pandas as pd
from datetime import datetime
from pop_psql import pg_query
import random
from copy import deepcopy
import numpy as np
from sklearn.externals.joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import json
from joblib import dump, load
import lightgbm
from utils import cross_validate
from sklearn.metrics import log_loss
from sklearn.utils import class_weight
from pop_psql import pg_query
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor



class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)

pred_data_winner = pd.read_csv(os.path.join(cur_path, 'test_data', 'pred_data_winner.csv'))
pred_data_winner.drop('Unnamed: 0', axis = 1, inplace = True)
pred_data_winner.set_index('bout_id', inplace = True)
pred_data_winner.drop('fighter_id', axis = 1, inplace = True)
pred_data_winner.drop('opponent_id', axis = 1, inplace = True)
pred_data_winner.drop('fight_date', axis = 1, inplace = True)

X = pred_data_winner[[i for i in list(pred_data_winner) if i != 'winner']]
Y = pred_data_winner['winner'].apply(lambda x: x if x == 1 else 0)


pred_df = pd.DataFrame(Y)
res_df = pd.DataFrame()
model_weights = {0: {}}
final_model_folder = os.path.join(cur_path, 'modelling', 'winner', 'final', 'models')
for mod_name in os.listdir(final_model_folder):
    if mod_name == '.DS_Store':
        continue
    model_path = os.listdir(os.path.join(final_model_folder, mod_name))
    model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
    feats_folder = os.path.join(cur_path, 'modelling', 'winner', 'final', 'features')
    with open(os.path.join(feats_folder, '%s.json' % (mod_name)), 'r') as fp:
        feats = json.load(fp)
        feats = feats[max(feats.keys())]
    result_folder = os.path.join(cur_path, 'modelling', 'winner', 'final', 'results')
    with open(os.path.join(result_folder, '%s.json' % (mod_name)), 'r') as fp:
        res = json.load(fp)
        res = res[max(res.keys())]
    model_weights[0][mod_name] = res
    feat_selector = FeatureSelector(feats)
    scale_folder = os.path.join(cur_path, 'modelling', 'winner', 'final', 'scalers', mod_name)
    scale_path = os.path.join(scale_folder, os.listdir(os.path.join(scale_folder))[0])
    scale = load(scale_path)
    mod_preds = cross_validate(X[feats],Y,model,scale, only_scores = False, njobs = 1)
    mod_preds.rename(columns = {0: mod_name}, inplace = True)
    pred_df = pred_df.join(mod_preds)
#    pipe = Pipeline([('feature_selection', feat_selector), ('scaler', scale), ('clf', model)])
    
pred_cols = [i for i in list(pred_df) if i != 'winner']

mod_scores = {}
for idx in pred_df.index:
    mod_scores[idx] = {}
    row = pred_df.loc[idx]
    for mod in pred_cols:
        row_score = logloss(row['winner'], row[mod])
        mod_scores[idx][mod] = row_score 
mod_scores = pd.DataFrame.from_dict(mod_scores).T
    

meta_data = mod_scores.join(X)
for col in [[i for i in list(meta_data) if i not in pred_cols]]:
    meta_data[col] = StandardScaler().fit_transform(meta_data[col])


PSQL = db_connection('psql')
bouts = pg_query(PSQL.client, "select b.bout_id, weight_desc from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id join ufc.weights w on b.weight_id = w.weight_id")
bouts.columns = ['bout_id', 'weight_id']
weights = pd.get_dummies(bouts['weight_id'])
weights['index'] = bouts['bout_id']
weights.drop_duplicates(inplace = True)
weights.set_index('index', inplace = True) 
meta_data = meta_data.join(weights)
meta_data.to_csv(os.path.join(cur_path, 'test_data', 'pred_res_winner.csv'))



adsfasfda





model_feats = {}
for mod in pred_cols:
    
    
    
    meta_y = (meta_data[mod] - 0.693) * -1
    meta_x = meta_data[[i for i in list(meta_data) if i not in pred_cols]]
        
    forest = ExtraTreesRegressor(n_estimators=250,
                                  random_state=0)
    
    forest.fit(meta_x, meta_y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    pd.DataFrame.from_dict({k:v for k,v in zip(list(meta_x), importances)}, orient = 'index').sort_values(0)


    meta = meta_data[[i for i in list(meta_data) if i not in pred_cols or i == mod]]
    cor = meta.corr()[mod]
    cor.loc[abs(cor) > .1]
    
    
    
    abs(cor).sort_values()
    
    print('%s result features' % (mod))
    meta_y = (meta_data[mod] - 0.693) * -1
    meta_x = meta_data[[i for i in list(meta_data) if i not in pred_cols]]
    
    
    
    
    clf = LassoCV(cv=5, random_state = 1108)
    sfm = SelectFromModel(clf)
    sfm.fit(meta_x, meta_y)
    sfm.get_support()

    selector = RFECV(LinearRegression(), cv = 8, scoring = 'neg_mean_absolute_error')
    selector.fit(meta_x, meta_y)
    model_feats[mod] = [i for i,j in zip(list(meta_x), selector.support_ ) if j]
    print(len(model_feats[mod]))
    
    
    
#    
#    
#weights = {}
#for 
#pred_df['comb_win'] = pred_df[pred_cols].mean(axis = 1)
#    
#test_weights = class_weight.compute_class_weight('balanced',
#                            np.unique(pred_df['winner']),pred_df['winner'])    
#test_weights_dict = {i:j for i,j in zip(np.unique(pred_df['winner']), test_weights)}    
#   
#model_weights[0]['comb'] = log_loss(pred_df['winner'], pred_df['comb_win'].values, sample_weight = [test_weights_dict[i] for i in pred_df['winner'].values]) * -1
