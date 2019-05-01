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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import json
from joblib import dump, load

#class FeatureSelector( BaseEstimator, TransformerMixin ):
#    #Class Constructor 
#    def __init__( self, feature_names ):
#        self._feature_names = feature_names 
#    
#    #Return self nothing else to do here    
#    def fit( self, X, y = None ):
#        return self 
#    
#    #Method that describes what we need this transformer to do
#    def transform( self, X, y = None ):
#        return X[ self._feature_names ] 

pred_data_length = pd.read_csv(os.path.join(cur_path, 'test_data', 'pred_data_length.csv'))
pred_data_length.drop('Unnamed: 0', axis = 1, inplace = True)
pred_data_length.set_index('bout_id', inplace = True)
pred_data_length.drop('fighter_id', axis = 1, inplace = True)
pred_data_length.drop('opponent_id', axis = 1, inplace = True)
pred_data_length.drop('fight_date', axis = 1, inplace = True)

X = pred_data_length[[i for i in list(pred_data_length) if i != 'length']]
Y = pred_data_length['length']/300

final_model_folder = os.path.join(cur_path, 'modelling', 'length', 'final', 'models')
for mod_name in os.listdir(final_model_folder):
    if mod_name == '.DS_Store':
        continue
    model_path = os.listdir(os.path.join(final_model_folder, mod_name))
    model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
    feats_folder = os.path.join(cur_path, 'modelling', 'length', 'final', 'features')
    with open(os.path.join(feats_folder, '%s.json' % (mod_name)), 'r') as fp:
        feats = json.load(fp)
        feats = feats[max(feats.keys())]
    result_folder = os.path.join(cur_path, 'modelling', 'length', 'final', 'results')
    with open(os.path.join(result_folder, '%s.json' % (mod_name)), 'r') as fp:
        res = json.load(fp)
        res = res[max(res.keys())]
#    feat_selector = FeatureSelector(feats)
    scale_folder = os.path.join(cur_path, 'modelling', 'length', 'final', 'scalers', mod_name)
    scale_path = os.path.join(scale_folder, os.listdir(os.path.join(scale_folder))[0])
    scale = load(scale_path)
    pipe = Pipeline([('scaler', scale), ('clf', model)])
    pipe.fit(X[feats], Y)
    dump(pipe, os.path.join(cur_path, 'fit_models', 'length', '%s.pkl' % (mod_name)))
    
