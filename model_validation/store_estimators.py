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
#from model_tuning.meta_scoring_data import pull_val_data
import pandas as pd

class FeatureSelector(object):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X):
        return X.loc[:,self.cols ] 
    def fit(self, X, y=None):
        return self


def pull_val_data(domain):
#    domain = 'winner'
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
    

def store_estimators(domain):
    X, Y = pull_val_data(domain)
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
    

def store_error_estimators(domain):
    
    pred_cols = os.listdir(os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'predictors'))
    pred_cols = [i.split('.')[0] for i in pred_cols if i != '.DS_Store']
    
    meta_data = pd.read_csv(os.path.join(cur_path, 'data', 'meta', 'meta_%s.csv' % (domain)))
    meta_data.set_index('Unnamed: 0', inplace = True)
    X = meta_data[[i for i in list(meta_data) if i not in pred_cols]]
    
    
    for mod_name in pred_cols:
        Y = meta_data[mod_name]
        final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'meta', mod_name, 'final')
        model_path = os.path.join(final_model_folder, 'models', 'LassoRegression', '5.pkl')
        model = load(model_path)
        feats_folder = os.path.join(final_model_folder, 'features', 'LassoRegression.json')
        with open(feats_folder, 'r') as fp:
            feats = json.load(fp)
            feats = feats[max(feats.keys())]
        feat_selector = FeatureSelector(feats)
        scale_path = os.path.join(final_model_folder, 'scalers', 'LassoRegression', '5.pkl')
        scale = load(scale_path)
        pipe = Pipeline([('feat_selection', feat_selector), ('scaler', scale), ('clf', model)])
        pipe.fit(X, Y)
        dump(pipe, os.path.join(cur_path, 'model_validation', 'fit_models', domain, 'meta', '%s.pkl' % (mod_name)))
    

if __name__ == '__main__':
    domain = 'winner'
    store_estimators(domain)
#    store_error_estimators(domain)