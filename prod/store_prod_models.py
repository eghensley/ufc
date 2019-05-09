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
from joblib import dump
from model_validation.model_validation import pull_models
from model_validation.store_estimators import FeatureSelector

def pull_train_data(domain):
    data = pd.read_csv(os.path.join(cur_path, 'data', '%s_data.csv' % (domain)))
    data.set_index('bout_id', inplace = True)  
    
    
    data.drop('fighter_id', axis = 1, inplace = True)
    data.drop('opponent_id', axis = 1, inplace = True)
    data.drop('fight_date', axis = 1, inplace = True)
    
    X = data[[i for i in list(data) if i != domain]]
    
    if domain == 'length':
        Y = data[domain]/300
    elif domain == 'winner':
        Y = data[domain].apply(lambda x: x if x == 1 else 0)
    return(X, Y)


def store_final(domain):
    all_models = pull_models(domain)
    X, Y = pull_train_data(domain)
    for k,v in all_models.items():
        dump(v['meta'], os.path.join(cur_path, 'prod', 'models', domain, 'error', '%s.pkl' % (k)))
        est = v['est']
        est.fit(X, Y)
        dump(est, os.path.join(cur_path, 'prod', 'models', domain, 'est', '%s.pkl' % (k)))
        

if __name__ == '__main__':
    domain = 'winner'
    store_final(domain)