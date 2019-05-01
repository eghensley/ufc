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
from sklearn.linear_model import LassoCV, Lasso
from joblib import dump


pred_cols = ['PolySVR',
             'LassoRegression',
             'LightGBR',
             'RFreg',
             'DartrGBM',
             'LinSVR',
             'RbfSVR']


len_res_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'pred_res_length.csv'))
len_res_data.set_index('Unnamed: 0', inplace = True)
X = len_res_data[[i for i in list(len_res_data) if i not in pred_cols]]

for meta_dimension in pred_cols:
    Y = len_res_data[meta_dimension]
    model = LassoCV(random_state = 1108, cv = 8, max_iter=5000)
    model.fit(X, Y)
        
    store_model = Lasso(random_state = 1108, alpha = model.alpha_)
    store_model.fit(X,Y)
    dump(store_model, os.path.join(cur_path, 'error_preds', 'length', '%s.pkl' % (meta_dimension)))
    
    
    
    
pred_cols = ['KNN',
             'LogRegression',
             'LinSVC',
             'RFclass',
             'RbfSVC',
             'PolySVC',
             'DartGBM',
             'LightGBM']
winner_res_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'pred_res_winner.csv'))
winner_res_data.set_index('Unnamed: 0', inplace = True)
X = winner_res_data[[i for i in list(winner_res_data) if i not in pred_cols]]

for meta_dimension in pred_cols:
    Y = winner_res_data[meta_dimension]
    model = LassoCV(random_state = 1108, cv = 8, max_iter=5000)
    model.fit(X, Y)
        
    store_model = Lasso(random_state = 1108, alpha = model.alpha_)
    store_model.fit(X,Y)
    dump(store_model, os.path.join(cur_path, 'error_preds', 'winner', '%s.pkl' % (meta_dimension)))
    
    


