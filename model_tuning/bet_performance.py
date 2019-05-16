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
#from model_tuning.meta_scoring_data import pull_val_data
import pandas as pd
from _connections import db_connection
from pop_psql import pg_query
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt



def conv_to_ml_odds(prob):
    if prob > .5:
        odds = (prob/(1-prob))*100
    else:
        odds = ((1-prob)/prob)*100
    return(odds)


def pot_payout(odd, stake = 40):
#    print(odd)
    odds = conv_to_ml_odds(odd)
#    print(odds)
    if odds > 0:
        po = stake*(odds/100)
    else:
        po = stake/(odds/100)*-1
#    print(po)
    return(po)
    
level = 'test'
pred_df = pd.read_csv(os.path.join(cur_path, 'model_%s.csv' % (level)))
pred_df.set_index('bout_id', inplace = True)

all_bouts = set(pred_df.index)

models = [i for i in list(pred_df) if i not in ['fighter_id', 'winner', 'VEGAS']]
pred_df['Comb'] = pred_df[models].mean(axis = 1)
models = [i for i in list(pred_df) if i not in ['fighter_id', 'winner', 'VEGAS']]

risk = 40
thresh_results = {}
thresh = 0
for model in models:
    bank = [250]
    for bout in all_bouts:
        bout_info = pred_df.loc[bout]
        bout_info['pred_diff'] = bout_info[model] - bout_info['VEGAS']
        
        bet_adv = bout_info.loc[bout_info['pred_diff'] > thresh/100]
        
        if len(bet_adv) != 1:
            bank.append(bank[-1])
            continue
         
        pot_profit = pot_payout(bet_adv['VEGAS'].values[0])
        if bet_adv['winner'].values[0] == 1:
            bank.append(bank[-1] + pot_profit)
        else:
            bank.append(bank[-1] - risk)
    thresh_results[model] = bank


for model in models:
    bank = [250]
    for bout in all_bouts:
        bout_info = pred_df.loc[bout]
        bout_info['pred_diff'] = bout_info[model] - bout_info['VEGAS']
        
        bet_adv = bout_info.loc[bout_info['pred_diff'] > thresh/100]
        
        if len(bet_adv) != 1:
            bank.append(bank[-1])
            continue
         
        risk = 40 + 40 * bet_adv['pred_diff'].values[0]
        pot_profit = pot_payout(bet_adv['VEGAS'].values[0], stake = risk)
        if bet_adv['winner'].values[0] == 1:
            bank.append(bank[-1] + pot_profit)
        else:
            bank.append(bank[-1] - risk)
    thresh_results[model+'_mult'] = bank
    

for model in models:
    bank = [250]
    for bout in all_bouts:
        bout_info = pred_df.loc[bout]
        bout_info['pred_diff'] = bout_info[model] - bout_info['VEGAS']
        
        bet_adv = bout_info.loc[bout_info['pred_diff'] > thresh/100]
        
        if len(bet_adv) != 1:
            bank.append(bank[-1])
            continue
         
        risk = 40 **(1+bet_adv['pred_diff'].values[0])
        pot_profit = pot_payout(bet_adv['VEGAS'].values[0], stake = risk)
        if bet_adv['winner'].values[0] == 1:
            bank.append(bank[-1] + pot_profit)
        else:
            bank.append(bank[-1] - risk)
    thresh_results[model+'_exp'] = bank
    
    
fig = plt.figure(figsize=(10,5))   
ax = fig.add_subplot(111)
for mod, res in thresh_results.items():
    if '_mult' in mod:
        ls = ':'
    elif '_exp' in mod:
        ls = '-.'
    else:
        ls = '-'
#    if '_mult' not in mod:
#        continue
    ax.plot(res, label = mod, linestyle = ls)
ax.legend(loc='best')
plt.show()    
    
#risk = 40
#thresh_results = {}
#for thresh in range(15):
#    bank = [250]
#    for bout in all_bouts:
#        bout_info = pred_df.loc[bout]
#        bout_info['pred_diff'] = bout_info[model] - bout_info['VEGAS']
#        
#        bet_adv = bout_info.loc[bout_info['pred_diff'] > thresh/100]
#        
#        if len(bet_adv) != 1:
#            bank.append(bank[-1])
#            continue
#            
#        pot_profit = pot_payout(bet_adv['VEGAS'].values[0])
#        if bet_adv['winner'].values[0] == 1:
#            bank.append(bank[-1] + pot_profit)
#        else:
#            bank.append(bank[-1] - risk)
#    thresh_results[thresh] = bank
#
#for threshold, res in thresh_results.items():
#    plt.plot(res, label = threshold)
#plt.show()