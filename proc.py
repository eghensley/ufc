import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))


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

PSQL = db_connection('psql')
#PSQL.reset_db_con()
cols = ['kd','ssa','sss','tsa','tss','sub','pas','rev','headssa','headsss','bodyssa', 'bodysss','legssa','legsss','disssa','dissss','clinssa','clinsss','gndssa','gndsss','tda','tds']

stats = pg_query(PSQL.client, 'SELECT bs.bout_id, date, fighter_id, kd, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa, legsss, disssa, dissss, clinssa, clinsss, gndssa, gndsss, tda, tds FROM ufc.bout_stats bs join ufc.bouts b on b.bout_id = bs.bout_id join ufc.fights f on f.fight_id = b.fight_id where champ is false;')
stats.columns = ['bout_id', 'fight_date', 'fighter_id', 'kd', 'ssa', 'sss', 'tsa', 'tss', 'sub', 'pas', 'rev', 'headssa', 'headsss', 'bodyssa', 'bodysss', 'legssa', 'legsss', 'disssa', 'dissss', 'clinssa', 'clinsss', 'gndssa', 'gndsss', 'tda', 'tds']

bouts = pg_query(PSQL.client, "select b.bout_id, weight_id, winner, loser from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id")
bouts.columns = ['bout_id', 'weight_id', 'winner', 'loser']
opponents = {i+j:k for i,j,k in bouts[['bout_id', 'winner', 'loser']].values}
for i,j,k in bouts[['bout_id', 'loser', 'winner']].values:
    opponents[i+j] = k
bouts = {i:{'winner':j, 'loser': k} for i,j,k in bouts[['bout_id', 'winner', 'loser']].values}
stats['opponent_id'] = (stats['bout_id'] + stats['fighter_id']).apply(lambda x: opponents[x] if x in opponents.keys() else np.nan)
stats.dropna(inplace = True)


def_stats = deepcopy(stats)
def_stats.drop('fighter_id', axis = 1, inplace = True)
def_stats.rename(columns = {'opponent_id': 'fighter_id'}, inplace = True)
def_stats.rename(columns = {i: 'd_'+i for i in cols}, inplace = True)

stats = pd.merge(stats, def_stats, left_on = ['fighter_id', 'bout_id', 'fight_date'], right_on = ['fighter_id', 'bout_id', 'fight_date'])
#stats = pd.merge(stats, opponents, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])

stat_avgs = {}
for fighter in stats.fighter_id.unique():
    f_stats = {}

    ostat = stats.loc[stats['fighter_id'] == fighter]
    ostat.sort_values('fight_date', inplace = True)

    dstat = stats.loc[stats['opponent_id'] == fighter]
    dstat.sort_values('fight_date', inplace = True)
    
    if len(dstat) != len(ostat):
        raise ValueError()
        
    if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
        raise ValueError()
        
    for i in range(len(ostat)):
        if i == 0:
            continue
        f_stats[ostat.iloc[i]['bout_id']] = {}
        for col in cols:
            f_stats[ostat.iloc[i]['bout_id']]['o_'+col] = ostat.iloc[:i][col].mean()  

    for i in range(len(dstat)):
        if i == 0:
            continue
        for col in cols:
            f_stats[dstat.iloc[i]['bout_id']]['d_'+col] = dstat.iloc[:i]['d_'+col].mean()

                      
    if len(f_stats.keys()) > 0:
        stat_avgs[fighter] = f_stats
        
        
        

        
        
        
        
        
        
        
        
        
        

#fight_length = pg_query(PSQL.client, 'SELECT bout_id, length from ufc.bout_results')
#fight_length = {i:k for i,k in fight_length.values}
#cols = ['kd', 'ssa', 'sss', 'tsa', 'tss', 'sub', 'pas', 'rev', 'headssa', 'headsss', 'bodyssa', 'bodysss', 'legssa', 'legsss', 'disssa', 'dissss', 'clinssa', 'clinsss', 'gndssa', 'gndsss', 'tda', 'tds']
#stats['length'] = stats['bout_id'].apply(lambda x: fight_length[x] if x in fight_length.keys() else np.nan)
#stats.dropna(inplace = True)
#for col in cols:
#    stats[col] = stats[col] / stats['length']        
        
fighters = pg_query(PSQL.client, 'select fighter_id, height, reach, stance, dob from ufc.fighters')
fighters.columns = ['fighter_id', 'height', 'reach', 'stance', 'dob']
fighters.set_index('fighter_id', inplace = True)
fighters = fighters.join(pd.get_dummies(fighters['stance']))
fighters.drop('stance', axis = 1, inplace = True)
fighters.rename(columns = {'': 'Missing Stance'}, inplace = True)
fighters.reset_index(inplace = True)
fighter_dob = {i:j for i,j in fighters[['fighter_id', 'dob']].values}
fighter_reach = {i:j for i,j in fighters[['fighter_id', 'reach']].values}
fighter_height = {i:j for i,j in fighters[['fighter_id', 'height']].values}      
        
        
        
        
off_stats = {}
for fighter in stats.fighter_id.unique():
    fstat = stats.loc[stats['fighter_id'] == fighter]
    fstat.sort_values('fight_date', inplace = True)
    f_stats = {}
    for i in range(len(fstat)):
        if i == 0:
            continue
        f_stats[fstat.iloc[i]['bout_id']] = {}
        for col in cols:
            f_stats[fstat.iloc[i]['bout_id']][col] = fstat.iloc[:i][col].mean()
#        f_stats[fstat.iloc[i]['bout_id']]['past_length'] = fstat.iloc[:i]['length'].sum()
#        f_stats[fstat.iloc[i]['bout_id']]['past_fights'] = len(fstat.iloc[:i])
#        f_stats[fstat.iloc[i]['bout_id']]['age'] = (fstat.iloc[i]['fight_date'] - fighter_dob[fighter]).days / 365
#        f_stats[fstat.iloc[i]['bout_id']]['height'] = fighter_height[fighter]
#        f_stats[fstat.iloc[i]['bout_id']]['reach'] = fighter_reach[fighter]
    if len(f_stats.keys()) > 0:
        off_stats[fighter] = f_stats
        













data = {}
i = 0
for k,v in all_stats.items():
    for kk, vv in v.items():
        vv['fighter_id'] = k
        vv['bout_id'] = kk
        data[i] = vv
        i += 1
        
data = pd.DataFrame.from_dict(data).T
        
bouts = pg_query(PSQL.client, "select b.bout_id, weight_id, winner, loser from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id")
bouts.columns = ['bout_id', 'weight_id', 'winner', 'loser']
bouts = {i:{'winner':j, 'loser': k} for i,j,k in bouts[['bout_id', 'winner', 'loser']].values}

winner_id = 0
winner = {}
for b,f in data[['bout_id', 'fighter_id']].values:
    if b in bouts.keys():
        if bouts[b]['winner'] == f:
            winner[winner_id] = {'bout_id': b, 'fighter_id': f, 'won': 1}
            winner_id += 1
        elif bouts[b]['loser'] == f:
            winner[winner_id] = {'bout_id': b, 'fighter_id': f, 'won': 0}
            winner_id += 1
        else:
            raise ValueError()

winner = pd.DataFrame.from_dict(winner).T
data = pd.merge(data, winner, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])




acc_stat_dict = {'acc_ss': ['ssa', 'sss'],
                   'acc_headss': ['headssa', 'headsss'],
                   'acc_bodyss': ['bodyssa', 'bodysss'],
                   'acc_legss': ['legssa', 'legsss'],
                   'acc_disss': ['disssa', 'dissss'],
                   'acc_clinss': ['clinssa', 'clinsss'],
                   'acc_gndss': ['gndssa', 'gndsss'],
                   'acc_td': ['tda', 'tds']}
share_ss_dict = {'share_headss': ['headssa', 'headsss'],
                   'share_bodyss': ['bodyssa', 'bodysss'],
                   'share_legss': ['legssa', 'legsss'],
                   'share_disss': ['disssa', 'dissss'],
                   'share_clinss': ['clinssa', 'clinsss'],
                   'share_gndss': ['gndssa', 'gndsss']}
for k, v in acc_stat_dict.items():
    stats[k] = stats[v[1]] / stats[v[0]]

for k, v in share_ss_dict.items():
    stats[k+'a'] = stats[v[0]]/stats['ssa']
    stats[k+'s'] = stats[v[0]]/stats['sss']










fighters = pg_query(PSQL.client, 'select fighter_id, height, reach, stance, dob from ufc.fighters')
fighters.columns = ['fighter_id', 'height', 'reach', 'stance', 'dob']
fighters.set_index('fighter_id', inplace = True)
fighters = fighters.join(pd.get_dummies(fighters['stance']))
fighters.drop('stance', axis = 1, inplace = True)
fighters.rename(columns = {'': 'Missing Stance'}, inplace = True)

bouts = pg_query(PSQL.client, "select date, b.bout_id, weight_id, winner, loser from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id")
bouts.columns = ['fight_date', 'bout_id', 'weight_id', 'winner', 'loser']
#bouts.weight_id.value_counts()
bouts.drop('weight_id', axis = 1, inplace = True)



















#winner_stats = bouts[['fight_date', 'bout_id', 'weight_id', 'winner']]
winner_stats = bouts[['fight_date', 'bout_id', 'winner']]
winner_stats.set_index('winner', inplace = True)
winner_stats = winner_stats.join(fighters)
winner_stats = winner_stats.reset_index().merge(stats, left_on = ['index', 'bout_id'], right_on = ['fighter_id', 'bout_id'])
winner_stats.drop('index', axis = 1, inplace = True)
winner_stats['age'] = (winner_stats['fight_date'] - winner_stats['dob']).apply(lambda x: x.days/365)
winner_stats.drop('dob', axis = 1, inplace = True)
winner_stats.drop('fight_date', axis = 1, inplace = True)
winner_stats.drop('fighter_id', axis = 1, inplace = True)


#loser_stats = bouts[['fight_date', 'bout_id', 'weight_id', 'loser']]
loser_stats = bouts[['fight_date', 'bout_id', 'loser']]
loser_stats.set_index('loser', inplace = True)
loser_stats = loser_stats.join(fighters)
loser_stats = loser_stats.reset_index().merge(stats, left_on = ['index', 'bout_id'], right_on = ['fighter_id', 'bout_id'])
loser_stats.drop('index', axis = 1, inplace = True)
loser_stats['age'] = (loser_stats['fight_date'] - loser_stats['dob']).apply(lambda x: x.days/365)
loser_stats.drop('dob', axis = 1, inplace = True)
loser_stats.drop('fight_date', axis = 1, inplace = True)
loser_stats.drop('fighter_id', axis = 1, inplace = True)


valid_bouts = [i for i in loser_stats.bout_id.values if i in winner_stats.bout_id.values]
random.shuffle(valid_bouts)
winner_bouts = valid_bouts[:int(len(valid_bouts)/2)]
loser_bouts = valid_bouts[int(len(valid_bouts)/2):]

if len([i for i in winner_bouts if i in loser_bouts]) > 0:
    raise ValueError()
    
win_stats_win = winner_stats.loc[winner_stats['bout_id'].apply(lambda x: True if x in winner_bouts else False)]
win_stats_win.set_index('bout_id', inplace = True)
win_stats_lose = loser_stats.loc[loser_stats['bout_id'].apply(lambda x: True if x in winner_bouts else False)]
win_stats_lose.set_index('bout_id', inplace = True)

win_stats_diff = {}
stat_cols = ['height','reach','kd','ssa','sss','tsa','tss','sub','pas','rev','headssa','headsss','bodyssa',
 'bodysss','legssa','legsss','disssa','dissss','clinssa','clinsss','gndssa','gndsss','tda','tds','age']
for idx in win_stats_win.index:
    win_stats_diff[idx] = {}
    for col in stat_cols:
        win_stats_diff[idx][col] = win_stats_win.loc[idx][col] - win_stats_lose.loc[idx][col]
win_stats_diff = pd.DataFrame.from_dict(win_stats_diff).T


lose_stats_win = loser_stats.loc[loser_stats['bout_id'].apply(lambda x: True if x in loser_bouts else False)]
lose_stats_win.set_index('bout_id', inplace = True)
lose_stats_lose = winner_stats.loc[winner_stats['bout_id'].apply(lambda x: True if x in loser_bouts else False)]
lose_stats_lose.set_index('bout_id', inplace = True)

lose_stats_diff = {}
stat_cols = ['height','reach','kd','ssa','sss','tsa','tss','sub','pas','rev','headssa','headsss','bodyssa',
 'bodysss','legssa','legsss','disssa','dissss','clinssa','clinsss','gndssa','gndsss','tda','tds','age']
for idx in lose_stats_win.index:
    lose_stats_diff[idx] = {}
    for col in stat_cols:
        lose_stats_diff[idx][col] = lose_stats_lose.loc[idx][col] - lose_stats_win.loc[idx][col]
lose_stats_diff = pd.DataFrame.from_dict(lose_stats_diff).T

fight_stats = win_stats_diff.append(lose_stats_diff)
fight_stats.reset_index(inplace = True)
fight_stats['winner'] = fight_stats['index'].apply(lambda x: 1 if x in winner_bouts else 0)
fight_stats.set_index('index', inplace = True)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
X = fight_stats[[i for i in list(fight_stats) if i != 'winner']]
Y = fight_stats['winner']

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

