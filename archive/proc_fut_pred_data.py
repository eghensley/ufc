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

def pull_avg_data(stats):
    
        stats = pd.read_csv(os.path.join(cur_path, 'test_data', 'stats.csv'))
        stats.drop('Unnamed: 0', axis = 1, inplace = True)
        
        fut_bouts = pg_query(PSQL.client, "SELECT date, bx.bout_id, bx.fighter_id, opponent_id FROM ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id")
        fut_bouts.columns = ['date', 'bout_id', 'fighter_id', 'opponent_id']
        stat_avgs = {}
        for date, bout, fighter, opponent in fut_bouts.values:
            f_stats = {}

            ostat = stats.loc[stats['fighter_id'] == fighter]
            ostat.sort_values('fight_date', inplace = True)
        
            dstat = stats.loc[stats['opponent_id'] == fighter]
            dstat.sort_values('fight_date', inplace = True)
            
            if len(dstat) != len(ostat):
                raise ValueError()
                
            if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
                raise ValueError()
            
            f_stats[bout] = {'fight_date': date, 'opponent_id': opponent}
            
            for col in cols:
                f_stats[bout]['avg_o_'+col] = ostat[col].mean()  

            for col in cols:
                f_stats[bout]['avg_d_'+col] = dstat['d_'+col].mean()
                                  
            if len(f_stats.keys()) > 0:
                stat_avgs[fighter] = f_stats
                
        avg_data = {}
        i = 0
        for k,v in stat_avgs.items():
            for kk, vv in v.items():
                vv['fighter_id'] = k
                vv['bout_id'] = kk
                avg_data[i] = vv
                i += 1
        
        avg_data = pd.DataFrame.from_dict(avg_data).T
        avg_data.to_csv(os.path.join(cur_path, 'test_data', 'avg_data_TEST.csv'))
        return(avg_data)
    
#def pull_adj_data(avg_data, stats):
#        stats = pd.read_csv(os.path.join(cur_path, 'test_data', 'stats.csv'))
#        stats.drop('Unnamed: 0', axis = 1, inplace = True)
#        
#        avg_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'avg_data.csv'))
#        avg_data.drop('Unnamed: 0', axis = 1, inplace = True)
#        
#        
#        avg_data_test = pd.read_csv(os.path.join(cur_path, 'test_data', 'avg_data_TEST.csv'))
#        avg_data_test.drop('Unnamed: 0', axis = 1, inplace = True)
#        
#        avg_data = avg_data.append(avg_data_test)
#        stat_adj = {}
#        for fighter in avg_data_test.fighter_id.unique():
#            
#            f_avgs = avg_data.loc[avg_data['fighter_id'] == fighter]
#            f_avgs.sort_values('fight_date', inplace = True)
#        
#            o_avgs = avg_data.loc[avg_data['opponent_id'] == fighter]
#            o_avgs.sort_values('fight_date', inplace = True)
#        
#            f_stats = stats.loc[stats['fighter_id'] == fighter]
#            f_stats.sort_values('fight_date', inplace = True)
#        
#            common_bouts = set([j for j in [i for i in f_avgs['bout_id'].values if i in o_avgs['bout_id'].values] if j in f_stats['bout_id'].values])
#            f_avgs = f_avgs.loc[f_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#            o_avgs = o_avgs.loc[o_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#            f_stats = f_stats.loc[f_stats['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#        
#            adj_stats = f_avgs[['bout_id', 'fight_date', 'opponent_id']]     
#            for col in cols:
#                
#        #        (f_stats['d_'+col]-f_avgs['d_'+col])/f_avgs['d_'+col+'_std']
#                adj_stats['adj_d_'+col] = (f_stats['d_'+col] /o_avgs['avg_o_'+col].apply(lambda x: np.nan if x == 0 else x)).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats['d_'+col]
#                adj_stats['adj_o_'+col] = (f_stats[col] / o_avgs['avg_d_'+col].apply(lambda x: np.nan if x == 0 else x)).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats[col]
#            if len(adj_stats) > 0:
#                stat_adj[fighter] = adj_stats.set_index('bout_id').T.to_dict()
#                
#        adj_data = {}
#        i = 0
#        for k,v in stat_adj.items():
#            for kk, vv in v.items():
#                vv['fighter_id'] = k
#                vv['bout_id'] = kk
#                adj_data[i] = vv
#                i += 1
#        
#        adj_data = pd.DataFrame.from_dict(adj_data).T
#        adj_data.to_csv(os.path.join(cur_path, 'test_data', 'adj_data.csv'))
#    return(adj_data)

def pull_adj_avg_data(adj_data):
        adj_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'adj_data.csv'))
        adj_data.drop('Unnamed: 0', axis = 1, inplace = True)

        fut_bouts = pg_query(PSQL.client, "SELECT date, bx.bout_id, bx.fighter_id, opponent_id FROM ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id")
        fut_bouts.columns = ['date', 'bout_id', 'fighter_id', 'opponent_id']
        adj_stat_avgs = {}
        for date, bout, fighter, opponent in fut_bouts.values:
            f_stats = {}
            ostat = adj_data.loc[adj_data['fighter_id'] == fighter]
            ostat.sort_values('fight_date', inplace = True)
        
            dstat = adj_data.loc[adj_data['opponent_id'] == fighter]
            dstat.sort_values('fight_date', inplace = True)
            
            if len(dstat) != len(ostat):
                raise ValueError()
                
            if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
                raise ValueError()

            f_stats[bout] = {'fight_date': date, 'opponent_id': opponent}
            for col in cols:
                f_stats[bout]['adj_avg_o_'+col] = ostat['adj_o_'+col].mean()  

            for col in cols:
                f_stats[bout]['adj_avg_d_'+col] = dstat['adj_d_'+col].mean()
                                  
            if len(f_stats.keys()) > 0:
                adj_stat_avgs[fighter] = f_stats
                
        adj_avg_data = {}
        i = 0
        for k,v in adj_stat_avgs.items():
            for kk, vv in v.items():
                vv['fighter_id'] = k
                vv['bout_id'] = kk
                adj_avg_data[i] = vv
                i += 1
    
        adj_avg_data = pd.DataFrame.from_dict(adj_avg_data).T
        adj_avg_data.to_csv(os.path.join(cur_path, 'test_data', 'adj_avg_data_TEST.csv'))    
        return(adj_avg_data)


def pull_pred_data(avg_data, adj_avg_data):
        avg_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'avg_data_TEST.csv'))
        avg_data.drop('Unnamed: 0', axis = 1, inplace = True)

        adj_avg_data = pd.read_csv(os.path.join(cur_path, 'test_data', 'adj_avg_data_TEST.csv'))
        adj_avg_data.drop('Unnamed: 0', axis = 1, inplace = True)

        
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
            avg_data['avg_o_'+k] = (avg_data['avg_o_'+v[1]] / avg_data['avg_o_'+v[0]]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            avg_data['avg_d_'+k] = (avg_data['avg_d_'+v[1]] / avg_data['avg_d_'+v[0]]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
        for k, v in share_ss_dict.items():
            avg_data['avg_o_'+k+'_a'] = (avg_data['avg_o_'+v[0]]/avg_data['avg_o_'+'ssa']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            avg_data['avg_d_'+k+'_a'] = (avg_data['avg_d_'+v[0]]/avg_data['avg_d_'+'ssa']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
            avg_data['avg_o_'+k+'_s'] = (avg_data['avg_o_'+v[1]]/avg_data['avg_o_'+'sss']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            avg_data['avg_d_'+k+'_s'] = (avg_data['avg_d_'+v[1]]/avg_data['avg_d_'+'sss']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
        
        for k, v in acc_stat_dict.items():
            adj_avg_data['adj_avg_o_'+k] = (adj_avg_data['adj_avg_o_'+v[1]] / adj_avg_data['adj_avg_o_'+v[0]]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            adj_avg_data['adj_avg_d_'+k] = (adj_avg_data['adj_avg_d_'+v[1]] / adj_avg_data['adj_avg_d_'+v[0]]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
        for k, v in share_ss_dict.items():
            adj_avg_data['adj_avg_o_'+k+'_a'] = (adj_avg_data['adj_avg_o_'+v[0]]/adj_avg_data['adj_avg_o_'+'ssa']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            adj_avg_data['adj_avg_d_'+k+'_a'] = (adj_avg_data['adj_avg_d_'+v[0]]/adj_avg_data['adj_avg_d_'+'ssa']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
            adj_avg_data['adj_avg_o_'+k+'_s'] = (adj_avg_data['adj_avg_o_'+v[1]]/adj_avg_data['adj_avg_o_'+'sss']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
            adj_avg_data['adj_avg_d_'+k+'_s'] = (adj_avg_data['adj_avg_d_'+v[1]]/adj_avg_data['adj_avg_d_'+'sss']).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 0)
        
        
        data = pd.merge(avg_data, adj_avg_data, left_on = ['bout_id', 'fighter_id', 'fight_date', 'opponent_id'], right_on = ['bout_id', 'fighter_id', 'fight_date', 'opponent_id'])
        data.dropna(inplace = True)
        fighters = pg_query(PSQL.client, 'select fighter_id, height, reach, stance, dob from ufc.fighters')
        fighters.columns = ['fighter_id', 'height', 'reach', 'stance', 'dob']
        fighters.set_index('fighter_id', inplace = True)
        fighters = fighters.join(pd.get_dummies(fighters['stance']))
        fighters.drop('stance', axis = 1, inplace = True)
        fighters.rename(columns = {'': 'Missing Stance'}, inplace = True)
        fighters.reset_index(inplace = True)
        fighter_dob = {i:j for i,j in fighters[['fighter_id', 'dob']].values}
        
        i = 0
        fighter_ages = {}
        for bout, fighter, date in data[['bout_id', 'fighter_id', 'fight_date']].values:
            fighter_ages[i] = {'bout_id': bout, 'fighter_id':fighter, 'age': (datetime.strptime(date.split(' ')[0], '%Y-%m-%d') - fighter_dob[fighter]).days/365}
            i += 1
        data = pd.merge(data, pd.DataFrame.from_dict(fighter_ages).T, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])
        
        fighter_reach = {i:j for i,j in fighters[['fighter_id', 'reach']].values}
        fighter_height = {i:j for i,j in fighters[['fighter_id', 'height']].values}   
        
        data['reach'] = data['fighter_id'].apply(lambda x: fighter_reach[x])
        data['height'] = data['fighter_id'].apply(lambda x: fighter_height[x])
        
        pred_data = {}
        hold_cols = ['bout_id', 'fighter_id', 'fight_date', 'opponent_id']
        for bout in data['bout_id'].unique():
            bout_data = data.loc[data['bout_id'] == bout].sample(frac=1)
            if len(bout_data) != 2:
                continue
            bout_data.reset_index(inplace = True, drop = True)
            bout_meta = bout_data[hold_cols]
            bout_data = bout_data[[i for i in list(bout_data) if i not in hold_cols]]
            bout_preds = {}
            for k,v in (bout_data.T[0] - bout_data.T[1]).to_dict().items():
                bout_preds[k+'_diff'] = v
            for k,v in ((bout_data.T[0] + bout_data.T[1])/2).to_dict().items():
                bout_preds[k+'_avg'] = v                
            cur_cols = list(bout_data)
            for col in cur_cols:
                if '_o_' in col:
                    bout_preds[col+'_xdif'] = bout_data.loc[0][col] - bout_data.loc[1][col.replace('_o_', '_d_')]            
                elif '_d_' in col:
                    bout_preds[col+'_xdif'] = bout_data.loc[0][col] - bout_data.loc[1][col.replace('_d_', '_o_')]
                else:
                    continue    
        #    bout_preds.pop('won_x')
            for k,v in bout_meta.T[0].to_dict().items():
                bout_preds[k] = v
            bout_preds.pop('bout_id')
            pred_data[bout] = bout_preds
            
        pred_data = pd.DataFrame.from_dict(pred_data).T
        pred_data.reset_index(inplace = True)
        pred_data.rename(columns = {'index':'bout_id'}, inplace = True)
        pred_data.to_csv(os.path.join(cur_path, 'test_data', 'pred_data_TEST.csv'))
        



asdfasdf
if __name__ == '__main__':
    stats = pull_stats()
    avg_data = pull_avg_data(stats)
    adj_data = pull_adj_data(avg_data, stats)
    adj_avg_data = pull_adj_avg_data(adj_data)
    
    lhglkh
    
    
    
    
    pred_data_winner, pred_data_length = pull_pred_data(avg_data, adj_avg_data)
    
    adsfasdfaf
    
