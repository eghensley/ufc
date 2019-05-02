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
from pop_psql import pg_query, pg_insert
import random
from copy import deepcopy
import numpy as np

PSQL = db_connection('psql')
#PSQL.reset_db_con()
cols = ['kd','ssa','sss','tsa','tss','sub','pas','rev','headssa','headsss','bodyssa', 'bodysss','legssa','legsss','disssa','dissss','clinssa','clinsss','gndssa','gndsss','tda','tds']

def pull_stats():
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
    
    bout_len = pg_query(PSQL.client, "SELECT bout_id, length from ufc.bout_results")
    bout_len.columns = ['bout_id', 'length']
    stats = pd.merge(stats, bout_len, left_on = 'bout_id', right_on = 'bout_id')
    for col in cols:
        stats[col] = stats[col] / stats['length']
    stats.drop('length', axis = 1, inplace = True)
    
    def_stats = deepcopy(stats)
    def_stats.drop('fighter_id', axis = 1, inplace = True)
    def_stats.rename(columns = {'opponent_id': 'fighter_id'}, inplace = True)
    def_stats.rename(columns = {i: 'd_'+i for i in cols}, inplace = True)
    
    stats = pd.merge(stats, def_stats, left_on = ['fighter_id', 'bout_id', 'fight_date'], right_on = ['fighter_id', 'bout_id', 'fight_date'])
    def_stats = None
    return(stats)


def pop_avg_data():
    stats = pull_stats()
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
            
            f_stats[ostat.iloc[i]['bout_id']] = {'fight_date': ostat.iloc[i]['fight_date'], 'opponent_id': ostat.iloc[i]['opponent_id']}
            for col in cols:
                f_stats[ostat.iloc[i]['bout_id']]['avg_o_'+col] = ostat.iloc[:i][col].mean()  
        for i in range(len(dstat)):
            if i == 0:
                continue
            for col in cols:
                f_stats[dstat.iloc[i]['bout_id']]['avg_d_'+col] = dstat.iloc[:i]['d_'+col].mean()
                              
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
    
    for avg_d_bodyssa, avg_d_bodysss, avg_d_clinssa, avg_d_clinsss, avg_d_disssa,\
     avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,avg_d_kd,avg_d_legssa,avg_d_legsss,\
     avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds,\
     avg_d_tsa, avg_d_tss, avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa,\
     avg_o_dissss,avg_o_gndssa,avg_o_gndsss,avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss,\
     avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss,bout_id,\
     fight_date,fighter_id,opponent_id in avg_data.values:
         
        script = "INSERT INTO ufc.avg_stats(\
        	fighter_id, bout_id, avg_o_kd, avg_o_ssa, avg_o_sss, avg_o_tsa, avg_o_tss, avg_o_sub, avg_o_pas, avg_o_rev,\
            avg_o_headssa, avg_o_headsss, avg_o_bodyssa, avg_o_bodysss, avg_o_legssa, avg_o_legsss, avg_o_disssa, avg_o_dissss,\
            avg_o_clinssa, avg_o_clinsss, avg_o_gndssa, avg_o_gndsss, avg_o_tda, avg_o_tds, avg_d_kd, avg_d_ssa, avg_d_sss,\
            avg_d_tsa, avg_d_tss, avg_d_sub, avg_d_pas, avg_d_rev, avg_d_headssa, avg_d_headsss, avg_d_bodyssa, avg_d_bodysss,\
            avg_d_legssa, avg_d_legsss, avg_d_disssa, avg_d_dissss, avg_d_clinssa, avg_d_clinsss, avg_d_gndssa, avg_d_gndsss, avg_d_tda, avg_d_tds)\
        	VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, bout_id, avg_d_bodyssa, avg_d_bodysss, \
            avg_d_clinssa, avg_d_clinsss, avg_d_disssa, avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,\
            avg_d_kd,avg_d_legssa,avg_d_legsss, avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds, avg_d_tsa, avg_d_tss, \
            avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa, avg_o_dissss,avg_o_gndssa,avg_o_gndsss,\
            avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss, avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,\
            avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss)
        pg_insert(PSQL.client, script)
    

def pull_avg_data():
    data = pg_query(PSQL.client, 'Select * from ufc.avg_stats')
    data.columns = ['fighter_id', 'bout_id', 'avg_o_kd', 'avg_o_ssa', 'avg_o_sss', 'avg_o_tsa', 'avg_o_tss', 'avg_o_sub', 'avg_o_pas', 'avg_o_rev',
                'avg_o_headssa', 'avg_o_headsss', 'avg_o_bodyssa', 'avg_o_bodysss', 'avg_o_legssa', 'avg_o_legsss', 'avg_o_disssa', 'avg_o_dissss',
                'avg_o_clinssa', 'avg_o_clinsss', 'avg_o_gndssa', 'avg_o_gndsss', 'avg_o_tda', 'avg_o_tds', 'avg_d_kd', 'avg_d_ssa', 'avg_d_sss',
                'avg_d_tsa', 'avg_d_tss', 'avg_d_sub', 'avg_d_pas', 'avg_d_rev', 'avg_d_headssa', 'avg_d_headsss', 'avg_d_bodyssa', 'avg_d_bodysss',
                'avg_d_legssa', 'avg_d_legsss', 'avg_d_disssa', 'avg_d_dissss', 'avg_d_clinssa', 'avg_d_clinsss', 'avg_d_gndssa', 'avg_d_gndsss', 'avg_d_tda', 'avg_d_tds']
    opponents = pg_query(PSQL.client, 'Select * from ufc.bout_fighter_xref')
    opponents.columns = ['bout_id', 'fighter_id', 'opponent_id']
    
    data = pd.merge(data, opponents, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])

    dates = pg_query(PSQL.client, 'Select bout_id, date from ufc.fights f join ufc.bouts b on b.fight_id = f.fight_id')
    dates.columns = ['bout_id', 'fight_date']

    data = pd.merge(data, dates, left_on = 'bout_id', right_on = 'bout_id')
    return(data)


def pop_adj_data():
    stats = pull_stats()
    avg_data = pull_avg_data()
    
    stat_adj = {}
    for fighter in avg_data.fighter_id.unique():
        
        f_avgs = avg_data.loc[avg_data['fighter_id'] == fighter]
        f_avgs.sort_values('fight_date', inplace = True)
    
        o_avgs = avg_data.loc[avg_data['opponent_id'] == fighter]
        o_avgs.sort_values('fight_date', inplace = True)
    
        f_stats = stats.loc[stats['fighter_id'] == fighter]
        f_stats.sort_values('fight_date', inplace = True)
    
        common_bouts = set([j for j in [i for i in f_avgs['bout_id'].values if i in o_avgs['bout_id'].values] if j in f_stats['bout_id'].values])
        f_avgs = f_avgs.loc[f_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
        o_avgs = o_avgs.loc[o_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
        f_stats = f_stats.loc[f_stats['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
    
        adj_stats = f_avgs[['bout_id', 'fight_date', 'opponent_id']]     
        for col in cols:                
            adj_stats['adj_d_'+col] = (f_stats['d_'+col] /o_avgs['avg_o_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats['d_'+col]
            adj_stats['adj_o_'+col] = (f_stats[col] / o_avgs['avg_d_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats[col]
        if len(adj_stats) > 0:
            stat_adj[fighter] = adj_stats.set_index('bout_id').T.to_dict()
            
    adj_data = {}
    i = 0
    for k,v in stat_adj.items():
        for kk, vv in v.items():
            vv['fighter_id'] = k
            vv['bout_id'] = kk
            adj_data[i] = vv
            i += 1
    
    adj_data = pd.DataFrame.from_dict(adj_data).T

    for adj_d_bodyssa, adj_d_bodysss,adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
        adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
        adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
        adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
        adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
        adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
        adj_o_tsa,adj_o_tss,bout_id,fight_date,fighter_id,opponent_id in adj_data.values:
        
            script = "INSERT INTO ufc.adj_stats(fighter_id, bout_id, adj_d_bodyssa, adj_d_bodysss,\
                adj_d_clinssa, adj_d_clinsss, adj_d_disssa, adj_d_dissss, adj_d_gndssa, adj_d_gndsss,\
                adj_d_headssa, adj_d_headsss, adj_d_kd, adj_d_legssa, adj_d_legsss, adj_d_pas, \
                adj_d_rev, adj_d_ssa, adj_d_sss, adj_d_sub, adj_d_tda, adj_d_tds, adj_d_tsa, adj_d_tss,\
                adj_o_bodyssa, adj_o_bodysss, adj_o_clinssa, adj_o_clinsss, adj_o_disssa, adj_o_dissss, \
                adj_o_gndssa, adj_o_gndsss, adj_o_headssa, adj_o_headsss, adj_o_kd, adj_o_legssa, \
                adj_o_legsss, adj_o_pas, adj_o_rev, adj_o_ssa, adj_o_sss, adj_o_sub, adj_o_tda, \
                adj_o_tds, adj_o_tsa, adj_o_tss) VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, \
                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, \
                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, \
                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, \
                bout_id, adj_d_bodyssa, adj_d_bodysss,adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
                adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
                adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
                adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
                adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
                adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
                adj_o_tsa,adj_o_tss)
            pg_insert(PSQL.client, script)


def pull_adj_data():
    data = pg_query(PSQL.client, 'Select * from ufc.adj_stats')
    data.columns = ['fighter_id','bout_id','adj_d_bodyssa','adj_d_bodysss','adj_d_clinssa','adj_d_clinsss',
                    'adj_d_disssa','adj_d_dissss','adj_d_gndssa','adj_d_gndsss','adj_d_headssa','adj_d_headsss',
                    'adj_d_kd','adj_d_legssa','adj_d_legsss','adj_d_pas','adj_d_rev','adj_d_ssa','adj_d_sss',
                    'adj_d_sub','adj_d_tda','adj_d_tds','adj_d_tsa','adj_d_tss','adj_o_bodyssa','adj_o_bodysss',
                    'adj_o_clinssa','adj_o_clinsss','adj_o_disssa','adj_o_dissss','adj_o_gndssa','adj_o_gndsss',
                    'adj_o_headssa','adj_o_headsss','adj_o_kd','adj_o_legssa','adj_o_legsss','adj_o_pas',
                    'adj_o_rev','adj_o_ssa','adj_o_sss','adj_o_sub','adj_o_tda','adj_o_tds','adj_o_tsa','adj_o_tss']
    opponents = pg_query(PSQL.client, 'Select * from ufc.bout_fighter_xref')
    opponents.columns = ['bout_id', 'fighter_id', 'opponent_id']
    
    data = pd.merge(data, opponents, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])

    dates = pg_query(PSQL.client, 'Select bout_id, date from ufc.fights f join ufc.bouts b on b.fight_id = f.fight_id')
    dates.columns = ['bout_id', 'fight_date']

    data = pd.merge(data, dates, left_on = 'bout_id', right_on = 'bout_id')
    return(data)


def pop_adj_avg_data():
    adj_data = pull_adj_data()
    adj_stat_avgs = {}
    for fighter in adj_data.fighter_id.unique():
        f_stats = {}
        
        ostat = adj_data.loc[adj_data['fighter_id'] == fighter]
        ostat.sort_values('fight_date', inplace = True)
    
        dstat = adj_data.loc[adj_data['opponent_id'] == fighter]
        dstat.sort_values('fight_date', inplace = True)
        
        if len(dstat) != len(ostat):
            raise ValueError()
            
        if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
            raise ValueError()
            
        for i in range(len(ostat)):
            if i == 0:
                continue
            
            f_stats[ostat.iloc[i]['bout_id']] = {'fight_date': ostat.iloc[i]['fight_date'], 'opponent_id': ostat.iloc[i]['opponent_id']}
            for col in cols:
                f_stats[ostat.iloc[i]['bout_id']]['adj_avg_o_'+col] = ostat.iloc[:i]['adj_o_'+col].mean()  
        for i in range(len(dstat)):
            if i == 0:
                continue
            for col in cols:
                f_stats[dstat.iloc[i]['bout_id']]['adj_avg_d_'+col] = dstat.iloc[:i]['adj_d_'+col].mean()
                              
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
    
    
    for adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
        adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,\
        adj_avg_d_kd,adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,\
        adj_avg_d_sss,adj_avg_d_sub,adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,\
        adj_avg_o_bodyssa,adj_avg_o_bodysss,adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,\
        adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,adj_avg_o_headssa,adj_avg_o_headsss,\
        adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,adj_avg_o_rev,adj_avg_o_ssa,\
        adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss,\
        bout_id,fight_date,fighter_id,opponent_id in adj_avg_data.values:
        
        
        script = "INSERT INTO ufc.adj_avg_stats(fighter_id, bout_id, adj_avg_d_bodyssa, adj_avg_d_bodysss,\
            adj_avg_d_clinssa, adj_avg_d_clinsss, adj_avg_d_disssa, adj_avg_d_dissss, adj_avg_d_gndssa, \
            adj_avg_d_gndsss, adj_avg_d_headssa, adj_avg_d_headsss, adj_avg_d_kd, adj_avg_d_legssa, adj_avg_d_legsss,\
            adj_avg_d_pas, adj_avg_d_rev, adj_avg_d_ssa, adj_avg_d_sss, adj_avg_d_sub, adj_avg_d_tda, \
            adj_avg_d_tds, adj_avg_d_tsa, adj_avg_d_tss, adj_avg_o_bodyssa, adj_avg_o_bodysss, adj_avg_o_clinssa,\
            adj_avg_o_clinsss, adj_avg_o_disssa, adj_avg_o_dissss, adj_avg_o_gndssa, adj_avg_o_gndsss, \
            adj_avg_o_headssa, adj_avg_o_headsss, adj_avg_o_kd, adj_avg_o_legssa, adj_avg_o_legsss, adj_avg_o_pas,\
            adj_avg_o_rev, adj_avg_o_ssa, adj_avg_o_sss, adj_avg_o_sub, adj_avg_o_tda, adj_avg_o_tds, adj_avg_o_tsa,\
            adj_avg_o_tss) VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, \
            bout_id, adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
            adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,adj_avg_d_kd,\
            adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,adj_avg_d_sss,adj_avg_d_sub,\
            adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,adj_avg_o_bodyssa,adj_avg_o_bodysss,\
            adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,\
            adj_avg_o_headssa,adj_avg_o_headsss,adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,\
            adj_avg_o_rev,adj_avg_o_ssa,adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss)
            
        pg_insert(PSQL.client, script)


def pull_adj_avg_data():
    data = pg_query(PSQL.client, 'Select * from ufc.adj_avg_stats')
    data.columns = ['fighter_id','bout_id', 'adj_avg_d_bodyssa','adj_avg_d_bodysss','adj_avg_d_clinssa','adj_avg_d_clinsss','adj_avg_d_disssa',
                     'adj_avg_d_dissss','adj_avg_d_gndssa','adj_avg_d_gndsss','adj_avg_d_headssa','adj_avg_d_headsss',
                     'adj_avg_d_kd','adj_avg_d_legssa','adj_avg_d_legsss','adj_avg_d_pas','adj_avg_d_rev','adj_avg_d_ssa',
                     'adj_avg_d_sss','adj_avg_d_sub','adj_avg_d_tda','adj_avg_d_tds','adj_avg_d_tsa','adj_avg_d_tss',
                     'adj_avg_o_bodyssa','adj_avg_o_bodysss','adj_avg_o_clinssa','adj_avg_o_clinsss','adj_avg_o_disssa',
                     'adj_avg_o_dissss','adj_avg_o_gndssa','adj_avg_o_gndsss','adj_avg_o_headssa','adj_avg_o_headsss',
                     'adj_avg_o_kd','adj_avg_o_legssa','adj_avg_o_legsss','adj_avg_o_pas','adj_avg_o_rev','adj_avg_o_ssa',
                     'adj_avg_o_sss','adj_avg_o_sub','adj_avg_o_tda','adj_avg_o_tds','adj_avg_o_tsa','adj_avg_o_tss']
    opponents = pg_query(PSQL.client, 'Select * from ufc.bout_fighter_xref')
    opponents.columns = ['bout_id', 'fighter_id', 'opponent_id']
    
    data = pd.merge(data, opponents, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])

    dates = pg_query(PSQL.client, 'Select bout_id, date from ufc.fights f join ufc.bouts b on b.fight_id = f.fight_id')
    dates.columns = ['bout_id', 'fight_date']

    data = pd.merge(data, dates, left_on = 'bout_id', right_on = 'bout_id')
    return(data)        


def pull_pred_data():
    avg_data = pull_avg_data()
    adj_avg_data = pull_adj_avg_data()
    
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
        fighter_ages[i] = {'bout_id': bout, 'fighter_id':fighter, 'age': (datetime.strptime(str(date).split(' ')[0], '%Y-%m-%d') - fighter_dob[fighter]).days/365}
        i += 1
    data = pd.merge(data, pd.DataFrame.from_dict(fighter_ages).T, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])
    
    fighter_reach = {i:j for i,j in fighters[['fighter_id', 'reach']].values}
    fighter_height = {i:j for i,j in fighters[['fighter_id', 'height']].values}   
    
    data['reach'] = data['fighter_id'].apply(lambda x: fighter_reach[x])
    data['height'] = data['fighter_id'].apply(lambda x: fighter_height[x])
    
    
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

    bout_len = pg_query(PSQL.client, "SELECT bout_id, length from ufc.bout_results")
    bout_len.columns = ['bout_id', 'length'] 
    
    data = pd.merge(data, bout_len, left_on = 'bout_id', right_on = 'bout_id')

    stats = pull_stats()
    stats = pd.merge(stats, bout_len, left_on = 'bout_id', right_on = 'bout_id')
    winner = {}
    for b,f in stats[['bout_id', 'fighter_id']].values:
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
    stats = pd.merge(stats, winner, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])
       
    streak_data = {}         
    for fighter in stats.fighter_id.unique():    
        add_data = stats.loc[stats['fighter_id'] == fighter][['bout_id', 'fight_date', 'length', 'won']]
        add_data.sort_values('fight_date', inplace = True)
        
        f_streak = {}
        for i in range(len(add_data)):
            if i == 0:
                continue
            f_streak[add_data.iloc[i]['bout_id']] = {}
            
            f_streak[add_data.iloc[i]['bout_id']]['len_avg'] = add_data.iloc[:i]['length'].mean()
            f_streak[add_data.iloc[i]['bout_id']]['win_avg'] = add_data.iloc[:i]['won'].mean()
            last_res = add_data.iloc[i-1]['won']
            streak_count = 0
            for res in reversed(add_data.iloc[:i]['won'].values):
                if res == last_res:
                    streak_count += 1
                else:
                    break
            if last_res == 0:
                streak_count *= -1
            f_streak[add_data.iloc[i]['bout_id']]['win_streak'] = streak_count
        if len(f_streak.keys()) > 0:
            streak_data[fighter] = f_streak

    streak_avg_data = {}
    i = 0
    for k,v in streak_data.items():
        for kk, vv in v.items():
            vv['fighter_id'] = k
            vv['bout_id'] = kk
            streak_avg_data[i] = vv
            i += 1

    streak_avg_data = pd.DataFrame.from_dict(streak_avg_data).T            
    data = pd.merge(data, streak_avg_data, left_on = ['bout_id', 'fighter_id'], right_on = ['bout_id', 'fighter_id'])
    
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
        bout_preds.pop('won_avg')         
        bout_preds.keys()
        bout_preds.pop('length_diff')              
        cur_cols = list(bout_data)
        for col in cur_cols:
            if '_o_' in col:
                bout_preds[col+'_xdif'] = bout_data.loc[0][col] - bout_data.loc[1][col.replace('_o_', '_d_')]            
            elif '_d_' in col:
                bout_preds[col+'_xdif'] = bout_data.loc[0][col] - bout_data.loc[1][col.replace('_d_', '_o_')]
            else:
                continue   
        for k,v in bout_meta.T[0].to_dict().items():
            bout_preds[k] = v
        bout_preds.pop('bout_id')
        pred_data[bout] = bout_preds
        
    pred_data = pd.DataFrame.from_dict(pred_data).T
    pred_data.reset_index(inplace = True)
    pred_data.rename(columns = {'index':'bout_id'}, inplace = True)
    pred_data = pd.merge(pred_data, bout_len, left_on = 'bout_id', right_on = 'bout_id')
    pred_data.rename(columns = {'won_diff': 'winner'}, inplace = True)
    pred_data.rename(columns = {'length_avg': 'length'}, inplace = True)

#    pred_data.drop('won_avg', axis = 1, inplace = True)
    
    pred_data_length = pred_data[[i for i in list(pred_data) if i != 'winner']]
    pred_data_winner = pred_data[[i for i in list(pred_data) if i != 'length']]
    
    pred_data_winner.set_index('bout_id', inplace = True)
    pred_data_winner.to_csv(os.path.join(cur_path, 'data', 'winner_data.csv'))
    pred_data_length.set_index('bout_id', inplace = True)
    pred_data_length.to_csv(os.path.join(cur_path, 'data', 'length_data.csv'))


def save_validation_data():
    pred_data_winner = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data.csv'))
    pred_data_winner.set_index('bout_id', inplace = True)
    pred_data_winner_validation = pred_data_winner.loc[pred_data_winner['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) < datetime(2019, 1, 1)]
    pred_data_winner_validation.to_csv(os.path.join(cur_path, 'data', 'winner_data_validation.csv'))
    pred_data_winner_test = pred_data_winner.loc[pred_data_winner['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) >= datetime(2019, 1, 1)]
    pred_data_winner_test.to_csv(os.path.join(cur_path, 'data', 'winner_data_test.csv'))
    
    pred_data_length = pd.read_csv(os.path.join(cur_path, 'data', 'length_data.csv'))
    pred_data_length.set_index('bout_id', inplace = True)
    pred_data_length_validation = pred_data_length.loc[pred_data_length['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) < datetime(2019, 1, 1)]
    pred_data_length_validation.to_csv(os.path.join(cur_path, 'data', 'length_data_validation.csv'))
    pred_data_length_test = pred_data_length.loc[pred_data_length['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) >= datetime(2019, 1, 1)]
    pred_data_length_test.to_csv(os.path.join(cur_path, 'data', 'length_data_test.csv'))



#if __name__ == '__main__':
#    stats = pull_stats()
#    avg_data = pull_avg_data(stats)
#    adj_data = pull_adj_data(avg_data, stats)
#    adj_avg_data = pull_adj_avg_data(adj_data)
#    pred_data_winner, pred_data_length = pull_pred_data(avg_data, adj_avg_data)
#    
#    adsfasdfaf
    
