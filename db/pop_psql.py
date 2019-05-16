import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(1, os.path.join(cur_path))

import requests
from lxml import html
from _connections import db_connection
from db.pg_tables import create_tables
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from progress_bar import progress
import re
import json
import numpy as np
from copy import deepcopy

PSQL = db_connection('psql')
cols = ['kd','ssa','sss','tsa','tss','sub','pas','rev','headssa','headsss','bodyssa', 'bodysss','legssa','legsss','disssa','dissss','clinssa','clinsss','gndssa','gndsss','tda','tds']


def pg_create_table(cur, table_name):  
#    cur, table_name = _psql, 
    try:
        # Truncate the table first
        for script in create_tables[table_name]:
            cur.execute(script)
            cur.execute("commit;")
        print("Created {}".format(table_name))
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        
        
def pg_query(cur, query):
    cur.execute(query)
    data = pd.DataFrame(cur.fetchall())
    return(data) 
    

def pg_drop(cur, query):
    cur.execute(query)
    cur.execute("commit;")

    
def pg_insert(cur, script):
    try:
        cur.execute(script)
        cur.execute("commit;")
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        raise(Exception)
        

def pull_stats():
    stats = pg_query(PSQL.client, 'SELECT bs.bout_id, date, fighter_id, kd, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa, legsss, disssa, dissss, clinssa, clinsss, gndssa, gndsss, tda, tds FROM ufc.bout_stats bs join ufc.bouts b on b.bout_id = bs.bout_id join ufc.fights f on f.fight_id = b.fight_id;')# where champ is false;')
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


#def pop_avg_data():
#    stats = pull_stats()
#    stat_avgs = {}
#    for fighter in stats.fighter_id.unique():
#        f_stats = {}
#    
#        ostat = stats.loc[stats['fighter_id'] == fighter]
#        ostat.sort_values('fight_date', inplace = True)
#    
#        dstat = stats.loc[stats['opponent_id'] == fighter]
#        dstat.sort_values('fight_date', inplace = True)
#        
#        if len(dstat) != len(ostat):
#            raise ValueError()
#            
#        if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
#            raise ValueError()
#            
#        for i in range(len(ostat)):
#            if i == 0:
#                continue
#            
#            f_stats[ostat.iloc[i]['bout_id']] = {'fight_date': ostat.iloc[i]['fight_date'], 'opponent_id': ostat.iloc[i]['opponent_id']}
#            for col in cols:
#                f_stats[ostat.iloc[i]['bout_id']]['avg_o_'+col] = ostat.iloc[:i][col].mean()  
#        for i in range(len(dstat)):
#            if i == 0:
#                continue
#            for col in cols:
#                f_stats[dstat.iloc[i]['bout_id']]['avg_d_'+col] = dstat.iloc[:i]['d_'+col].mean()
#                              
#        if len(f_stats.keys()) > 0:
#            stat_avgs[fighter] = f_stats
#            
#    avg_data = {}
#    i = 0
#    for k,v in stat_avgs.items():
#        for kk, vv in v.items():
#            vv['fighter_id'] = k
#            vv['bout_id'] = kk
#            avg_data[i] = vv
#            i += 1
#        
#    avg_data = pd.DataFrame.from_dict(avg_data).T
#    
#    for avg_d_bodyssa, avg_d_bodysss, avg_d_clinssa, avg_d_clinsss, avg_d_disssa,\
#     avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,avg_d_kd,avg_d_legssa,avg_d_legsss,\
#     avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds,\
#     avg_d_tsa, avg_d_tss, avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa,\
#     avg_o_dissss,avg_o_gndssa,avg_o_gndsss,avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss,\
#     avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss,bout_id,\
#     fight_date,fighter_id,opponent_id in avg_data.values:
#         
#        script = "INSERT INTO ufc.avg_stats(\
#        	fighter_id, bout_id, avg_o_kd, avg_o_ssa, avg_o_sss, avg_o_tsa, avg_o_tss, avg_o_sub, avg_o_pas, avg_o_rev,\
#            avg_o_headssa, avg_o_headsss, avg_o_bodyssa, avg_o_bodysss, avg_o_legssa, avg_o_legsss, avg_o_disssa, avg_o_dissss,\
#            avg_o_clinssa, avg_o_clinsss, avg_o_gndssa, avg_o_gndsss, avg_o_tda, avg_o_tds, avg_d_kd, avg_d_ssa, avg_d_sss,\
#            avg_d_tsa, avg_d_tss, avg_d_sub, avg_d_pas, avg_d_rev, avg_d_headssa, avg_d_headsss, avg_d_bodyssa, avg_d_bodysss,\
#            avg_d_legssa, avg_d_legsss, avg_d_disssa, avg_d_dissss, avg_d_clinssa, avg_d_clinsss, avg_d_gndssa, avg_d_gndsss, avg_d_tda, avg_d_tds)\
#        	VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
#            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
#            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, bout_id, avg_d_bodyssa, avg_d_bodysss, \
#            avg_d_clinssa, avg_d_clinsss, avg_d_disssa, avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,\
#            avg_d_kd,avg_d_legssa,avg_d_legsss, avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds, avg_d_tsa, avg_d_tss, \
#            avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa, avg_o_dissss,avg_o_gndssa,avg_o_gndsss,\
#            avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss, avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,\
#            avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss)
#        pg_insert(PSQL.client, script)
    

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


#def pop_adj_data():
#    stats = pull_stats()
#    avg_data = pull_avg_data()
#    
#    stat_adj = {}
#    for fighter in avg_data.fighter_id.unique():
#        
#        f_avgs = avg_data.loc[avg_data['fighter_id'] == fighter]
#        f_avgs.sort_values('fight_date', inplace = True)
#    
#        o_avgs = avg_data.loc[avg_data['opponent_id'] == fighter]
#        o_avgs.sort_values('fight_date', inplace = True)
#    
#        f_stats = stats.loc[stats['fighter_id'] == fighter]
#        f_stats.sort_values('fight_date', inplace = True)
#    
#        common_bouts = set([j for j in [i for i in f_avgs['bout_id'].values if i in o_avgs['bout_id'].values] if j in f_stats['bout_id'].values])
#        f_avgs = f_avgs.loc[f_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#        o_avgs = o_avgs.loc[o_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#        f_stats = f_stats.loc[f_stats['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
#    
#        adj_stats = f_avgs[['bout_id', 'fight_date', 'opponent_id']]     
#        for col in cols:                
#            adj_stats['adj_d_'+col] = (f_stats['d_'+col] /o_avgs['avg_o_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats['d_'+col]
#            adj_stats['adj_o_'+col] = (f_stats[col] / o_avgs['avg_d_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats[col]
#        if len(adj_stats) > 0:
#            stat_adj[fighter] = adj_stats.set_index('bout_id').T.to_dict()
#            
#    adj_data = {}
#    i = 0
#    for k,v in stat_adj.items():
#        for kk, vv in v.items():
#            vv['fighter_id'] = k
#            vv['bout_id'] = kk
#            adj_data[i] = vv
#            i += 1
#    
#    adj_data = pd.DataFrame.from_dict(adj_data).T
#
#    for adj_d_bodyssa, adj_d_bodysss,adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
#        adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
#        adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
#        adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
#        adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
#        adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
#        adj_o_tsa,adj_o_tss,bout_id,fight_date,fighter_id,opponent_id in adj_data.values:
#        
#            script = "INSERT INTO ufc.adj_stats(fighter_id, bout_id, adj_d_bodyssa, adj_d_bodysss,\
#                adj_d_clinssa, adj_d_clinsss, adj_d_disssa, adj_d_dissss, adj_d_gndssa, adj_d_gndsss,\
#                adj_d_headssa, adj_d_headsss, adj_d_kd, adj_d_legssa, adj_d_legsss, adj_d_pas, \
#                adj_d_rev, adj_d_ssa, adj_d_sss, adj_d_sub, adj_d_tda, adj_d_tds, adj_d_tsa, adj_d_tss,\
#                adj_o_bodyssa, adj_o_bodysss, adj_o_clinssa, adj_o_clinsss, adj_o_disssa, adj_o_dissss, \
#                adj_o_gndssa, adj_o_gndsss, adj_o_headssa, adj_o_headsss, adj_o_kd, adj_o_legssa, \
#                adj_o_legsss, adj_o_pas, adj_o_rev, adj_o_ssa, adj_o_sss, adj_o_sub, adj_o_tda, \
#                adj_o_tds, adj_o_tsa, adj_o_tss) VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, \
#                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, \
#                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, \
#                %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, \
#                bout_id, adj_d_bodyssa, adj_d_bodysss,adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
#                adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
#                adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
#                adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
#                adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
#                adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
#                adj_o_tsa,adj_o_tss)
#            try:
#                pg_insert(PSQL.client, script)
#            except:
#                pass


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


#def pop_adj_avg_data():
#    adj_data = pull_adj_data()
#    adj_stat_avgs = {}
#    for fighter in adj_data.fighter_id.unique():
#        f_stats = {}
#        
#        ostat = adj_data.loc[adj_data['fighter_id'] == fighter]
#        ostat.sort_values('fight_date', inplace = True)
#    
#        dstat = adj_data.loc[adj_data['opponent_id'] == fighter]
#        dstat.sort_values('fight_date', inplace = True)
#        
#        if len(dstat) != len(ostat):
#            raise ValueError()
#            
#        if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
#            raise ValueError()
#            
#        for i in range(len(ostat)):
#            if i == 0:
#                continue
#            
#            f_stats[ostat.iloc[i]['bout_id']] = {'fight_date': ostat.iloc[i]['fight_date'], 'opponent_id': ostat.iloc[i]['opponent_id']}
#            for col in cols:
#                f_stats[ostat.iloc[i]['bout_id']]['adj_avg_o_'+col] = ostat.iloc[:i]['adj_o_'+col].mean()  
#        for i in range(len(dstat)):
#            if i == 0:
#                continue
#            for col in cols:
#                f_stats[dstat.iloc[i]['bout_id']]['adj_avg_d_'+col] = dstat.iloc[:i]['adj_d_'+col].mean()
#                              
#        if len(f_stats.keys()) > 0:
#            adj_stat_avgs[fighter] = f_stats
#            
#    adj_avg_data = {}
#    i = 0
#    for k,v in adj_stat_avgs.items():
#        for kk, vv in v.items():
#            vv['fighter_id'] = k
#            vv['bout_id'] = kk
#            adj_avg_data[i] = vv
#            i += 1
#
#    adj_avg_data = pd.DataFrame.from_dict(adj_avg_data).T
#    
#    
#    for adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
#        adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,\
#        adj_avg_d_kd,adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,\
#        adj_avg_d_sss,adj_avg_d_sub,adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,\
#        adj_avg_o_bodyssa,adj_avg_o_bodysss,adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,\
#        adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,adj_avg_o_headssa,adj_avg_o_headsss,\
#        adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,adj_avg_o_rev,adj_avg_o_ssa,\
#        adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss,\
#        bout_id,fight_date,fighter_id,opponent_id in adj_avg_data.values:
#        
#        
#        script = "INSERT INTO ufc.adj_avg_stats(fighter_id, bout_id, adj_avg_d_bodyssa, adj_avg_d_bodysss,\
#            adj_avg_d_clinssa, adj_avg_d_clinsss, adj_avg_d_disssa, adj_avg_d_dissss, adj_avg_d_gndssa, \
#            adj_avg_d_gndsss, adj_avg_d_headssa, adj_avg_d_headsss, adj_avg_d_kd, adj_avg_d_legssa, adj_avg_d_legsss,\
#            adj_avg_d_pas, adj_avg_d_rev, adj_avg_d_ssa, adj_avg_d_sss, adj_avg_d_sub, adj_avg_d_tda, \
#            adj_avg_d_tds, adj_avg_d_tsa, adj_avg_d_tss, adj_avg_o_bodyssa, adj_avg_o_bodysss, adj_avg_o_clinssa,\
#            adj_avg_o_clinsss, adj_avg_o_disssa, adj_avg_o_dissss, adj_avg_o_gndssa, adj_avg_o_gndsss, \
#            adj_avg_o_headssa, adj_avg_o_headsss, adj_avg_o_kd, adj_avg_o_legssa, adj_avg_o_legsss, adj_avg_o_pas,\
#            adj_avg_o_rev, adj_avg_o_ssa, adj_avg_o_sss, adj_avg_o_sub, adj_avg_o_tda, adj_avg_o_tds, adj_avg_o_tsa,\
#            adj_avg_o_tss) VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
#            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
#            %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter_id, \
#            bout_id, adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
#            adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,adj_avg_d_kd,\
#            adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,adj_avg_d_sss,adj_avg_d_sub,\
#            adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,adj_avg_o_bodyssa,adj_avg_o_bodysss,\
#            adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,\
#            adj_avg_o_headssa,adj_avg_o_headsss,adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,\
#            adj_avg_o_rev,adj_avg_o_ssa,adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss)
#            
#        pg_insert(PSQL.client, script)


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
    
    pred_data_1 = {}
    hold_cols = ['bout_id', 'fighter_id', 'fight_date', 'opponent_id']
    for bout in data['bout_id'].unique():
        bout_data = data.loc[data['bout_id'] == bout]
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
        pred_data_1[bout] = bout_preds

    pred_data_1 = pd.DataFrame.from_dict(pred_data_1).T
    pred_data_1.reset_index(inplace = True)
    pred_data_1.rename(columns = {'index':'bout_id'}, inplace = True)
#    pred_data = pd.merge(pred_data, bout_len, left_on = 'bout_id', right_on = 'bout_id')
    pred_data_1.rename(columns = {'won_diff': 'winner'}, inplace = True)
    pred_data_1.rename(columns = {'length_avg': 'length'}, inplace = True)


    pred_data_2 = {}
    for bout in data['bout_id'].unique():
        bout_data = data.loc[data['bout_id'] == bout]
        if len(bout_data) != 2:
            continue
        bout_data = bout_data.iloc[[1,0]]
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
        pred_data_2[bout] = bout_preds
    pred_data_2 = pd.DataFrame.from_dict(pred_data_2).T
    pred_data_2.reset_index(inplace = True)
    pred_data_2.rename(columns = {'index':'bout_id'}, inplace = True)
#    pred_data = pd.merge(pred_data, bout_len, left_on = 'bout_id', right_on = 'bout_id')
    pred_data_2.rename(columns = {'won_diff': 'winner'}, inplace = True)
    pred_data_2.rename(columns = {'length_avg': 'length'}, inplace = True)
    pred_data = pred_data_1.append(pred_data_2)
#    pred_data.drop('won_avg', axis = 1, inplace = True)
#    [i for i in list(pred_data) if 'len' in i]
    pred_data_length = pred_data[[i for i in list(pred_data) if i != 'winner']]
    pred_data_winner = pred_data[[i for i in list(pred_data) if i != 'length']]
    pred_data_winner.set_index('bout_id', inplace = True)
    pred_data_winner.to_csv(os.path.join(cur_path, 'data', 'winner_data.csv'))
#    pred_data_length.set_index('bout_id', inplace = True)
#    pred_data_length.to_csv(os.path.join(cur_path, 'data', 'length_data.csv'))


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


def odds_converter(odds):
#    odds = win_odds
    if odds > 0:
        imp_prob = 100 / (odds +100)
    elif odds < 0:
        imp_prob = (odds * -1) / ((-1 * odds)+100)
    return(imp_prob)


def pop_winner_odds():
    all_fights = pg_query(PSQL.client, "select f.fight_id, date, br.bout_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id full join ufc.bout_results br on br.bout_id = b.bout_id full join ufc.winner_consensus_odds wco on wco.bout_id = br.bout_id where date > '1-1-2002' and br.bout_id is not NULL and wco.bout_id is NULL;")
    fight_meta = all_fights[[0, 1]].drop_duplicates()

    cur_bouts = set(pg_query(PSQL.client, "SELECT bout_id from ufc.winner_consensus_odds;")[0])
    
    all_fighters = pg_query(PSQL.client, "select name, fighter_id from ufc.fighters;")
    all_fighters = {k:v for k,v in all_fighters.values}
    
    tot_fights = len(fight_meta)
    for fight_num, (fight_id, date) in enumerate(fight_meta.values):
        progress(fight_num + 1, tot_fights)  
        
        url = 'https://www.sportsbookreview.com/betting-odds/ufc/?date=%s' % (str(date).split(' ')[0].replace('-',''))
        page = requests.get(url)
        if page.status_code == 404:
            raise ValueError()  
    
        soup = BeautifulSoup(page.content, 'html.parser')        

        script = soup.find("script", text=re.compile("window.__INITIAL_STATE__")).text
        data = json.loads(re.search("window.__INITIAL_STATE__+=+(\{.*\})", script).group(1))
        
        poss_bouts = pg_query(PSQL.client, "select bx.bout_id, fighter_id, opponent_id from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id where b.fight_id = '%s';" % (fight_id))
        poss_bouts.columns = ['bout_id', 'fighter_id', 'opponent_id']
        odds = data['events']['events']
        fight_odds = {}
        for k,v in odds.items():
            
            bout_odds = {}
            for i,(fighter) in enumerate(v['des'].split('@')):
                if fighter not in all_fighters.keys():
                    continue
                bout_odds[i] = {'name': fighter, 'fighter_id': all_fighters[fighter], 'odds':[]}
            if len(bout_odds.keys()) != 2:
                continue
            for vv in v['currentLines'].values():
                if not isinstance(vv, list):
                    continue
                for kk in bout_odds.keys():
                    bout_odds[kk]['odds'].append(odds_converter(int(vv[kk]['ap'])))
            for kk in bout_odds.keys():
                bout_odds[kk]['odds'] = np.mean(bout_odds[kk]['odds'])
            if bout_odds[0]['odds'] != bout_odds[0]['odds']:
                continue
            if bout_odds[1]['odds'] != bout_odds[1]['odds']:
                continue
            try:
                fight_odds[poss_bouts.loc[(poss_bouts['fighter_id'] == bout_odds[0]['fighter_id']) & (poss_bouts['opponent_id'] == bout_odds[1]['fighter_id']), 'bout_id'].values[0]] = bout_odds
            except IndexError:
                continue
                
        imp_odds = {}
        odds_bouts = fight_odds.keys()
        for bout in odds_bouts:
            for vvv in fight_odds[bout].values():
                imp_odds[vvv['fighter_id']] = {'bout_id': bout, 'odds': vvv['odds']}
        imp_odds = pd.DataFrame.from_dict(imp_odds).T.reset_index()
        
        for fighter, bout_idx, prob in imp_odds.values:
            if bout_idx in cur_bouts:
                continue
            script = "INSERT INTO ufc.winner_consensus_odds(\
                	bout_id, fighter_id, imp_prob)\
                	VALUES ('%s', '%s', %.4f);" % (bout_idx, fighter, prob)
            pg_insert(PSQL.client, script)
            
        
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        
        
def pop_bout_fighter_xref():    
    cur_bouts = pg_query(PSQL.client, "select distinct(bout_id) from ufc.bout_fighter_xref")
    cur_bouts = set(cur_bouts[0].values)
    fights = pg_query(PSQL.client, "select distinct(fight_id) from ufc.bouts b full join ufc.bout_fighter_xref bx on bx.bout_id = b.bout_id where bx.bout_id is NULL ")
    tot_fights = len(fights.values)
    
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters")
    all_fighters = set([i[0] for i in all_fighters.values])
    for fight_num, (fight_id) in enumerate(fights.values):
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id[0])
        page = requests.get(url)
        tree = html.fromstring(page.content)
    
        fighter_profs = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[2]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
        
        bout_ids = tree.xpath('/html/body/section/div/div/table/tbody/tr/@data-link')
        bout_ids = [i.replace('http://www.ufcstats.com/fight-details/','') for i in bout_ids]
        
        if len(fighter_profs) != 2*len(bout_ids):
            raise ValueError()
            
        fighter_profs = list(divide_chunks(fighter_profs, 2))
        
        for fighters, bout in zip(fighter_profs, bout_ids):
            if bout in cur_bouts:
                continue
            
            if fighters[0] not in all_fighters or fighters[1] not in all_fighters:
                continue
            
            script = "INSERT INTO ufc.bout_fighter_xref(\
                        bout_id, fighter_id, opponent_id)\
                        	VALUES ('%s', '%s', '%s');" % (bout, fighters[0], fighters[1])
            pg_insert(PSQL.client, script)

            script = "INSERT INTO ufc.bout_fighter_xref(\
                        bout_id, fighter_id, opponent_id)\
                        	VALUES ('%s', '%s', '%s');" % (bout, fighters[1], fighters[0])
            pg_insert(PSQL.client, script)   
            cur_bouts.add(bout)
        progress(fight_num + 1, tot_fights)  
    
        
def fighter_wiki_decode(input_val):
#    input_val = val
    try:
        output_val = datetime.strptime(input_val, '%B %d, %Y')
        output_key = 'born'
        return(output_key, output_val)
    except:
        if 'ft' in input_val and 'in' in input_val and 'm' in input_val:
            heights = {'feet': False, 'inches': False}
            height_splits = input_val.split(' ')
            for ht in height_splits:
                if 'ft' in ht:
                    heights['feet'] = int(ht.encode('ascii', 'ignore').decode().replace('ft', '').replace('(','').replace(')', ''))
                elif 'in' in ht:
                    heights['inches'] = int(ht.encode('ascii', 'ignore').decode().replace('in', '').replace('(','').replace(')', ''))
            output_val = (heights['feet'] * 12) + heights['inches']
            output_key = 'height'
            return(output_key, output_val)
        elif 'in' in input_val and 'cm' in input_val:
            reach_splits = input_val.split(' ')  
            for rch in reach_splits:
                if 'in' in rch:
                    try:
                        output_val = int(float(rch.encode('ascii', 'ignore').decode().replace('in', '')))
                        output_key = 'reach'
                        return(output_key, output_val)
                    except:
                        pass
                elif 'cm' in rch:
#                    rch = reach_splits[1]
                    cm = int(float(rch.encode('ascii', 'ignore').decode().replace('cm', '').replace('(','').replace(')', '')))
                    output_val = round(0.393701 * cm)
                    output_key = 'reach'
                    return(output_key, output_val)    
    return(False, False)
    

def add_fighter_bio(fighter, _all_fighters):            
#    fighter, _all_fighters = fighter[1], all_fighters
    f_url = 'http://www.ufcstats.com/fighter-details/%s' % (fighter)
    f_page = requests.get(f_url)
    f_tree = html.fromstring(f_page.content)
    if f_page.status_code == 404:
        print('404')
    f_name = f_tree.xpath('/html/body/section/div/h2/span[1]/text()')
    f_name = f_name[0].strip()
    fighter_data = {'height': False,
                    'reach': False,
                    'born': False
                    }
    try:
        f_height = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[1]/text()')
        f_height_inches = int(f_height[1].strip().split("'")[1].strip().replace('"', ''))
        f_height_feet = int(f_height[1].strip().split("'")[0].strip())
        f_height = f_height_feet * 12 + f_height_inches
        fighter_data['height'] = f_height
    except :
        pass
    try:
        f_reach = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[3]/text()')
        f_reach = int(f_reach[1].strip().replace('"',''))
        fighter_data['reach'] = f_reach
    except ValueError:
        pass
    try:
        f_dob = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[5]/text()')
        f_dob = datetime.strptime(f_dob[1].strip(), '%b %d, %Y')
        fighter_data['born'] = f_dob
    except:
        pass
    
    if (not fighter_data['height'] or not fighter_data['reach'] or not fighter_data['born']):
        lookup_url = 'https://en.wikipedia.org/wiki/%s' % (f_name.replace(' ', '_'))
        lookup_page = requests.get(lookup_url)
        lookup_tree = html.fromstring(lookup_page.content)
        
        values = lookup_tree.xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr/td/text()')
        values = [i.strip() for i in values if i.strip() not in [',','']]
        
#        if values == ['Please', 'to check for alternative titles or spellings.'] or len(values) < 5:
#            return(_all_fighters, False)
            
        kv_data = {}
        for val in values:
            k,v = fighter_wiki_decode(val)
            if isinstance(k, str):
                kv_data[k] = v                
    
        try:
            for key in ['height', 'reach', 'born']:
                if not fighter_data[key]:
                    fighter_data[key] = kv_data[key]
        except KeyError:
            return(_all_fighters, False)
                                
    f_stance = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[4]/text()')
    f_stance = f_stance[1].strip()
    script = "INSERT INTO ufc.fighters(\
                fighter_id, name, height, reach, stance, dob)\
                VALUES ('%s', '%s', %i, %i, '%s', '%s');" % (fighter, f_name.replace("'",''), fighter_data['height'], fighter_data['reach'], f_stance, fighter_data['born'])
    pg_insert(PSQL.client, script)
    _all_fighters.add(fighter)
    return(_all_fighters, True)
        

    
def pop_bout_stats():
    all_bouts = pg_query(PSQL.client, 'select b.bout_id from ufc.bout_results b full join ufc.bout_stats bs on bs.bout_id = b.bout_id where bs.bout_id is NULL')
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values]) 
    tot_bouts = len(all_bouts.values)
    for bout_num, (bout) in enumerate(all_bouts.values):
        url = 'http://www.ufcstats.com/fight-details/%s' % (bout[0])
        page = requests.get(url)
        if page.status_code != 200:
            raise ValueError()
        tree = html.fromstring(page.content)   
        
        kd = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[2]/p/text()')
        kd = [int(i.strip()) for i in kd]
        
        ss = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[3]/p/text()')
        ss = [i.strip() for i in ss]  
        ssa = [int(i.split(' of ')[1]) for i in ss]
        sss = [int(i.split(' of ')[0]) for i in ss]
    
        ts = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[5]/p/text()')
        ts = [i.strip() for i in ts]  
        tsa = [int(i.split(' of ')[1]) for i in ts]
        tss = [int(i.split(' of ')[0]) for i in ts]
        
        td = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[6]/p/text()')
        td = [i.strip() for i in td]  
        tda = [int(i.split(' of ')[1]) for i in td]
        tds = [int(i.split(' of ')[0]) for i in td]   
        
        sub = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[8]/p/text()')
        sub = [i.strip() for i in sub]  
    
        pas = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[9]/p/text()')
        pas = [i.strip() for i in pas] 
        
        rev = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[9]/p/text()')
        rev = [i.strip() for i in rev] 
        
        headss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[4]/p/text()')
        headss = [i.strip() for i in headss]  
        headssa = [int(i.split(' of ')[1]) for i in headss]
        headsss = [int(i.split(' of ')[0]) for i in headss]
        
        bodyss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[5]/p/text()')
        bodyss = [i.strip() for i in bodyss]  
        bodyssa = [int(i.split(' of ')[1]) for i in bodyss]
        bodysss = [int(i.split(' of ')[0]) for i in bodyss]
    
        legss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[6]/p/text()')
        legss = [i.strip() for i in legss]  
        legssa = [int(i.split(' of ')[1]) for i in legss]
        legsss = [int(i.split(' of ')[0]) for i in legss]    
        
        distss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        distss = [i.strip() for i in distss]  
        distssa = [int(i.split(' of ')[1]) for i in distss]
        distsss = [int(i.split(' of ')[0]) for i in distss]   
        
        clinss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p/text()')
        clinss = [i.strip() for i in clinss]  
        clinssa = [int(i.split(' of ')[1]) for i in clinss]
        clinsss = [int(i.split(' of ')[0]) for i in clinss] 
    
        gndss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[9]/p/text()')
        clinss = [i.strip() for i in gndss]  
        gndssa = [int(i.split(' of ')[1]) for i in gndss]
        gndsss = [int(i.split(' of ')[0]) for i in gndss] 
        
        fighter_profs = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[1]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
        
        for _kd, _fighter_profs, _ssa, _sss, _tsa, _tss, _sub, _pas, _rev, _headssa, _headsss, _bodyssa, _bodysss, _legssa, _legsss, _distssa, _distsss, _clinssa, _clinsss, _gndssa, _gndsss, _tda, _tds in zip(kd, fighter_profs, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa,legsss, distssa, distsss, clinssa, clinsss, gndssa, gndsss, tda, tds):
            if _fighter_profs not in all_fighters:
                added = False
                all_fighters, added = add_fighter_bio(_fighter_profs, all_fighters)
                if not added:
#                    print('Skipped')
                    continue
                
            script = "INSERT INTO ufc.bout_stats(\
                    bout_id, fighter_id, kd, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa, legsss, disssa, dissss, clinssa, clinsss, gndssa, gndsss, tda, tds) \
                    	VALUES ('%s', '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);" % (bout[0], _fighter_profs, _kd, _ssa, _sss, _tsa, _tss, _sub, _pas, _rev, _headssa, _headsss, _bodyssa, _bodysss, _legssa, _legsss, _distssa, _distsss, _clinssa, _clinsss, _gndssa, _gndsss, _tda, _tds)
            pg_insert(PSQL.client, script) 
#            print('Added %s' % (bout[0]))
        progress(bout_num + 1, tot_bouts)  
            

def pop_bout_res():
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id full join ufc.bout_results br on br.bout_id = b.bout_id where date > '1-1-2002' and br.bout_id is NULL;")
    
#    modern_fights = pg_query(PSQL.client, "select bs.bout_id from ufc.bout_results br full join ufc.bout_stats bs on bs.bout_id = br.bout_id where br.bout_id is Null")
    
    method_dict = pg_query(PSQL.client, "select * from ufc.methods;")
    method_dict = {v:k for k,v in method_dict.values}
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values])    
    
    all_bouts = pg_query(PSQL.client, "select bout_id from ufc.bout_results")
    all_bouts = set([i[0] for i in all_bouts.values])
    tot_fights = len(modern_fights[0].unique())
    for fight_num, (fight_id) in enumerate(modern_fights[0].unique()):
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        if page.status_code != 200:
            print('Request denied')
            continue
        tree = html.fromstring(page.content)
    
        fighter_profs = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[2]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
            
        methods = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p[1]/text()')
        methods = [i.strip() for i in methods]
        
        rounds = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[9]/p/text()')
        try:
            rounds = [int(i.strip()) for i in rounds]
        except ValueError:
            continue
        times = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[10]/p/text()')
        times = [i.strip() for i in times]
        minutes = [int(i.split(':')[0]) for i in times]
        seconds = [int(i.split(':')[1]) for i in times]
        
        times = [i*60 + j for i,j in zip(minutes, seconds)]
        lengths =  [(i-1)*300 + j for i,j in zip(rounds, times)]
        
        bouts = tree.xpath('/html/body/section/div/div/table/tbody/tr/@data-link')
        bouts = [i.replace('http://www.ufcstats.com/fight-details/', '') for i in bouts]   
        
        fighters = [fighter_profs[i: i+2] for i in range(0, len(fighter_profs), 2)]
        
        for fighter, bout, method, rnd, time, length in zip(fighters, bouts, methods, rounds, times, lengths):        
            if fighter[0] not in all_fighters:
                added = False
                all_fighters, added = add_fighter_bio(fighter[0], all_fighters)
                if not added:
#                    print('Skipped')
                    continue
            if fighter[1] not in all_fighters:
                added = False
                all_fighters, added = add_fighter_bio(fighter[1], all_fighters)
                if not added:
#                    print('Skipped')
                    continue
                
            if bout in all_bouts:
                continue
            script = "INSERT INTO ufc.bout_results(\
                        	bout_id, winner, loser, method_id, rounds, time, length)\
                        	VALUES ('%s', '%s', '%s', %i, %i, %i, %i);" % (bout, fighter[0], fighter[1], method_dict[method], rnd, time, length)
            pg_insert(PSQL.client, script)
#            print('Added %s' % (bout))
        progress(fight_num + 1, tot_fights)  
    
    
            
def pop_bouts():
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
#    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f where date > '1-1-2002';")
    modern_fights.drop_duplicates(inplace = True)
    wc_dict = pg_query(PSQL.client, "select * from ufc.weights;")
    wc_dict = {v:k for k,v in wc_dict.values}
    current_bouts = pg_query(PSQL.client, "select bout_id from ufc.bouts;")
    current_bouts = set(current_bouts[0].values)
    champ_icon = 'http://1e49bc5171d173577ecd-1323f4090557a33db01577564f60846c.r80.cf1.rackcdn.com/belt.png'
    for fight_id in modern_fights[0].values:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        if page.status_code != 200:
            raise ValueError()
        tree = html.fromstring(page.content)
        weight_classes = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        weight_classes = [i.strip() for i in weight_classes if i.strip() != '']
            
        champ_fights = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/img/@src')
        champ_fights = len([i for i in champ_fights if i ==  champ_icon])
        
        bouts = tree.xpath('/html/body/section/div/div/table/tbody/tr/@data-link')
        bouts = [i.replace('http://www.ufcstats.com/fight-details/', '') for i in bouts]   
        
        for i, (bout, weight) in enumerate(zip(bouts, weight_classes)):
            if i <= champ_fights:
                champ = 'TRUE'
            else:
                champ = 'FALSE'
            
            if bout in current_bouts:
                continue

            script = "INSERT INTO ufc.bouts(\
                        	bout_id, fight_id, weight_id, champ)\
                        	VALUES ('%s', '%s', %i, %s);" % (bout, fight_id, wc_dict[weight.replace("'", '')], champ)
            pg_insert(PSQL.client, script) 
            print('Added %s' % (bout))


def pop_wc_meth():
    all_weight_class = []
    all_methods = []
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values])
    for fight_id in modern_fights[0].values[1:]:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
        
        weight_classes = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        weight_classes = [i.strip() for i in weight_classes if i.strip() != '']
            
        methods = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p[1]/text()')
        methods = [i.strip() for i in methods]
        
        for method in methods:
            all_methods.append(method)
        for w_class in weight_classes:
            all_weight_class.append(w_class)
            
    wc_set = set(all_weight_class)
    meth_set = set(all_methods)
    
    wc_id = 0
    
    cur_wc = pg_query(PSQL.client, 'Select weight_id, weight_desc from ufc.weights')
    wc_dict = {k:j for j,k in cur_wc.values}
    wc_id = max(wc_dict.values()) + 1
    for wc in wc_set:
        if wc in wc_dict.keys():
            continue
        script = "INSERT INTO ufc.weights(\
                    	weight_id, weight_desc)\
                    	VALUES (%i, '%s');" % (wc_id, wc.replace("'",''))
        pg_insert(PSQL.client, script)        
        wc_id += 1
        
    cur_meth = pg_query(PSQL.client, 'Select method_id, method_desc from ufc.methods')
    meth_dict = {k:j for j,k in cur_meth.values}
    meth_id = max(meth_dict.values()) + 1
    for meth in meth_set:
        if meth in meth_dict.keys() or meth == '':
            continue
        script = "INSERT INTO ufc.methods(\
                    	method_id, method_desc)\
                    	VALUES (%i, '%s');" % (meth_id, meth)
        pg_insert(PSQL.client, script)        
        meth_id += 1

    
def pop_fighters():
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = [i for i in all_fighters[0].values]
    for fight_id in modern_fights[0].values:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
    
        fighter_profs = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[2]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
        
        for fighter in fighter_profs:
            if fighter in all_fighters:
                continue
            
            f_url = 'http://www.ufcstats.com/fighter-details/%s' % (fighter)
            f_page = requests.get(f_url)
            f_tree = html.fromstring(f_page.content)
            if f_page.status_code == 404:
                print('404')
                continue
            f_name = f_tree.xpath('/html/body/section/div/h2/span[1]/text()')
#            f_tree.xpath('/html/body/section/div/h2/span[1]/text()')
            f_name = f_name[0].strip()
            f_height = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[1]/text()')
            try:
                f_height_inches = int(f_height[1].strip().split("'")[1].strip().replace('"', ''))
            except:
                all_fighters.append(fighter)
                continue
            f_height_feet = int(f_height[1].strip().split("'")[0].strip())
            f_height = f_height_feet * 12 + f_height_inches
            f_reach = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[3]/text()')
            try:
                f_reach = int(f_reach[1].strip().replace('"',''))
            except:
                all_fighters.append(fighter)
                continue
            f_stance = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[4]/text()')
            f_stance = f_stance[1].strip()
            
            f_dob = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[5]/text()')
            try:
                f_dob = datetime.strptime(f_dob[1].strip(), '%b %d, %Y')
            except:
                all_fighters.append(fighter)
                continue            
            
            script = "INSERT INTO ufc.fighters(\
                        fighter_id, name, height, reach, stance, dob)\
                        VALUES ('%s', '%s', %i, %i, '%s', '%s');" % (fighter, f_name.replace("'",''), f_height, f_reach, f_stance, f_dob)
            pg_insert(PSQL.client, script)
            all_fighters.append(fighter)
        
        
def pop_fights():
    #   pg_create_table(PSQL.client, 'winner_consensus_odds')
    #http://www.ufcstats.com/event-details/
    url = 'http://www.ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    tree = html.fromstring(page.content)
    
    ids = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/a/@href')
    ids = [i.replace('http://www.ufcstats.com/event-details/', '') for i in ids]
    names = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/a/text()')
    names = [i.strip() for i in names]
    locs = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[2]/text()')
    locs = [i.strip() for i in locs]
    countries = [i.split(',')[-1].strip() for i in locs]
    states = [i.split(',')[1].strip() if len(i.split(',')) == 3 else '' for i in locs]
    cities = [i.split(',')[0].strip() for i in locs]
    dates = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/span/text()')
    dates = [i.strip() for i in dates]
    dates = [datetime.strptime(i, '%B %d, %Y') for i in dates]
    
    cur_countries = pg_query(PSQL.client, 'Select * from ufc.countries')
    country_dict = {k:j for j,k in cur_countries.values}
    country_id = max(country_dict.values()) + 1
    for country in countries:
        if country in country_dict.keys():
            continue
        script = "INSERT INTO ufc.countries(\
        	country_id, country_name)\
        	VALUES (%i, '%s');" % (country_id, country)
        pg_insert(PSQL.client, script)
        country_dict[country] = country_id
        country_id += 1
        
        
    cur_states = pg_query(PSQL.client, 'Select state_id, state_name from ufc.states')
    state_dict = {k:j for j,k in cur_states.values}
    state_id = max(state_dict.values()) + 1
    for state, country in zip(states, countries):
        if state == '':
            continue
        if state in state_dict.keys():
            continue
        script = "INSERT INTO ufc.states(\
        	state_id, state_name, country_id)\
        	VALUES (%i, '%s', %i);" % (state_id, state, country_dict[country])
        pg_insert(PSQL.client, script)
        state_dict[state] = state_id
        state_id += 1  
        
    
    cur_cities = pg_query(PSQL.client, 'Select city_id, city_name from ufc.cities')
    city_dict = {k:j for j,k in cur_cities.values}
    city_id = max(city_dict.values()) + 1
    for city, state, country in zip(cities, states, countries):
        if city in city_dict.keys():
            continue
        if state == '':
            use_state = 'Null'
        else:
            use_state = state_dict[state]
        script = "INSERT INTO ufc.cities(\
        	city_id, city_name, country_id, state_id)\
        	VALUES (%i, '%s', %i, %s);" % (city_id, city, country_dict[country], use_state)
        pg_insert(PSQL.client, script)
        city_dict[city] = city_id
        city_id += 1  
    
    
    cur_fights = pg_query(PSQL.client, 'Select fight_id from ufc.fights')
    cur_fights = set([i[0] for i in cur_fights.values])
    for fight_id, name, country, state, city, date in zip(ids, names, countries, states, cities, dates):
        if fight_id in cur_fights:
            continue
        if state == '':
            use_state = 'Null'
        else:
            use_state = state_dict[state]    
        
        print('Adding %s' % (name))
        script = "INSERT INTO ufc.fights(\
                	fight_id, name, country_id, state_id, city_id, date)\
                	VALUES ('%s', '%s', %i, %s, %s, '%s');" % (fight_id, name.replace("'",''), country_dict[country], use_state, city_dict[city], date)
        pg_insert(PSQL.client, script)


def update_avg_data():
    print('Updating Average Stats')

    stats = pull_stats()    
    missing_bouts = pg_query(PSQL.client, "select b.bout_id from ufc.bouts b full join ufc.avg_stats avs on b.bout_id = avs.bout_id join ufc.fights f on f.fight_id = b.fight_id where avs.bout_id is NULL and date < NOW() + INTERVAL '1 day'")
    fight_dates = pg_query(PSQL.client, "SELECT fight_id, date from ufc.fights;")
    fight_dates = {k:v for k,v in fight_dates.values}
    tot_bouts = len(missing_bouts)
    for bout_num, (bout_id) in enumerate(missing_bouts.values):  
        progress(bout_num + 1, tot_bouts)  
        bout_info = pg_query(PSQL.client, "select bfx.bout_id, fighter_id, opponent_id, fight_id from ufc.bout_fighter_xref bfx join ufc.bouts b on bfx.bout_id = b.bout_id where bfx.bout_id = '%s';" % (bout_id[0]))
        if len(bout_info) != 2:
            continue
        bout_info.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_id']      
        for bout, fighter, opponent, fight in bout_info.values:
            fight_date = fight_dates[fight]
            f_stats = {}
            ostat = stats.loc[(stats['fighter_id'] == fighter) & (stats['fight_date'] < fight_date)]
            ostat.sort_values('fight_date', inplace = True)
            dstat = stats.loc[(stats['opponent_id'] == fighter) & (stats['fight_date'] < fight_date)]
            dstat.sort_values('fight_date', inplace = True)
            if len(dstat) != len(ostat):
                raise ValueError()
            if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
                raise ValueError()
            if len(ostat) == 0:
                continue
            f_stats[bout] = {'fight_date': fight_date, 'opponent_id': opponent}
            for col in cols:
                f_stats[bout]['avg_o_'+col] = ostat[col].mean()  
            for col in cols:
                f_stats[bout]['avg_d_'+col] = dstat['d_'+col].mean()
            if len(f_stats.keys()) == 0:
                continue
            avg_data = pd.DataFrame.from_dict(f_stats).T

            for avg_d_bodyssa, avg_d_bodysss, avg_d_clinssa, avg_d_clinsss, avg_d_disssa,\
             avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,avg_d_kd,avg_d_legssa,avg_d_legsss,\
             avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds,\
             avg_d_tsa, avg_d_tss, avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa,\
             avg_o_dissss,avg_o_gndssa,avg_o_gndsss,avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss,\
             avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss,\
             fight_date,opponent_id in avg_data.values:
                 
                script = "INSERT INTO ufc.avg_stats(\
                	fighter_id, bout_id, avg_o_kd, avg_o_ssa, avg_o_sss, avg_o_tsa, avg_o_tss, avg_o_sub, avg_o_pas, avg_o_rev,\
                    avg_o_headssa, avg_o_headsss, avg_o_bodyssa, avg_o_bodysss, avg_o_legssa, avg_o_legsss, avg_o_disssa, avg_o_dissss,\
                    avg_o_clinssa, avg_o_clinsss, avg_o_gndssa, avg_o_gndsss, avg_o_tda, avg_o_tds, avg_d_kd, avg_d_ssa, avg_d_sss,\
                    avg_d_tsa, avg_d_tss, avg_d_sub, avg_d_pas, avg_d_rev, avg_d_headssa, avg_d_headsss, avg_d_bodyssa, avg_d_bodysss,\
                    avg_d_legssa, avg_d_legsss, avg_d_disssa, avg_d_dissss, avg_d_clinssa, avg_d_clinsss, avg_d_gndssa, avg_d_gndsss, avg_d_tda, avg_d_tds)\
                	VALUES ('%s', '%s', %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
                    %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f,\
                    %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter, bout, avg_d_bodyssa, avg_d_bodysss, \
                    avg_d_clinssa, avg_d_clinsss, avg_d_disssa, avg_d_dissss,avg_d_gndssa, avg_d_gndsss,avg_d_headssa,avg_d_headsss,\
                    avg_d_kd,avg_d_legssa,avg_d_legsss, avg_d_pas,avg_d_rev,avg_d_ssa,avg_d_sss,avg_d_sub,avg_d_tda,avg_d_tds, avg_d_tsa, avg_d_tss, \
                    avg_o_bodyssa, avg_o_bodysss, avg_o_clinssa, avg_o_clinsss, avg_o_disssa, avg_o_dissss,avg_o_gndssa,avg_o_gndsss,\
                    avg_o_headssa,avg_o_headsss,avg_o_kd,avg_o_legssa,avg_o_legsss, avg_o_pas,avg_o_rev,avg_o_ssa,avg_o_sss,\
                    avg_o_sub,avg_o_tda,avg_o_tds,avg_o_tsa,avg_o_tss)
                pg_insert(PSQL.client, script)
                                                            
        
def update_adj_data():
    print('Updating Adjusted Stats')
    stats = pull_stats()
    avg_data = pull_avg_data()
    missing_bouts = pg_query(PSQL.client, "select b.bout_id from ufc.bouts b full join ufc.adj_stats avs on b.bout_id = avs.bout_id join ufc.fights f on f.fight_id = b.fight_id where avs.bout_id is NULL and date < NOW() + INTERVAL '1 day'")
    fight_dates = pg_query(PSQL.client, "SELECT fight_id, date from ufc.fights;")
    fight_dates = {k:v for k,v in fight_dates.values}
    tot_bouts = len(missing_bouts)
    for bout_num, (bout_id) in enumerate(missing_bouts.values):  
#        if bout_id == 'bf719cf83cab229b':
#            asdfasdf
        progress(bout_num + 1, tot_bouts)  
        bout_info = pg_query(PSQL.client, "select bfx.bout_id, fighter_id, opponent_id, fight_id from ufc.bout_fighter_xref bfx join ufc.bouts b on bfx.bout_id = b.bout_id where bfx.bout_id = '%s';" % (bout_id[0]))
        if len(bout_info) != 2:
            continue
        bout_info.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_id']      
        for bout, fighter, opponent, fight in bout_info.values:
            fight_date = fight_dates[fight]
            f_stats = {}
            f_avgs = avg_data.loc[(avg_data['fighter_id'] == fighter) & (avg_data['fight_date'] == fight_date)]
            f_avgs.sort_values('fight_date', inplace = True)
            o_avgs = avg_data.loc[(avg_data['opponent_id'] == fighter) & (avg_data['fight_date'] == fight_date)]
            o_avgs.sort_values('fight_date', inplace = True)
            f_stats = stats.loc[(stats['fighter_id'] == fighter) & (stats['fight_date'] == fight_date)]
            f_stats.sort_values('fight_date', inplace = True)
            common_bouts = set([j for j in [i for i in f_avgs['bout_id'].values if i in o_avgs['bout_id'].values] if j in f_stats['bout_id'].values])
            f_avgs = f_avgs.loc[f_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
            o_avgs = o_avgs.loc[o_avgs['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
            f_stats = f_stats.loc[f_stats['bout_id'].apply(lambda x: True if x in common_bouts else False)].reset_index(drop = True)
            adj_stats = f_avgs[['bout_id', 'fight_date', 'opponent_id']]     
            for col in cols:                
                adj_stats['adj_d_'+col] = (f_stats['d_'+col] /o_avgs['avg_o_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats['d_'+col]
                adj_stats['adj_o_'+col] = (f_stats[col] / o_avgs['avg_d_'+col]).apply(lambda x: x if x == x and x not in [np.inf, -np.inf] else 1) * f_stats[col]
            if len(adj_stats) == 0:
                continue
            for bout_id,fight_date,opponent_id, adj_d_bodyssa, adj_d_bodysss,\
                adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
                adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
                adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
                adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
                adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
                adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
                adj_o_tsa,adj_o_tss in adj_stats.values:
                
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
                        %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter, \
                        bout_id, adj_d_bodyssa, adj_d_bodysss,adj_d_clinssa,adj_d_clinsss,adj_d_disssa,adj_d_dissss,\
                        adj_d_gndssa,adj_d_gndsss,adj_d_headssa,adj_d_headsss,adj_d_kd,adj_d_legssa,\
                        adj_d_legsss,adj_d_pas,adj_d_rev,adj_d_ssa,adj_d_sss,adj_d_sub,adj_d_tda,adj_d_tds,\
                        adj_d_tsa,adj_d_tss,adj_o_bodyssa,adj_o_bodysss,adj_o_clinssa,adj_o_clinsss,adj_o_disssa,\
                        adj_o_dissss,adj_o_gndssa,adj_o_gndsss,adj_o_headssa,adj_o_headsss,adj_o_kd,adj_o_legssa,\
                        adj_o_legsss,adj_o_pas,adj_o_rev,adj_o_ssa,adj_o_sss,adj_o_sub,adj_o_tda,adj_o_tds,\
                        adj_o_tsa,adj_o_tss)
                    pg_insert(PSQL.client, script)                
        
        
def update_adj_avg_data():
    print('Updating Average Adjusted Stats')
    
    adj_data = pull_adj_data()
    missing_bouts = pg_query(PSQL.client, "select b.bout_id from ufc.bouts b full join ufc.adj_avg_stats avs on b.bout_id = avs.bout_id join ufc.fights f on f.fight_id = b.fight_id where avs.bout_id is NULL and date < NOW() + INTERVAL '1 day'")
    fight_dates = pg_query(PSQL.client, "SELECT fight_id, date from ufc.fights;")
    fight_dates = {k:v for k,v in fight_dates.values}
    tot_bouts = len(missing_bouts)
    for bout_num, (bout_id) in enumerate(missing_bouts.values):  
        progress(bout_num + 1, tot_bouts)  
        bout_info = pg_query(PSQL.client, "select bfx.bout_id, fighter_id, opponent_id, fight_id from ufc.bout_fighter_xref bfx join ufc.bouts b on bfx.bout_id = b.bout_id where bfx.bout_id = '%s';" % (bout_id[0]))
        if len(bout_info) != 2:
            continue
        bout_info.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_id']      
        for bout, fighter, opponent, fight in bout_info.values:
            fight_date = fight_dates[fight]
            f_stats = {}            
            ostat = adj_data.loc[(adj_data['fighter_id'] == fighter) & (adj_data['fight_date'] < fight_date)]
            ostat.sort_values('fight_date', inplace = True)        
            dstat = adj_data.loc[(adj_data['opponent_id'] == fighter) & (adj_data['fight_date'] < fight_date)]
            dstat.sort_values('fight_date', inplace = True)            
            if len(dstat) != len(ostat):
                raise ValueError()                
            if (dstat[['bout_id', 'fight_date']].values != ostat[['bout_id', 'fight_date']].values).any():
                raise ValueError()            
            if len(dstat) == 0 or len(ostat) == 0:
                continue
            f_stats[bout] = {'fight_date': fight_date, 'opponent_id': opponent}
            for col in cols:
                f_stats[bout]['adj_avg_o_'+col] = ostat['adj_o_'+col].mean()                  
            for col in cols:
                f_stats[bout]['adj_avg_d_'+col] = dstat['adj_d_'+col].mean()
            adj_avg_data = pd.DataFrame.from_dict(f_stats).T
            for adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
                adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,\
                adj_avg_d_kd,adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,\
                adj_avg_d_sss,adj_avg_d_sub,adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,\
                adj_avg_o_bodyssa,adj_avg_o_bodysss,adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,\
                adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,adj_avg_o_headssa,adj_avg_o_headsss,\
                adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,adj_avg_o_rev,adj_avg_o_ssa,\
                adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss,\
                fight_date,opponent_id in adj_avg_data.values:
                
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
                    %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f);" % (fighter, \
                    bout, adj_avg_d_bodyssa,adj_avg_d_bodysss,adj_avg_d_clinssa,adj_avg_d_clinsss,adj_avg_d_disssa,\
                    adj_avg_d_dissss,adj_avg_d_gndssa,adj_avg_d_gndsss,adj_avg_d_headssa,adj_avg_d_headsss,adj_avg_d_kd,\
                    adj_avg_d_legssa,adj_avg_d_legsss,adj_avg_d_pas,adj_avg_d_rev,adj_avg_d_ssa,adj_avg_d_sss,adj_avg_d_sub,\
                    adj_avg_d_tda,adj_avg_d_tds,adj_avg_d_tsa,adj_avg_d_tss,adj_avg_o_bodyssa,adj_avg_o_bodysss,\
                    adj_avg_o_clinssa,adj_avg_o_clinsss,adj_avg_o_disssa,adj_avg_o_dissss,adj_avg_o_gndssa,adj_avg_o_gndsss,\
                    adj_avg_o_headssa,adj_avg_o_headsss,adj_avg_o_kd,adj_avg_o_legssa,adj_avg_o_legsss,adj_avg_o_pas,\
                    adj_avg_o_rev,adj_avg_o_ssa,adj_avg_o_sss,adj_avg_o_sub,adj_avg_o_tda,adj_avg_o_tds,adj_avg_o_tsa,adj_avg_o_tss)
                pg_insert(PSQL.client, script)
            
            
def update_deriv():
    update_adj_data()
    update_adj_avg_data()
    update_avg_data()
    
    
def update_base_data():
#    pop_fights()
#    pop_fighters()
#    pop_wc_meth()
#    pop_bouts()
#    pop_bout_res()
#    pop_bout_stats()
    pop_bout_fighter_xref()
#    pop_winner_odds()
    

#hgds
if __name__ == '__main__':
#    update_base_data()
    update_deriv()
