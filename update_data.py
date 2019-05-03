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
from pop_proc_data import pull_stats, cols, pull_avg_data, pull_adj_data



PSQL = db_connection('psql')


def update_avg_data():
    stats = pull_stats()
    missing_bouts = pg_query(PSQL.client, 'select b.bout_id from ufc.bouts b full join ufc.avg_stats avs on b.bout_id = avs.bout_id where avs.bout_id is NULL')
    missing_bouts = set(missing_bouts[0].values)
    nxt_fight = pg_query(PSQL.client, 'select fight_id from ufc.fights where date > NOW() order by date asc limit 1')
    nxt_bouts = pg_query(PSQL.client, "select bx.bout_id, fighter_id, opponent_id, date from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id where b.fight_id = '%s';" % (nxt_fight[0].values[0]))
    
    nxt_bouts.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date']
    
    for bout in nxt_bouts.bout_id.values:
        if bout not in missing_bouts:
            raise ValueError()
    
    stat_avgs = {}
    for bout, fighter, opponent, fight_date in nxt_bouts.values:
        f_stats = {}
        
        ostat = stats.loc[stats['fighter_id'] == fighter]
        ostat.sort_values('fight_date', inplace = True)
    
        dstat = stats.loc[stats['opponent_id'] == fighter]
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
        
        
        
        
        
#def update_adj_data():
#    stats = pull_stats()
#    avg_data = pull_avg_data()
#
#
#    missing_bouts = pg_query(PSQL.client, 'select b.bout_id from ufc.bouts b full join ufc.adj_stats adj on b.bout_id = adj.bout_id where adj.bout_id is NULL')
#    missing_bouts = set(missing_bouts[0].values)
#    nxt_fight = pg_query(PSQL.client, 'select fight_id from ufc.fights where date > NOW() order by date asc limit 1')
#    nxt_bouts = pg_query(PSQL.client, "select bx.bout_id, fighter_id, opponent_id, date from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id where b.fight_id = '%s';" % (nxt_fight[0].values[0]))
#    
#    nxt_bouts.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date']
#    
#    for bout in nxt_bouts.bout_id.values:
#        if bout not in missing_bouts:
#            raise ValueError()
#            
#    stat_adj = {}
#    for bout, fighter, opponent, fight_date in nxt_bouts.values:
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
#            pg_insert(PSQL.client, script)
        
        
def update_adj_avg_data():
    adj_data = pull_adj_data()
    
    
    missing_bouts = pg_query(PSQL.client, 'select b.bout_id from ufc.bouts b full join ufc.adj_avg_stats adv on b.bout_id = adv.bout_id where adv.bout_id is NULL')
    missing_bouts = set(missing_bouts[0].values)
    nxt_fight = pg_query(PSQL.client, 'select fight_id from ufc.fights where date > NOW() order by date asc limit 1')
    nxt_bouts = pg_query(PSQL.client, "select bx.bout_id, fighter_id, opponent_id, date from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id where b.fight_id = '%s';" % (nxt_fight[0].values[0]))
    
    nxt_bouts.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date']
    
    for bout in nxt_bouts.bout_id.values:
        if bout not in missing_bouts:
            raise ValueError()    
    
    
    adj_stat_avgs = {}
    for bout, fighter, opponent, fight_date in nxt_bouts.values:
        f_stats = {}
        
        ostat = adj_data.loc[adj_data['fighter_id'] == fighter]
        ostat.sort_values('fight_date', inplace = True)
    
        dstat = adj_data.loc[adj_data['opponent_id'] == fighter]
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