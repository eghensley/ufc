import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))


from _connections import db_connection
import pandas as pd
from db.pop_psql import pg_query, pg_insert
from db.pop_proc_data import pull_stats, cols, pull_adj_data, pull_avg_data
import numpy as np
from progress_bar import progress


PSQL = db_connection('psql')


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
    
    
if __name__ == '__main__':
    update_deriv()