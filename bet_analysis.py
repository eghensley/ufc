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
from pop_psql import pg_query, pg_insert
from pop_proc_data import pull_stats, cols, pull_adj_data, pull_adj_avg_data, pull_avg_data
import numpy as np
from datetime import datetime
from joblib import load

PSQL = db_connection('psql')


def pull_pred_data():
    avg_data = pull_avg_data()
    adj_avg_data = pull_adj_avg_data()
    
    nxt_bouts = pg_query(PSQL.client, "select bx.bout_id, fighter_id, opponent_id, date from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id join ufc.fights f on f.fight_id = b.fight_id where b.fight_id = '%s';" % ('351264d11286d09a'))
    nxt_bouts.columns = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date']

    avg_data = pd.merge(avg_data, nxt_bouts, left_on = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date'], right_on = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date'], how = 'inner')
    adj_avg_data = pd.merge(adj_avg_data, nxt_bouts, left_on = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date'], right_on = ['bout_id', 'fighter_id', 'opponent_id', 'fight_date'], how = 'inner')

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
    

    stats = pull_stats()    
    bouts = pg_query(PSQL.client, "select b.bout_id, weight_id, winner, loser from ufc.bouts b join ufc.bout_results br on br.bout_id = b.bout_id join ufc.fights f on f.fight_id = b.fight_id")
    bouts.columns = ['bout_id', 'weight_id', 'winner', 'loser']
    bouts = {i:{'winner':j, 'loser': k} for i,j,k in bouts[['bout_id', 'winner', 'loser']].values}
    winner_id = 0
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

    bout_len = pg_query(PSQL.client, "SELECT bout_id, length from ufc.bout_results")
    bout_len.columns = ['bout_id', 'length'] 
    stats = pd.merge(stats, bout_len, left_on = 'bout_id', right_on = 'bout_id')

    stats = pd.merge(stats, data['fighter_id'], left_on = 'fighter_id', right_on = 'fighter_id', how = 'inner')

    streak_data = {}         
    for fighter in stats.fighter_id.unique():    
        add_data = stats.loc[stats['fighter_id'] == fighter][['bout_id', 'fight_date', 'length', 'won']]
        add_data.sort_values('fight_date', inplace = True)
        
        f_streak = {}

        f_streak[data.loc[data['fighter_id'] == fighter]['bout_id'].values[0]] = {}
        
        f_streak[data.loc[data['fighter_id'] == fighter]['bout_id'].values[0]]['len_avg'] = add_data['length'].mean()
        f_streak[data.loc[data['fighter_id'] == fighter]['bout_id'].values[0]]['win_avg'] = add_data['won'].mean()
        last_res = add_data.iloc[-1]['won']
        streak_count = 0
        for res in reversed(add_data['won'].values):
            if res == last_res:
                streak_count += 1
            else:
                break
        if last_res == 0:
            streak_count *= -1
        f_streak[data.loc[data['fighter_id'] == fighter]['bout_id'].values[0]]['win_streak'] = streak_count
        
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
    pred_data.set_index('bout_id', inplace = True)


    bouts = pg_query(PSQL.client, "select b.bout_id, weight_desc from ufc.bouts b join ufc.fights f on f.fight_id = b.fight_id join ufc.weights w on b.weight_id = w.weight_id")
    bouts.columns = ['bout_id', 'weight_id']
    weights = pd.get_dummies(bouts['weight_id'])
    weights['index'] = bouts['bout_id']
    weights.drop_duplicates(inplace = True)
    weights.set_index('index', inplace = True) 
    META_X = pred_data.join(weights)
    
    return(pred_data, META_X)



def pull_prod_models(domain):
    all_models = {}
    for mod in os.listdir(os.path.join(cur_path, 'prod', 'models', domain, 'est')):
        if mod == '.DS_Store':
            continue
        
        all_models[mod.split('.')[0]] = {}
        est = load(os.path.join(cur_path, 'prod', 'models', domain, 'est', mod))
        meta = load(os.path.join(cur_path, 'prod', 'models', domain, 'error', mod))
        all_models[mod.split('.')[0]]['est'] = est
        all_models[mod.split('.')[0]]['meta'] = meta
    return(all_models)
    

def comb_preds(domain, PRED_X, META_X):

    all_models = pull_prod_models(domain)
    
    results = pd.DataFrame()
    for k,v in all_models.items():
    
        if domain == 'winner':
            res_pred = v['est'].predict_proba(PRED_X)
            res_pred = [i[1] for i in res_pred]
        elif domain == 'length':
            res_pred = v['est'].predict(PRED_X)

    #    v['est'].predict(PRED_X)
        err_pred = v['meta'].predict(META_X)
        
        mod_res = pd.DataFrame([res_pred, err_pred]).T
        mod_res.index = PRED_X.index
        mod_res.columns = ['%s_pred' % (k), '%s_err' % (k)]
        
        if len(results) == 0:
            results = mod_res
        else:
            results = results.join(mod_res)
                   
    pred_cols = [i for i in list(results) if 'pred' in i]
    err_cols = [i for i in list(results) if 'err' in i]
    
    predictions = results[pred_cols].mean(axis = 1)
    predicted_errors = results[err_cols].mean(axis = 1)

    return(predictions, predicted_errors)
    
    
def odds_converter(odds):
#    odds = win_odds
    if odds[0] == '+':
        imp_prob = 100 / (int(odds[1:]) +100)
    elif odds[0] == '-':
        imp_prob = (int(odds[1:])) / ((int(odds[1:]))+100)
    return(imp_prob)
    

PRED_X, META_X= pull_pred_data()

meta = PRED_X[['fighter_id', 'opponent_id']]
win_predictions, win_predicted_errors = comb_preds('winner', PRED_X, META_X)
len_predictions, len_predicted_errors = comb_preds('length', PRED_X, META_X)

win_predictions.name = 'win'
win_predicted_errors.name = 'win_error'

len_predictions.name = 'length'
len_predicted_errors.name = 'length_error'


predictions = meta.join(win_predictions).join(win_predicted_errors).join(len_predictions).join(len_predicted_errors)

fighters = pg_query(PSQL.client, "select fighter_id, name from ufc.fighters")
fighters = {k:v for k,v in fighters.values}

predictions['fighter_id'] = predictions['fighter_id'].apply(lambda x: fighters[x])
predictions['opponent_id'] = predictions['opponent_id'].apply(lambda x: fighters[x])

f_to_code = pg_query(PSQL.client, "select fighter_id, name from ufc.fighters")
f_to_code = {v:k for k,v in f_to_code.values}

odds = {}

for bout in predictions.index:
    print('Fight: %s VS %s' % (predictions.loc[bout]['fighter_id'], predictions.loc[bout]['opponent_id']))
    win_odds = input('Vegas odds for %s:   ' % (predictions.loc[bout]['fighter_id']))
    win_prob = odds_converter(win_odds)
    lose_prob = input('Vegas odds for %s:   ' % (predictions.loc[bout]['opponent_id']))
    lose_prob = odds_converter(lose_prob)

    length = input('Base for fight length:   ')
    over_odds = input('Vegas odds for over:   ')
    over_odds = odds_converter(over_odds)
    under_odds = input('Vegas odds for under:   ')
    under_odds = odds_converter(under_odds)

    odds[bout] = {'win_odds': win_prob, 'lose_odds': lose_prob, 'length_over': length, 'over_odds': over_odds, 'under_odds': under_odds}
    
    
for bout in predictions.index:
    print('~~~~~~~~~~~~~~~~~')
    print('Fight: %s VS %s' % (predictions.loc[bout]['fighter_id'], predictions.loc[bout]['opponent_id']))
    print('')
    print('%s win probability: %.1f%%' % (predictions.loc[bout]['fighter_id'], 100*predictions.loc[bout]['win']))
    print('Vegas odds: %.1f%%' % (100*odds[bout]['win_odds']))
    win_odds_dif = 100* (predictions.loc[bout]['win'] - odds[bout]['win_odds'])
    if win_odds_dif >0:
        print('%.1f%% ADVANTAGE for %s to WIN' % (win_odds_dif,predictions.loc[bout]['fighter_id'] ))
    else:
        lose_odds_dif = 100 * ((1 - predictions.loc[bout]['win']) - odds[bout]['lose_odds'])
        
        if lose_odds_dif > 0:
            print('%.1f%% ADVANTAGE for %s to WIN' % (lose_odds_dif,predictions.loc[bout]['opponent_id'] ))
        else:
            print('NO betting advantage')
            
    print('')
    print('Predicted fight length: %.2f Minutes +- %.1f' % (predictions.loc[bout]['length'], predictions.loc[bout]['length_error']))
    print('O/U set at: %.2f Minutes' % (float(odds[bout]['length_over'])))
    
    print('OVER implied probability: %.1f%%' % (100 * odds[bout]['over_odds']))
    print('UNDER implied probability: %.1f%%' % (100* odds[bout]['under_odds']))
    
