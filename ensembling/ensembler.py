import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))  
    
if os.path.join(cur_path, 'lib', 'python3.7', 'site-packages') not in sys.path:  
    sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
if os.path.join(cur_path, 'lib','LightGBM', 'python-package') not in sys.path:  
    sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))
if cur_path not in sys.path:  
    sys.path.insert(1, os.path.join(cur_path))

import pandas as pd
import numpy as np
from copy import deepcopy
from gaus_proc import bayesian_optimisation
try:
    import matplotlib.pyplot as plt
except:
    pass
import random
from progress_bar import progress
import json
from Simple import SimpleTuner
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from _connections import db_connection
from db.pop_psql import pg_query

#def best_fit_slope_and_intercept(xs,ys):
#    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
#         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
#    
#    b = np.mean(ys) - m*np.mean(xs)
#    return m, b


#def bout_model_errors(bout, _params):
#    model_errors = {}
#    for val in _params['features']:
#        model_errors[val] = logloss(bout['winner'], bout[val])
#    return(model_errors)


def conv_to_ml_odds(prob):
    if prob > .5:
        odds = -1 * (prob/(1-prob))*100
    else:
        odds = ((1-prob)/prob)*100
    return(odds)


def pot_payout(odd, stake = 10):
    odds = conv_to_ml_odds(odd)
    if odds > 0:
        po = stake*(odds/100)
    else:
        po = stake/(odds/100)*-1
    return(po)
    

#pot_payout(0.5476)

def logloss(true_label, predicted, eps=1e-15):
  p = np.clip(predicted, eps, 1 - eps)
  if true_label == 1:
    return -np.log(p)
  else:
    return -np.log(1 - p)


def det_stake(adv, const, mult, exp):
#    return(const)
    return(const * ((mult * (1 + adv)) * (exp * (1 + adv) ** 2)))
#    return(const * (mult * (1 + adv - thresh) + exp * (1 + adv - thresh) ** 2))


def proc_result_meta(results):
    odds_diff = []
    winning = []
    win = []
    for v in results.values():
        if v[0] == 0:
            continue
        winning.append(v[0])
        if v[0] > 0:
            win.append(1)
        else:
            win.append(0)
        odds_diff.append(v[1][v[2][0]]['odds_diff'])
    
    df = pd.DataFrame([odds_diff, winning, win]).T
    
    acc = df[2].mean()
    money = df[1].mean()
    
    bins = np.linspace(0, .6, 15)
    groups = df.groupby(pd.cut(df[0], bins))
    
    binned_money = groups.mean()[1]
    binned_acc = groups.mean()[2]
    binned_count = groups.count()
    binned_diffs = bins
    
    comps = {
            'acc': acc,
            'money': money,
            'binned_money': binned_money,
            'binned_acc': binned_acc,
            'binned_count': binned_count,
            'binned_diffs': binned_diffs
            }
    return(comps)




      
class ensembled_predictor:
    
    def __init__(self):
        self.features = ['KNN',
                         'LogRegression',
                         'LinSVC',
                         'RFclass',
                         'RbfSVC',
                         'PolySVC',
                         'DartGBM',
                         'LightGBM']
        self.weights = {'KNN':1,
                         'LogRegression':1,
                         'LinSVC':1,
                         'RFclass':1,
                         'RbfSVC':1,
                         'PolySVC':1,
                         'DartGBM':1,
                         'LightGBM':1
                         }
        for i in self.features:
            self.weights[i] /= len(self.features)
        self.models = {}
#        for mod in self.features:
#            self.models[mod] =  load(os.path.join(cur_path, 'ensembling', 'model', 'features', '%s.pkl' % (mod)))
        self.threshold = 0
        self.bet_const = 1
        self.bet_exp = 1
        self.bet_mult = 1
        self.tuned = False
        self.training_data = None
        self.training_results = None
        self.loaded_feature_models = False


    def load_feature_models(self, retrain = True, train_data = None):
        if retrain:
            train_data = train_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
            X_train = train_data[[i for i in list(train_data) if i != 'winner']]
            Y_train = train_data['winner'].apply(lambda x: x if x == 1 else 0)
        
            final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', 'winner', 'final', 'models')
            
        for mod in self.features:
            if retrain:
                model_path = os.listdir(os.path.join(final_model_folder, mod))
                new_model = load(os.path.join(final_model_folder, mod, model_path[0])) 
                new_model.fit(X_train, Y_train)
                self.models[mod] = new_model
                dump(new_model, os.path.join(cur_path, 'ensembling', 'model', 'features', '%s.pkl' % (mod)))
            else:
                self.models[mod] =  load(os.path.join(cur_path, 'ensembling', 'model', 'features', '%s.pkl' % (mod)))
        self.loaded_feature_models = True

                
            
    def _ret_validation_set(self, test_data):    
        PSQL = db_connection('psql')
        
#        test_data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data_test.csv'))
        test_data = test_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
        X_test = test_data[[i for i in list(test_data) if i != 'winner']]
        Y_test = test_data['winner'].apply(lambda x: x if x == 1 else 0)
        
        pred_df = pd.DataFrame(Y_test)        
        for mod_name, model in self.models.items():
            model_preds = model.predict_proba(X_test)
            model_preds = [i[1] for i in model_preds]
            pred_df[mod_name] = model_preds
            
        vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
        vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
        vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)
        vegas_pred_df = pred_df.join(vegas_preds).dropna()
        vegas_pred_df.reset_index(inplace = True)
        return(vegas_pred_df)
    

    def plot_training_feature_results(self, stage = 'weights'):

#        if stage == 'weights':
#            for i,(k,v) in enumerate(zip(self.features, self.weights.values())):
#                plt.bar(i, v, label = k)
#            plt.ylabel('Weights')
#            plt.legend()
#            plt.show()        
        
        if self.training_results is None:
            self.training_result_data()
            
        comp_results = {}
        results = {}
        for bout in self.training_data.bout_id.unique():
            results[bout] = self.bet_analysis(self.training_data.loc[self.training_data['bout_id'] == bout])
        comps = proc_result_meta(results)
        comp_results['ENSEMBLE'] = comps

        save_weights = deepcopy(self.weights)
        fake_params = {}
        for mod in self.features:
            fake_params[mod] = 0
        self.weights = fake_params
        
        for upstream_mod in self.features:
            for downstream_mod in self.features:
                fake_params[downstream_mod] = 0
            fake_params[upstream_mod] = 1
            self.weights = fake_params
            results = {}
            for bout in self.training_data.bout_id.unique():
                results[bout] = self.bet_analysis(self.training_data.loc[self.training_data['bout_id'] == bout])
            comps = proc_result_meta(results)
            comp_results[upstream_mod] = comps
        self.weights = save_weights

        if stage == 'weights':
            for k,v in comp_results.items():
                if k == 'ENSEMBLE':
                    line = '-'
                else:
                    line = ':'
                plt.plot(v['binned_diffs'][:-1], v['binned_acc'].values, label = k, linestyle = line)
            plt.xlabel('Exp Advantage')
            plt.ylabel('Confidence')
            plt.legend()
            plt.show()
            
            for k,v in comp_results.items():
                if k == 'ENSEMBLE':
                    line = '-'
                else:
                    line = ':' 
                plt.plot(v['binned_diffs'][:-1], v['binned_count'][0].values/np.sum(v['binned_count'][0].values), linestyle = line, label = k)
            plt.ylabel('% of Fights')
            plt.xlabel('Exp Advantage')
            plt.legend()
            plt.show()
        
            for i,(k,v) in enumerate(comp_results.items()):
                plt.bar(i, v['acc'], label = k)
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
            
#        if stage == 'betting':     
#        for k,v in comp_results.items():
#            if k == 'ENSEMBLE':
#                line = '-'
#            else:
#                line = ':'
#            plt.plot(v['binned_diffs'][:-1], v['binned_money'].values, label = k, linestyle = line)
#        plt.xlabel('Exp Advantage')
#        plt.ylabel('Avg Winnings ($)')
#        plt.legend()
#        plt.show()
#
#        for i,(k,v) in enumerate(comp_results.items()):
#            plt.bar(i, v['money'], label = k)
#        plt.ylabel('Avg Winnings ($)')
#        plt.legend()
#        plt.show()        
#            self.training_results = None
        

    def training_result_data(self):    
        if not self.tuned:
            raise ValueError('Model has not been tuned')
            
        results = {}
        for bout in self.training_data.bout_id.unique():
            results[bout] = self.bet_analysis(self.training_data.loc[self.training_data['bout_id'] == bout])
         
        odds_diff = []
        winning = []
        win = []
        for v in results.values():
            if v[0] == 0:
                continue
            winning.append(v[0])
            if v[0] > 0:
                win.append(1)
            else:
                win.append(0)
            odds_diff.append(v[1][v[2][0]]['odds_diff'])
        
        df = pd.DataFrame([odds_diff, winning, win]).T
        df.columns = ['Odds Difference', 'Winning', 'W/L']
        self.training_results = df
        
    
    def plot_training_results(self):  
        if self.training_results is None:
            self.training_result_data()
            
        bins = np.linspace(self.training_results['Odds Difference'].min(), self.training_results['Odds Difference'].max(), 15)
        groups = self.training_results.groupby(pd.cut(self.training_results['Odds Difference'], bins))
    
        binned_df = groups.mean()
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('Exp Bet Advantage')
        ax1.set_ylabel('Exp Winnings ($)', color=color)
        ax1.plot(binned_df['Odds Difference'], binned_df['Winning'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        color = 'tab:blue'
        ax2.set_ylabel('Confidence (%)', color=color)  # we already handled the x-label with ax1
        ax2.plot(binned_df['Odds Difference'], binned_df['W/L'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
    def win_confidence_model(self):
        if not self.tuned:
            raise ValueError('Model has not been tuned')
        
        win_odds_est = None
        try:
            win_odds_est = load(os.path.join(cur_path, 'ensembling', 'model', 'error_est.pkl'))
        except FileNotFoundError:
            if self.training_results is None:
                self.training_result_data()
            win_odds_est = LogisticRegression().fit(self.training_results['Odds Difference'].values.reshape(-1, 1), self.training_results['W/L'])
            dump(win_odds_est, os.path.join(cur_path, 'ensembling', 'model', 'error_est.pkl'))


    def bout_ensemble_error(self, bout):
        pred_win_prob = self.bout_ensemble_pred(bout)
        return(logloss(bout['winner'].values[0], pred_win_prob))
    

    def bout_ensemble_pred(self, bout_data):
        model_preds = {}
        for val in self.features:
            try:
                model_preds[val] = bout_data[val].values[0] * self.weights[val]
            except AttributeError:
                model_preds[val] = bout_data[val] * self.weights[val]                
        pred_win_prob = np.sum([i for i in model_preds.values()])
        return(pred_win_prob)
    

    def make_bet_pred(self, bout_data):
        pred_win_prob = self.bout_ensemble_pred(bout_data)
        vegas_win_prob = bout_data['VEGAS']
        odds_diff = pred_win_prob - vegas_win_prob - self.threshold
        return({'vegas_odds': vegas_win_prob, 'ensemble_probability': pred_win_prob, 'odds_diff': odds_diff})
        

    def vegas_fighter_pred(self, bouts):    
        if len(bouts) != 2:
            raise ValueError('Need both fighters, recieved %i' % (len(bouts)))
        bout_preds = {}
        for bout_idx, bout in bouts.iterrows():
            bout_preds[bout['fighter_id']] = self.make_bet_pred(bout)
            if bout_preds[bout['fighter_id']]['odds_diff'] > 0:
                bout_preds[bout['fighter_id']]['make_bet'] = True
            else:
                bout_preds[bout['fighter_id']]['make_bet'] = False
        adv_bets = [k for k,v in bout_preds.items() if v['make_bet']]
        if len(adv_bets) > 1:
            max_diff = max([k['odds_diff'] for k in bout_preds.values()])
            best_adv = [k for k in adv_bets if bout_preds[k]['odds_diff'] == max_diff]
            for k in adv_bets:
                if k != best_adv[0]:
                    bout_preds[k]['make_bet'] = False
            adv_bets = best_adv
        return(bout_preds, adv_bets)    
    
    
    def bet_outcome(self, bouts):
        if sum([i for i in self.weights.values()]) != 1:
            raise ValueError('Weights sum to %.2f' % (sum([i for i in self.weights.values()]) ))
        pred, adv = self.vegas_fighter_pred(bouts)
        if len(adv) == 0:
            return(0)
        stake = det_stake(pred[adv[0]]['odds_diff'], self.bet_const, self.bet_mult, self.bet_exp)
        payout = pot_payout(bouts.loc[bouts['fighter_id'] == adv[0]]['VEGAS'].values[0], stake = stake)
        if bouts.loc[bouts['fighter_id'] == adv[0]]['winner'].values[0] == 1:
            return(payout)
        else:
            return(-stake)        


    def bet_analysis(self, bouts):
        if sum([i for i in self.weights.values()]) != 1:
            raise ValueError('Weights sum to %.2f' % (sum([i for i in self.weights.values()]) ))
        pred, adv = self.vegas_fighter_pred(bouts)
        if len(adv) == 0:
            return(0, pred, adv)
        stake = det_stake(pred[adv[0]]['odds_diff'], self.bet_const, self.bet_mult, self.bet_exp)
        payout = pot_payout(bouts.loc[bouts['fighter_id'] == adv[0]]['VEGAS'].values[0], stake = stake)
        if bouts.loc[bouts['fighter_id'] == adv[0]]['winner'].values[0] == 1:
            return(payout, pred, adv)
        else:
            return(-stake, pred, adv)  
            

    def log_loss_funct(self, gp_weights):
        gp_weights = [i for i in gp_weights]
        gp_weights.append(1 - np.sum(gp_weights))
        for i,w in zip(self.features, gp_weights):
            self.weights[i] = w
        loss = self.training_data.groupby(['bout_id', 'fighter_id']).apply(lambda x: self.bout_ensemble_error(x))
        loss = loss.groupby('bout_id').mean().mean() * -1 
        return(loss)
    

    def money_loss_function(self, gp_params):
        self.threshold = gp_params[0]
        self.bet_mult = gp_params[1]
        self.bet_exp = gp_params[2]
        results = self.training_data.groupby('bout_id').apply(lambda x: self.bet_outcome(x))   
        return(results.mean())  
    

    def tune_weights(self, num_iter, explore):
        print('')
        print('Tuning Weights')
        vertices = []
        for i in range(len(self.features)):
            vert = [0 for i in range(len(self.features) - 1)]
            if i != 0:
                vert[i-1] = 1
            vertices.append(vert)
        tuner = SimpleTuner(vertices, self.log_loss_funct, exploration_preference=explore)
        tuner.optimize(num_iter)
        best_objective_value, best_coords = tuner.get_best()
        best_coords = list(best_coords)
        best_coords.append(1 - np.sum(best_coords))
        tuner.trackBestCoord
        print('Best objective value ', best_objective_value)
        for i,w in zip(self.features, best_coords):
            self.weights[i] = w
        

    def tune_bet_params(self, num_iter, explore):
        print('')
        print('Tuning Bet Parameters')
        vertices = [[0, 10, 0], [0, 10, 10], [.5, 0, 10], [.5, 10, 10]]
        tuner = SimpleTuner(vertices, self.money_loss_function, exploration_preference=explore)
        tuner.optimize(num_iter)
        best_objective_value, best_coords = tuner.get_best()
        print('Best objective value ', best_objective_value)
        self.threshold = best_coords[0]
        self.bet_mult = best_coords[1]
        self.bet_exp = best_coords[2]

        
    def fit(self, data, number_of_iterations = 100, exploration = 0.15, plot = True):
        if not self.loaded_feature_models:
            raise ValueError('Feature models have not been loaded')
            
        self.training_data = self._ret_validation_set(data)
        self.tuned = True
         # optional, default 0.15
        self.tune_weights(number_of_iterations, exploration)
#        if plot:
#            self.plot_training_feature_results()
#        print('Model Parameters:')
#        for k,v in zip(self.features, self.weights.values()):
#            print('%s : %.2f' % (k,v))
#        self.tune_bet_params(number_of_iterations, exploration)
#        if plot:
#            self.plot_training_feature_results(stage = 'betting')
#        self.training_data = None
#        print('Betting function: %.1f x (%.1f x (Bet Adv - %.1f) + (%.1f x (Bet Adv - %.1f)^2))' % (self.bet_const, self.bet_mult, self.threshold, self.bet_exp, self.threshold) )


    def score(self):
        if not self.loaded_feature_models:
            raise ValueError('Feature models have not been loaded')

        print('')
        self.training_data = self._ret_validation_set()
        self.plot_training_feature_results()
        self.plot_training_feature_results(stage = 'betting')

        loss = self.training_data.groupby(['bout_id', 'fighter_id']).apply(lambda x: self.bout_ensemble_error(x))
        loss = loss.groupby('bout_id').mean().mean()
        money = self.training_data.groupby('bout_id').apply(lambda x: self.bet_outcome(x)).mean()

        print('~~ Ensemble Validation Results ~~')
        print('Ensemble Log Loss : %.3f' % (loss))
        print('Ensemble Avg $ : %.2f' % (money))
        
        self.training_data = None
        
        
        
        
#if __name__ == '__main__':      
#    data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_ens_tuning.csv'))
#    model = ensembled_predictor()
#    model.fit(data)  
##    model.score()
#    dump(model, os.path.join(cur_path, 'ensembling', 'model', 'ensemble.pkl'))    
##    model = load(os.path.join(cur_path, 'ensembling', 'model', 'ensemble.pkl'))

#train_data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_est_training.csv'))
#model.load_feature_models(retrain = True, train_data = train_data)
model = ensembled_predictor()
model.load_feature_models(retrain = False)
data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_ens_training.csv'))
model.training_data = model._ret_validation_set(data)
#model.fit(data)

import math


cv_scores = []
for j in range(10):
    save_full_set = deepcopy(model.training_data)
    sample_set = random.sample(list(model.training_data.index), int(len(model.training_data.index)/10))
    
    vertices = []
    for i in range(len(model.features)):
        vert = [0 for i in range(len(model.features) - 1)]
        if i != 0:
            vert[i-1] = 1
        vertices.append(vert)
    explore = .15
    num_iter = 30
    model.training_data = model.training_data.loc[sample_set]
    tuner = SimpleTuner(vertices, model.log_loss_funct, exploration_preference=explore)
    tuner.optimize(num_iter)
    
    score_shares = []
    best_objective_value, best_coords = tuner.get_best()
    best_coords = list(best_coords)
    best_coords.append(1 - np.sum(best_coords))
    for i,w in zip(model.features, best_coords):
        model.weights[i] = w
    model.training_data = save_full_set.loc[~save_full_set.index.isin(sample_set)]   
    oob_score = model.log_loss_funct([i for i in model.weights.values()][:-1])
    for i,w in zip(model.features, best_coords):
        score_shares.append(0 if w == 0 else oob_score*w)
    cv_scores.append(score_shares)  
    model.training_data = save_full_set

np.array(cv_scores)


#train_loss = model.log_loss_funct([i for i in model.weights.values()])
#
#test_data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data_test.csv'))
#model.training_data = model._ret_validation_set(test_data)
#
#test_loss = model.log_loss_funct([i for i in model.weights.values()])






khgcku
model.training_data = model._ret_validation_set(data)

#for bout in model.training_data.bout_id.unique():
#    adsfafs
#    thresh_res = {}
#    for thresh in np.linspace(0,.1,10):   
#        print(thresh)
#        model.threshold = thresh
#        thresh_res[thresh] = model.bet_analysis(model.training_data.loc[model.training_data['bout_id'] == bout])[0]
# 
#
#
#
#smple_bout = model.training_data.loc[model.training_data['bout_id'] == bout]
#model.bet_analysis(smple_bout)
#
##asdfasdf
#
##dump(model, os.path.join(cur_path, 'ensembling', 'model', 'ensemble.pkl'))    
##model = load(os.path.join(cur_path, 'ensembling', 'model', 'ensemble.pkl'))
##data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_ens_training.csv'))
#
#
#
#hfdjfjc
#
#
#
#
#def fit(self, data, number_of_iterations = 100, exploration = 0.15, plot = True):
#    if not model.loaded_feature_models:
#        raise ValueError('Feature models have not been loaded')
#        
#    model.training_data = model._ret_validation_set(data)
#    model.tuned = True
#     # optional, default 0.15
##    model.tune_weights(number_of_iterations, exploration)
#    if plot:
#        model.plot_training_feature_results()
#    print('Model Parameters:')
#    for k,v in zip(model.features, model.weights.values()):
#        print('%s : %.2f' % (k,v))
##    print(parameters)
#    model.tune_bet_params(number_of_iterations, exploration)
#    if plot:
#        model.plot_training_feature_results(stage = 'betting')
##    print(parameters)
#    model.training_data = None
#    print('Betting function: %.1f x (%.1f x (Bet Adv - %.1f) + (%.1f x (Bet Adv - %.1f)^2))' % (model.bet_const, model.bet_mult, model.threshold, model.bet_exp, model.threshold) )
#
#
#
#
##model.weights = save_weights
##def plot_training_feature_results(model, stage = 'weights'):
#
#    
#    
#for thresh in np.linspace(0,.1,10):   
#    print(thresh)
#    model.threshold = 0
#    model.bet_const = 1
#    model.bet_exp = 1
#    model.bet_mult = 1
#    stage = 'weights'
##    if stage == 'weights':
##        for i,(k,v) in enumerate(zip(model.features, model.weights.values())):
##            plt.bar(i, v, label = k)
##        plt.ylabel('Weights')
##        plt.legend()
##        plt.show()        
#    
#    if model.training_results is None:
#        model.training_result_data()
#        
#    comp_results = {}
#    results = {}
#    for bout in model.training_data.bout_id.unique():
#        results[bout] = model.bet_analysis(model.training_data.loc[model.training_data['bout_id'] == bout])
#    comps = proc_result_meta(results)
#    comp_results['ENSEMBLE'] = comps
#
#    save_weights = deepcopy(model.weights)
#    fake_params = {}
#    for mod in model.features:
#        fake_params[mod] = 0
#    model.weights = fake_params
#    
#    for upstream_mod in model.features:
#        for downstream_mod in model.features:
#            fake_params[downstream_mod] = 0
#        fake_params[upstream_mod] = 1
#        model.weights = fake_params
#        results = {}
#        for bout in model.training_data.bout_id.unique():
#            results[bout] = model.bet_analysis(model.training_data.loc[model.training_data['bout_id'] == bout])
#        comps = proc_result_meta(results)
#        comp_results[upstream_mod] = comps
#    model.weights = save_weights
#
#
#
#
#    for bout in model.training_data.bout_id.unique():
#        adsfafs
#        thresh_res = {}
#        for thresh in np.linspace(0,.1,10):   
#            print(thresh)
#            model.threshold = thresh
#            thresh_res[thresh] = model.bet_analysis(model.training_data.loc[model.training_data['bout_id'] == bout])
#            
#
#
#
#
#
#
#def bet_analysis(model, bouts):
#    if sum([i for i in model.weights.values()]) != 1:
#        raise ValueError('Weights sum to %.2f' % (sum([i for i in model.weights.values()]) ))
#    pred, adv = model.vegas_fighter_pred(bouts)
#    if len(adv) == 0:
#        return(0, pred, adv)
#    stake = det_stake(pred[adv[0]]['odds_diff'], model.bet_const, model.bet_mult, model.threshold, model.bet_exp)
#    payout = pot_payout(bouts.loc[bouts['fighter_id'] == adv[0]]['VEGAS'].values[0], stake = stake)
#    if bouts.loc[bouts['fighter_id'] == adv[0]]['winner'].values[0] == 1:
#        return(payout, pred, adv)
#    else:
#        return(-stake, pred, adv)  
#
#
#
#    
#
#def proc_result_meta(results):
#    odds_diff = []
#    winning = []
#    win = []
#    log_loss = []
#    for v in results.values():
#        if v[0] == 0:
#            continue
#        winning.append(v[0])
#        if v[0] > 0:
#            win.append(1)
#            log_loss.append(logloss(1, v[1][v[2][0]]['ensemble_probability']))
#        else:
#            win.append(0)
#            log_loss.append(logloss(0, v[1][v[2][0]]['ensemble_probability']))
#
#        odds_diff.append(v[1][v[2][0]]['odds_diff'])
#    
#    df = pd.DataFrame([odds_diff, winning, win, log_loss]).T
#    df.sort_values(0, inplace = True)
#
#
#
#from sklearn.utils import resample
#df_majority = df[df[2]==0]
#df_minority = df[df[2]==1]
#df_minority_upsampled = resample(df_minority, 
#                                 replace=True,     # sample with replacement
#                                 n_samples=len(df_majority),    # to match majority class
#                                 random_state=123) # reproducible results
# 
## Combine majority class with upsampled minority class
#df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# 
## Display new class counts
#df_upsampled[2].value_counts()
#df_upsampled.sort_values(0, inplace = True)
#
#odds_diff_sample = np.linspace(0, .3, 30)
##winnings_est = np.poly1d(np.polyfit(df_upsampled[0], df_upsampled[1], 4))
##plt.plot(df_upsampled[0], [winnings_est(i) for i in df_upsampled[0].values])   
#winnings_est = np.poly1d(np.polyfit(df_upsampled[0], df_upsampled[1], 4))
#plt.plot(df_upsampled[0], [winnings_est(i) for i in df_upsampled[0].values])  



    
    
#    plt.scatter(df_upsampled[0], df_upsampled[2])
#    poly = np.poly1d(np.polyfit(df_upsampled[0], df_upsampled[2], 2))
#    plt.plot(df_upsampled[0], [poly(i) for i in df_upsampled[0].values])   
#    
#    
#    plt.scatter(df[0], df[3])
#    
#    
#    plt.scatter(df[0], df[1])
##    x0, x1, x2, x3 = np.polyfit(df[0], df[1], 3)
#    poly = np.poly1d(np.polyfit(df[0], df[1], 2))
#    plt.plot(df[0], [poly(i) for i in df[0].values])
#    
#    plt.scatter(df[0], df[3])
##    x0, x1, x2, x3 = np.polyfit(df[0], df[1], 3)
#    poly = np.poly1d(np.polyfit(df[0], df[3], 2))
#    plt.plot(df[0], [poly(i) for i in df[0].values])
#    
#    acc = df[2].mean()
#    money = df[1].mean()
#    
#    bins = np.linspace(min(df[0]), max(df[0]), 15)
#    groups = df.groupby(pd.cut(df[0], bins))
#    
#    binned_money = groups.mean()[1]
#    binned_acc = groups.mean()[2]
#    binned_count = groups.count()
#    binned_diffs = bins
#    
#    comps = {
#            'acc': acc,
#            'money': money,
#            'binned_money': binned_money,
#            'binned_acc': binned_acc,
#            'binned_count': binned_count,
#            'binned_diffs': binned_diffs
#            }
#    return(comps)
    
    
    
    
    
    
    
    
#    if stage == 'weights':
#        for k,v in comp_results.items():
#            if k == 'ENSEMBLE':
#                line = '-'
#            else:
#                line = ':'
#            plt.plot(v['binned_diffs'][:-1], v['binned_acc'].values, label = k, linestyle = line)
#        plt.xlabel('Exp Advantage')
#        plt.ylabel('Confidence')
#        plt.legend()
#        plt.show()
#        
#        for k,v in comp_results.items():
#            if k == 'ENSEMBLE':
#                line = '-'
#            else:
#                line = ':' 
#            plt.plot(v['binned_diffs'][:-1], v['binned_count'][0].values/np.sum(v['binned_count'][0].values), linestyle = line, label = k)
#        plt.ylabel('% of Fights')
#        plt.xlabel('Exp Advantage')
#        plt.legend()
#        plt.show()
#    
#        for i,(k,v) in enumerate(comp_results.items()):
#            plt.bar(i, v['acc'], label = k)
#        plt.ylabel('Accuracy')
#        plt.legend()
#        plt.show()
        
#        if stage == 'betting':     
#    for k,v in comp_results.items():
#        if k == 'ENSEMBLE':
#            line = '-'
#        else:
#            line = ':'
#        plt.plot(v['binned_diffs'][:-1], v['binned_money'].values, label = k, linestyle = line)
#    plt.xlabel('Exp Advantage')
#    plt.ylabel('Avg Winnings ($)')
#    plt.legend()
#    plt.show()

#    for i,(k,v) in enumerate(comp_results.items()):
#        plt.bar(i, v['money'], label = k)
#    plt.ylabel('Avg Winnings ($)')
#    plt.legend()
#    plt.show()        
#            model.training_results = None

