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
#from copy import deepcopy
#from gaus_proc import bayesian_optimisation
try:
    import matplotlib.pyplot as plt
except:
    pass
#import random
#from progress_bar import progress
#import json
from Simple import SimpleTuner
from joblib import load, dump
#from sklearn.linear_model import LogisticRegression
#from _connections import db_connection
#from db.pop_psql import pg_query
from datetime import datetime
from keras import Sequential
from keras.layers import Dense
#from livelossplot import PlotLossesKeras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit


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

                
            
#    def _ret_validation_set(self, test_data):    
#        PSQL = db_connection('psql')
#        
##        test_data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data_test.csv'))
#        test_data = test_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
#        X_test = test_data[[i for i in list(test_data) if i != 'winner']]
#        Y_test = test_data['winner'].apply(lambda x: x if x == 1 else 0)
#        
#        pred_df = pd.DataFrame(Y_test)        
#        for mod_name, model in self.models.items():
#            model_preds = model.predict_proba(X_test)
#            model_preds = [i[1] for i in model_preds]
#            pred_df[mod_name] = model_preds
#            
#        vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
#        vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
#        vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)
#        vegas_pred_df = pred_df.join(vegas_preds).dropna()
#        vegas_pred_df.reset_index(inplace = True)
#        return(vegas_pred_df)
        

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
        





def _split_x_y(test_data):    
    X = test_data[[i for i in list(test_data) if i != 'winner']]
    Y = test_data['winner'].apply(lambda x: x if x == 1 else 0)
    return(X, Y)
    

def _add_fit_models(model, feature_data, split = True):
    if split:
        x,y = _split_x_y(feature_data)
    else:
        x,y = feature_data
    final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', 'winner', 'final', 'models')
    for mod in model.features:
        model_path = os.listdir(os.path.join(final_model_folder, mod))
        new_model = load(os.path.join(final_model_folder, mod, model_path[0])) 
        new_model.fit(x, y)
        model.models[mod] = new_model

def _feature_pred(model, name, test_df):
    return(model.models[name].predict_proba(test_df)[:,1])


def ensemble_prob_pred(model, data):
    comb_preds = np.zeros(len(data))
    for feature_name, feature_model in model.models.items():
        comb_preds += _feature_pred(model, feature_name, data) * model.weights[feature_name]
    return(comb_preds)
 

def _comb_feature_pred(model, data):
    comb_preds = pd.DataFrame()
    for feature_name, feature_model in model.models.items():
        comb_preds[feature_name] = _feature_pred(model, feature_name, data)
    return(comb_preds)
    
    
def _batch_log_loss(actual, pred):
    return(np.mean([logloss(i,j) for i,j in zip(actual, pred)]))


def _split(input_data, splitter):
    x, y = _split_x_y(input_data)
    train_idx, test_idx = [(i,j) for i,j in feature_splitter.split(x,y)][0]
    train = input_data.iloc[train_idx]
    test = input_data.iloc[test_idx]
    return(train, test)


def _add_hist_res(training_loss, test_loss, training_acc, test_acc, history):
    training_loss += history.history['loss']
    test_loss += history.history['val_loss']
    training_acc += history.history['acc']
    test_acc += history.history['val_acc']
    return(training_loss, test_loss, training_acc, test_acc)


def _gen_new_data(model, data, feature_splitter):
    feat_train_data, ens_data = _split(data, feature_splitter)
    ens_train_data, ens_validation_data = _split(ens_data, feature_splitter)
    _add_fit_models(model, feat_train_data)
    
    ens_x_train, ens_y_train = _comb_feature_pred(model, ens_train_data), ens_train_data['winner']
    ens_x_val, ens_y_val = _comb_feature_pred(model, ens_validation_data), ens_validation_data['winner']
    return(ens_x_train, ens_y_train, ens_x_val, ens_y_val)


def _rec_progress(training_loss, test_loss, training_acc, test_acc, history, show = False):
    training_loss, test_loss, training_acc, test_acc = _add_hist_res(training_loss, test_loss, training_acc, test_acc, history)
    
    # Create count of the number of epochs
    if show:
        epoch_count = range(1, len(training_loss) + 1)
        fig, ax1 = plt.subplots()
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Log Loss')
        ax1.plot(epoch_count, training_loss, 'b--', alpha = .5, label = 'Training Loss')
        ax1.plot(epoch_count, test_loss, 'b-', alpha = .8, label = 'Validation Loss')
        ax1.tick_params(axis='y')
        ax1.legend(loc=0)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.set_ylabel('Accuracy') 
        ax2.plot(epoch_count, training_acc, 'r--', alpha = .5, label = 'Training Accuracy')
        ax2.plot(epoch_count, test_acc, 'r-', alpha = .8, label = 'Validation Accuracy')
        
        ax2.tick_params(axis='y')
        ax2.legend(loc=1)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
    return(training_loss, test_loss, training_acc, test_acc)


def create_ensemble():
    ensemble = Sequential()
    #First Hidden Layer
    ensemble.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=8))
    #Second  Hidden Layer
    ensemble.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
    #Output Layer
    ensemble.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    ensemble.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    return(ensemble)


#model = ensembled_predictor()
#data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data.csv'))    
#data = data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
#data['winner'] = data['winner'].apply(lambda x: 0 if x == -1 else x)
#feature_data = data.loc[data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) < datetime(2018, 1, 1)]
#_add_fit_models(model, feature_data)
#test_data = data.loc[data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) >= datetime(2018, 1, 1)]
#validation_data = test_data.loc[test_data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) >= datetime(2019, 1, 1)]
#test_data = test_data.loc[test_data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) < datetime(2019, 1, 1)]


early_stop = EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor='val_loss',
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-4,
        # "no longer improving" being further defined as "for at least 2 epochs"
        mode = 'max',
        patience=5,
        verbose=0)

    
ensemble = create_ensemble()
model = ensembled_predictor()
data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data.csv'))    
data = data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
data['winner'] = data['winner'].apply(lambda x: 0 if x == -1 else x)
new_data = data.loc[data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) >= datetime(2019, 1, 1)]

data = data.loc[data['fight_date'].apply(lambda x: datetime.strptime(x.split(' ')[0], '%Y-%m-%d')) < datetime(2019, 1, 1)]
feature_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = .5)
training_loss, test_loss, training_acc, test_acc = [], [], [], []

holdout_loss, holdout_acc = [], []
avg_training_loss, avg_test_loss, avg_training_acc, avg_test_acc = [], [], [], []

for i in range(2000):
    ens_x_train, ens_y_train, ens_x_val, ens_y_val = _gen_new_data(model, data, feature_splitter)
    ens_x_new, ens_y_new = _comb_feature_pred(model, new_data), new_data['winner']

    history = ensemble.fit(ens_x_train, ens_y_train, 
                   batch_size=20, 
                   epochs=10,
                   validation_data=(ens_x_val, ens_y_val),
                   callbacks=[early_stop],
                   verbose = 0)
    training_loss, test_loss, training_acc, test_acc = _rec_progress(training_loss, test_loss, training_acc, test_acc, history)
    eval_loss, eval_acc = ensemble.evaluate(ens_x_new, ens_y_new, verbose = 0)
    holdout_loss.append(eval_loss)
    holdout_acc.append(eval_acc)
    avg_training_loss.append(np.mean(history.history['loss']))
    avg_test_loss.append(np.mean(history.history['val_loss']))
    avg_training_acc.append(np.mean(history.history['acc']))
    avg_test_acc.append(np.mean(history.history['val_acc']))
    
    epoch_count = range(1, i + 502)
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log Loss')
    ax1.plot(epoch_count, avg_training_loss, 'b:', alpha = .3, label = 'Avg Training Loss')
    ax1.plot(epoch_count, avg_test_loss, 'b--', alpha = .5, label = 'Avg Validation Loss')
    ax1.plot(epoch_count, holdout_loss, 'b-', alpha = 1, label = 'Test Loss')
    ax1.tick_params(axis='y')
    ax1.legend(loc=0)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    ax2.set_ylabel('Accuracy') 
    ax2.plot(epoch_count, avg_training_acc, 'r:', alpha = .3, label = 'Avg Training Accuracy')
    ax2.plot(epoch_count, avg_test_acc, 'r--', alpha = .5, label = 'Avg Validation Accuracy')
    ax2.plot(epoch_count, holdout_acc, 'r-', alpha = 1, label = 'Test Accuracy')
    
    ax2.tick_params(axis='y')
    ax2.legend(loc=1)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


#from sklearn.linear_model import LogisticRegressionCV
#import math
#
#log = LogisticRegressionCV(fit_intercept = False, class_weight = 'balanced')
#log.fit(feature_preds, test_data['winner'])
#
#log.C_
#[math.exp(i) for i in log.coef_[0]]
#log.decision_function(feature_preds)
#
#log.get_params()

#    PSQL = db_connection('psql')            
#    pred_df = pd.DataFrame(Y_test)        
#    for mod_name, model in self.models.items():
#        model_preds = model.predict_proba(X_test)
#        model_preds = [i[1] for i in model_preds]
#        pred_df[mod_name] = model_preds
#        
#    vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
#    vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
#    vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)
#    vegas_pred_df = pred_df.join(vegas_preds).dropna()
#    vegas_pred_df.reset_index(inplace = True)
#    return(vegas_pred_df)
    
    
    









