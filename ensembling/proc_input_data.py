import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))
sys.path.insert(2, os.path.join(cur_path, 'lib','LightGBM', 'python-package'))

from joblib import load
import pandas as pd
from _connections import db_connection
from db.pop_psql import pg_query
from progress_bar import progress


def conv_to_ml_odds(prob):
    if prob > .5:
        odds = (prob/(1-prob))*100
    else:
        odds = ((1-prob)/prob)*100
    return(odds)
    
    
PSQL = db_connection('psql')

alpha_data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_est_training.csv'))
alpha_data = alpha_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
X0 = alpha_data[[i for i in list(alpha_data) if i != 'winner']]
Y0 = alpha_data['winner'].apply(lambda x: x if x == 1 else 0)


beta_data = pd.read_csv(os.path.join(cur_path, 'data', 'pred_data_winner_ens_training.csv'))
beta_data = beta_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
X2 = beta_data[[i for i in list(beta_data) if i != 'winner']]
Y2 = beta_data['winner'].apply(lambda x: x if x == 1 else 0)


domain = 'winner'
final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'models')
tot_mods = len(os.listdir(final_model_folder))
pred_df = pd.DataFrame(Y2)


for mod_num, (mod_name) in enumerate(os.listdir(final_model_folder)):
    progress(mod_num + 1, tot_mods, mod_name)  
    if mod_name == '.DS_Store':
        continue
    model_path = os.listdir(os.path.join(final_model_folder, mod_name))
    model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
    model.fit(X0, Y0)
    model_preds = model.predict_proba(X2)
    model_preds = [i[1] for i in model_preds]
#    mod_preds.rename(columns = {0: mod_name}, inplace = True)
    pred_df[mod_name] = model_preds


vegas_preds = pg_query(PSQL.client, "SELECT * from ufc.winner_consensus_odds;")
vegas_preds.columns = ['bout_id', 'fighter_id', 'VEGAS']
vegas_preds.set_index(['bout_id', 'fighter_id'], inplace = True)
vegas_pred_df = pred_df.join(vegas_preds).dropna()


vegas_pred_df.to_csv(os.path.join(cur_path, 'data', 'pred_data_winner_ens_tuning.csv'))




train_data = pd.read_csv(os.path.join(cur_path, 'data', 'winner_data_validation.csv'))
train_data = train_data.sort_values('bout_id').set_index(['bout_id', 'fighter_id'])
X_train = train_data[[i for i in list(train_data) if i != 'winner']]
Y_train = train_data['winner'].apply(lambda x: x if x == 1 else 0)


domain = 'winner'
final_model_folder = os.path.join(cur_path, 'model_tuning', 'modelling', domain, 'final', 'models')
tot_mods = len(os.listdir(final_model_folder))


for mod_num, (mod_name) in enumerate(os.listdir(final_model_folder)):
    progress(mod_num + 1, tot_mods, mod_name)  
    if mod_name == '.DS_Store':
        continue
    model_path = os.listdir(os.path.join(final_model_folder, mod_name))
    model = load(os.path.join(final_model_folder, mod_name, model_path[0]))
    model.fit(X_train, Y_train)
    dump(model, os.path.join(cur_path, 'ensembling', 'model', 'features', '%s.pkl' % (mod_name)))
    
