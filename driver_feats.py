import os, sys, platform
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

if platform.system() == 'Darwin':
    while cur_path.split('/')[-1] not in ['fifth_third', 'fifth_third copy', '53_vis']:
        cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
elif platform.system() == 'Windows':
    while cur_path.split('\\')[-1] not in ['fifth_third', 'fifth_third copy', '53_vis']:
        cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))
sys.path.insert(1, os.path.join(cur_path, 'PySpark_Codebase_DC'))
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))

import os, sys
import pandas as pd
from load_es import proccess_data
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
#import operator
#from skmultilearn.model_selection import IterativeStratification
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.metrics import classification_report
import warnings
from progress_bar import progress
import json
from model_vars import new_20, first_training_set

warnings.filterwarnings("ignore")


avail = proccess_data(values = True, _require_intent = False, _inc_intent = False, _drop_missing = True)
labelled = proccess_data(values = True, _require_intent = True, _inc_intent = True, _drop_missing = True)


added_labels = [i for i in labelled.keys() if i not in first_training_set]
missing_labels = [i for i in avail.keys() if i not in labelled.keys()]
print('%i calls needing label' % (len(missing_labels)))

qual_added = [i for i in missing_labels if i not in new_20]
print('%i added due to QUALITY' % (len(qual_added + added_labels)))
quant_added = [i for i in missing_labels if i in new_20]
print('%i added due to QUANTITY' % (len(quant_added)))



def prep_data(data):
    df = pd.DataFrame.from_dict(data).T
    
    
    df = df[['Business Case', 'Agent Sentiment','Agent Cum Tone: Excited','Agent Cum Tone: Impolite','Agent Cum Tone: Polite','Agent Cum Tone: Sad',
             'Agent Cum Tone: Satisfied','Agent Cum Tone: Sympathetic','Agent Cum Tone: frustrated','Customer Sentiment',
             'Customer Cum Tone: Excited','Customer Cum Tone: Impolite','Customer Cum Tone: Polite','Customer Cum Tone: Sad','Customer Cum Tone: Satisfied',
             'Customer Cum Tone: Sympathetic','Customer Cum Tone: frustrated','Cum Tone: Excited','Cum Tone: Frustrated','Cum Tone: Impolite',
             'Cum Tone: Polite','Cum Tone: Sad','Cum Tone: Satisfied','Cum Tone: Sympathetic','agent agreeableness','agent analytical',
             'agent confident','agent conscientiousness','agent emotional range','agent extraversion','agent line count',
             'agent openness','agent tentative','agent word count','agent_anger','agent_disgust','agent_fear','agent_joy',
             'agent_sadness','agreeableness','analytical','anger','confident','conscientiousness',
             'customer agreeableness','customer analytical','customer confident','customer conscientiousness','customer emotional range',
             'customer extraversion','customer line count','customer openness','customer tentative','customer word count',
             'customer_anger','customer_disgust','customer_fear','customer_joy','customer_sadness','disgust','emotional range',
             'extraversion','fear','joy','line count','openness','sadness','tentative','word count']]
    
    drivers = ['Agent Sentiment','Agent Cum Tone: Excited','Agent Cum Tone: Impolite','Agent Cum Tone: Polite','Agent Cum Tone: Sad',
             'Agent Cum Tone: Satisfied','Agent Cum Tone: Sympathetic','Agent Cum Tone: frustrated','Customer Sentiment',
             'Customer Cum Tone: Excited','Customer Cum Tone: Impolite','Customer Cum Tone: Polite','Customer Cum Tone: Sad','Customer Cum Tone: Satisfied',
             'Customer Cum Tone: Sympathetic','Customer Cum Tone: frustrated','Cum Tone: Excited','Cum Tone: Frustrated','Cum Tone: Impolite',
             'Cum Tone: Polite','Cum Tone: Sad','Cum Tone: Satisfied','Cum Tone: Sympathetic','agent agreeableness','agent analytical',
             'agent confident','agent conscientiousness','agent emotional range','agent extraversion','agent line count',
             'agent openness','agent tentative','agent word count','agent_anger','agent_disgust','agent_fear','agent_joy',
             'agent_sadness','agreeableness','analytical','anger','confident','conscientiousness',
             'customer agreeableness','customer analytical','customer confident','customer conscientiousness','customer emotional range',
             'customer extraversion','customer line count','customer openness','customer tentative','customer word count',
             'customer_anger','customer_disgust','customer_fear','customer_joy','customer_sadness','disgust','emotional range',
             'extraversion','fear','joy','line count','openness','sadness','tentative','word count']
    
    
    b_cases = []
    for idx in df.index:
        comb_row = df.loc[idx]
        for intent in comb_row['Business Case']:
            b_cases.append(intent)
    b_cases = set(b_cases)
    
    business_cases = pd.DataFrame()
    for idx in df.index:
        comb_row = df.loc[idx]
        business_intents = []
        for b_intent in b_cases:
            if b_intent in comb_row['Business Case']:
                business_intents.append(1)
            else:
                business_intents.append(0)
        business_cases = business_cases.append(pd.DataFrame(business_intents).T)
    business_cases.columns = list(b_cases)
    business_cases['index'] = df.index
    business_cases.set_index('index', inplace = True)
    
    proc_data = df[drivers].join(business_cases)
    
    
    for col in drivers:
        proc_data[col] = proc_data[col].astype(float)

    return(proc_data, b_cases, drivers)


def _single_core_solver(input_vals):
#   trainx, testx, trainy, testy, model, metric = job
    trainx, testx, trainy, testy, model, metric = input_vals

    test_weights = class_weight.compute_class_weight('balanced',
                                np.unique(trainy),trainy)    
    test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}    

    model.fit(trainx, trainy)   
    
     
    if metric == 'logloss':
        pred = model.predict_proba(testx)
        score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy])
        score *= -1
    elif metric == 'accuracy':
        pred = model.predict(testx)
        score = accuracy_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
    elif metric == 'f1':
        pred = model.predict(testx)
        score = f1_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
    elif metric == 'recall':
        pred = model.predict(testx)
        score = recall_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
    elif metric == 'prec':
        pred = model.predict(testx)
        score = precision_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
        
    return(score)


def _single_core_scorer(input_vals):
#   trainx, testx, trainy, testy, model, metric = job
    trainx, testx, trainy, testy, model, _ = input_vals

    test_weights = class_weight.compute_class_weight('balanced',
                                np.unique(trainy),trainy)    
    test_weights_dict = {i:j for i,j in zip(np.unique(trainy), test_weights)}    

    model.fit(trainx, trainy)   
    
    scores = {}
    for metric in ['accuracy', 'f1', 'recall', 'prec']:
        if metric == 'logloss':
            pred = model.predict_proba(testx)
            score = log_loss(testy, pred, sample_weight = [test_weights_dict[i] for i in testy])
            score *= -1
        elif metric == 'accuracy':
            pred = model.predict(testx)
            score = accuracy_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
        elif metric == 'f1':
            pred = model.predict(testx)
            score = f1_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
        elif metric == 'recall':
            pred = model.predict(testx)
            score = recall_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
        elif metric == 'prec':
            pred = model.predict(testx)
            score = precision_score(testy, pred)#, sample_weight = [test_weights_dict[i] for i in testy])
        scores[metric] = score
    return(scores)
    
    
def cross_validate(x,y,est,scaler, scorer, scores = False):
#    x,y,est,scaler, scorer = X,Y,predictor,StandardScaler(), 'logloss'
    splitter = StratifiedKFold(n_splits = 3, random_state = 53)
    all_folds = []
    for fold in splitter.split(x, y):
        all_folds.append(fold)
    
    jobs = []
    for train, test in all_folds:
        jobs.append([scaler.fit_transform(x.iloc[train]), scaler.fit_transform(x.iloc[test]), y.iloc[train], y.iloc[test], est, scorer])
        
    cv_results = []
    for job in jobs:
        if scores:
            cv_results.append(_single_core_scorer(job))
        else:
            cv_results.append(_single_core_solver(job))
    
    if scores:
        results = [{met: np.mean(i[met]) for i in cv_results} for met in ['accuracy', 'f1', 'recall', 'prec'] ]
    else:
        results = np.mean(cv_results)
    
    return(results)


def log_feat_selector(x_, y_):
    log_cross_val_scores = {}
    for solve in ['newton-cg', 'sag', 'saga', 'lbfgs', 'liblinear']:
        for c in np.logspace(-4,4,20):
            _x,_y = deepcopy(x_), deepcopy(y_)
            estimator = LogisticRegression(class_weight = 'balanced', random_state = 53, C = c, solver = solve)
            log_cross_val_scores[cross_validate(_x,_y,estimator,StandardScaler(), 'logloss')] = {'C': c, 'solver': solve}
    log_params = log_cross_val_scores[max(log_cross_val_scores.keys())]
    
    log_model = LogisticRegression(class_weight = 'balanced', random_state = 53, C = log_params['C'], solver = log_params['solver'])
    
    acc_score = cross_validate(_x,_y,estimator,StandardScaler(), 'accuracy')
    rec_score = cross_validate(_x,_y,estimator,StandardScaler(), 'recall')
    prec_score = cross_validate(_x,_y,estimator,StandardScaler(), 'prec')
    f_score = cross_validate(_x,_y,estimator,StandardScaler(), 'f1')

    log_sfm = SelectFromModel(log_model)
    log_sfm.fit(x_, y_)
    log_sig_feats = [i for i,j in zip(list(x_), log_sfm.get_support()) if j]
    return(log_sig_feats, log_model, max(log_cross_val_scores.keys()), acc_score, rec_score, prec_score, f_score)

def feat_selector_optimization(x_, y_):
#    x_, y_ = proc_data[drivers], proc_data[target_col]
    
    cross_val_scores = {}
    
    tot_runs = 10 * 7 * 5 * 6
    run_num = 0
    for trees in np.linspace(10,100,10):
        for depth in np.linspace(3, 21, 7):
            for samples in np.linspace(.1, .9, 5):
                for feats in np.linspace(.1,1, 6):
                    run_num += 1
                    _x,_y = deepcopy(x_), deepcopy(y_)
                    estimator = ExtraTreesClassifier(max_features = feats, min_samples_split = samples, max_depth = int(depth), n_estimators = int(trees), random_state = 53)
                    cross_val_scores[cross_validate(_x,_y,estimator,StandardScaler(), 'logloss')] = {'max_features': feats, 'min_samples_split': samples, 'max_depth': depth, 'n_estimators': trees}
                    
                    
                    progress(run_num, tot_runs)   
    print('\n')
    best_params = cross_val_scores[max(cross_val_scores.keys())]
    
    return(best_params)


def ml_cv(_b_cases, _proc_data, _drivers):
#    _b_cases, _proc_data, _drivers = b_cases_, proc_data_, drivers_
    pred_scores = {}
    for target_col in list(_b_cases):
    #target_col = list(b_cases)[0]
        sig_feats, base_model, base_logloss, base_acc, base_rec, base_prec, base_f1 = log_feat_selector(_proc_data[_drivers], _proc_data[target_col])
        predictor = CatBoostClassifier()
        X = _proc_data[sig_feats]
        Y = _proc_data[target_col]
        
        base_feature_perf = cross_validate(X,Y,base_model,StandardScaler(), 'logloss', scores = True)
        base_feature_perf = {k: v for d in base_feature_perf for k, v in d.items()}
        base_catboost_perf = cross_validate(X,Y,predictor,StandardScaler(), 'logloss', scores = True)
        base_catboost_perf = {k: v for d in base_catboost_perf for k, v in d.items()}
    
        
        pred_scores[target_col] = {'base': {'logloss': base_logloss, 'accuracy': base_acc, 'recall': base_rec, 'precision': base_prec, 'f1': base_f1}, 'feature_selection': base_feature_perf, 'base_catboost': base_catboost_perf, 'significant_features': sig_feats}
    return(pred_scores)
#    with open(os.path.join(cur_path, 'modelling', 'structured_ml_cv_v2.json'), 'w') as fp:
#        json.dump(pred_scores, fp)
     

def score_model(train, test, b_cases, drivers):
#    train, test, b_cases, drivers = proc_data_train, proc_data_test, b_cases_, sig_feats
    pred_scores = {}
    
    for target_col  in list(b_cases):
        if target_col in ['Authentication', 'None']:
            continue
        train_x = deepcopy(StandardScaler().fit_transform(train[drivers[target_col]]))
        train_y = deepcopy(train[target_col])
        test_x = deepcopy(StandardScaler().fit_transform(test[drivers[target_col]]))
        test_y = deepcopy(test[target_col])
        
        predictor = CatBoostClassifier()
        
        predictor.fit(train_x, train_y)
        predictions = predictor.predict(test_x)
    
        predictor.save_model(os.path.join(cur_path, 'modelling', 'models', '%s_classifier.mod' % (target_col.replace('/','_'))))
#        pred2 = CatBoostClassifier().load_model(os.path.join(cur_path, 'modelling', 'models', '%s_classifier.mod' % (target_col)))
        
        predictions = [i for i,j in zip(predictions, test_y.values) if j == j]
        
        pred_scores[target_col] = accuracy_score(test_y.dropna(), predictions)
        
    return(pred_scores)

    
if __name__ == '__main__':
    proc_data_, b_cases_, drivers_ = prep_data(labelled)
    
    pred_scores = ml_cv(b_cases_, proc_data_, drivers_)
    with open(os.path.join(cur_path, 'modelling', 'structured_ml_cv.json'), 'w') as fp:
        json.dump(pred_scores, fp)
        
#    cv_scores = ml_cv(b_cases_, proc_data_, drivers_)
#    
#    sig_feats = {i:j['significant_features'] for i,j in cv_scores.items()}
#    
#    testing_data = deepcopy(avail)
#    
#    for k,v in {i:j['Business Case'] for i,j in labelled.items()}.items():
#        testing_data[k]['Business Case'] = v
#    
#    for k,v in pd.read_csv(os.path.join(cur_path, 'business_case_test.csv'))[['filename', 'Business Case']].values:
#        testing_data[k.replace('\ufeff', '')]['Business Case'] = [i.strip() for i in v.split(',')]    
#    
#    drop_idx = []
#    for k in testing_data.keys():
#        if 'Business Case' not in testing_data[k].keys():
#            drop_idx.append(k)
#    for idx in drop_idx:
#        testing_data.pop(idx)
#        
#        
#    proc_data_, b_cases_, drivers_ = prep_data(testing_data)
#    
#    proc_data_test = proc_data_.loc[missing_labels]
#    proc_data_train = proc_data_.loc[labelled.keys()]
#    
#    test_scores = score_model(proc_data_train, proc_data_test, b_cases_, sig_feats)
#
#    for k,v in test_scores.items():
#        cv_scores[k]['test'] = {'accuracy': v}
#    
#    with open(os.path.join(cur_path, 'modelling', 'structured_ml_cv_v2.json'), 'w') as fp:
#        json.dump(cv_scores, fp)     





