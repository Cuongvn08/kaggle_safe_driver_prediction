# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import gc
import xgboost_tuner
import lightgbm_tuner
import printer as ptr


import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


################################################################################
## STEP0: do setting
class Settings(Enum):
    global train_path
    global test_path
    global submission_path
    global IS_PARAMS_TUNNING

   
    train_path      = '/data/kaggle/safe_driver_prediction/train.csv'
    test_path       = '/data/kaggle/safe_driver_prediction/test.csv'
    submission_path = '/data/kaggle/safe_driver_prediction/sample_submission.csv'
    IS_PARAMS_TUNNING = False

    
    def __str__(self):
        return self.value
        

################################################################################    
## STEP1: process data
def process_data():
    ptr.print_log('STEP1: processing data ...')
    
    global data_x
    global data_y
    global test_x
    
    # load data
    train_df, test_df = _load_data()
        
    # fill NA: nothing to do because the missing cells were already filled with -1
    
    # encode features

    # add features
    _add_features(train_df)
    _add_features(test_df)
    
    # remove outliers
    _remove_outliers(train_df)
        
    # select and drop features
    _select_drop_features(train_df)
    _select_drop_features(test_df)
    
    # prepare train and valid data
    ptr.print_log('Preparing train and test data ...')
    
    data_y = train_df['target']
    data_x = train_df.drop(['id', 'target'], axis=1)
    test_x = test_df[data_x.columns]

    data_x = data_x.values
    data_y = data_y.values
    test_x = test_x.values
    
    ptr.print_log('train x shape: {}'.format(data_x.shape))
    ptr.print_log('train y shape: {}'.format(data_y.shape))
    ptr.print_log('test x shape : {}'.format(test_x.shape))
    
    # release
    del train_df
    del test_df
    gc.collect()
        
def _load_data():
    ptr.print_log('Loading data ...')
    
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    
    for c in train_df.select_dtypes(include=['float64']).columns:
        train_df[c] = train_df[c].astype(np.float32)
        test_df[c]  = test_df[c].astype(np.float32)
        
    for c in train_df.select_dtypes(include=['int64']).columns[2:]:
        train_df[c] = train_df[c].astype(np.int32)
        test_df[c]  = test_df[c].astype(np.int32)
    
    print('train shape: ', train_df.shape)
    print('test shape : ', test_df.shape)    
    
    return train_df, test_df
    
def _fill_NA(df):
    ptr.print_log('Filling data ...')
    
    for feature in df:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna("None")
        else:
            df[feature] = df[feature].fillna(-1)
    
def _encode_features(df):    
    ptr.print_log('Encoding features ...')
    
def _add_features(df):
    ptr.print_log('Adding features ...')
    
def _remove_outliers(df):
    ptr.print_log('Removing features ...')
    
    df.drop(df[df['ps_car_12'] > 1.0].index, axis=0, inplace = True)
    df.drop(df[df['ps_car_12'] < 0.25].index, axis=0, inplace = True)
    df.drop(df[df['ps_car_13'] > 3.0].index, axis=0, inplace = True)
    df.drop(df[df['ps_car_14'] < 0.0].index, axis=0, inplace = True)
    df.drop(df[df['ps_reg_03'] > 3.0].index, axis=0, inplace = True)
    
def _select_drop_features(df):
    ptr.print_log('Selecting and dropping features according to feature importance ...')
    
    drop_features = ['ps_ind_10_bin',
                     'ps_ind_11_bin',
                     'ps_calc_16_bin',
                     'ps_calc_15_bin',
                     'ps_calc_20_bin',
                     'ps_calc_18_bin',
                     'ps_ind_13_bin',
                     'ps_ind_18_bin',
                     'ps_calc_19_bin',
                     'ps_calc_17_bin',
                     'ps_car_08_cat',
                     'ps_ind_09_bin',
                     'ps_car_02_cat',
                     'ps_ind_14']
    
    df.drop(drop_features, axis=1, inplace=True)

    
################################################################################        
## STEP2: build model
def build_model():
    ptr.print_log('STEP2: building model ...')
    
    global xgb_params
    global xgb_rounds
    
    global lgb_params
    global lgb_rounds
    
    # xgboost params
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.005,
        'max_depth': 5, 
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 2.4,
        'lambda': 14.0,
        'silent': 1
    }
        
    # lightgbm params
    lgb_params = {
        'objective': 'binary',
        'metric' : 'auc',
        'learning_rate' : 0.0025,
        'max_depth' : 6,
        'num_leaves' : 50,
        'min_data_in_leaf' : 200,
        'max_bin' : 50,
        'verbosity' : 0
    }
    
        
################################################################################    
## STEP3: train and predict with kfold   
def train_predict():
    ptr.print_log('STEP3: training ...')
    
    global xgb_pred
    global lgb_pred
        
    kfold = 5
    
    # xgboost
    xgb_pred = 0.0
    d_test = xgb.DMatrix(test_x)
    skf = StratifiedKFold(n_splits=kfold)
    
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y)):
        ptr.print_log('xgboost kfold: {}'.format(i+1))

        train_x, valid_x = data_x[train_index], data_x[valid_index]
        train_y, valid_y = data_y[train_index], data_y[valid_index]
                
        d_train = xgb.DMatrix(train_x, train_y) 
        d_valid = xgb.DMatrix(valid_x, valid_y)
        
        evals = [(d_train, 'train'), (d_valid, 'valid')]
        evals_result = {}
        xgb_model = xgb.train(xgb_params, d_train, 
                              num_boost_round = 10000,
                              evals = evals, feval = _gini_xgb,
                              evals_result = evals_result,
                              maximize = True, early_stopping_rounds = 100, verbose_eval = 100)
                
        xgb_pred += xgb_model.predict(d_test, ntree_limit = xgb_model.best_ntree_limit)
        
        if False:
            result_train_gini = evals_result['train']
            result_valid_gini = evals_result['valid']
            for j in range(xgb_model.best_iteration+1):
                train_gini = result_train_gini['gini'][j]
                valid_gini = result_valid_gini['gini'][j]
                ptr.print_log('round, train_gini, valid_gini: {0:04}, {1:0.6}, {2:0.6}'.format(j, train_gini, valid_gini), False)
            
    xgb_pred = xgb_pred / kfold
    gc.collect()
        
    # lightgbm
    lgb_pred = 0.0
    skf = StratifiedKFold(n_splits=kfold)

    if False:
        for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y)):
            ptr.print_log('lightgbm kfold: {}'.format(i+1))
            
            train_x, valid_x = data_x[train_index], data_x[valid_index]
            train_y, valid_y = data_y[train_index], data_y[valid_index]
            
            d_train = lgb.Dataset(train_x, train_y) 
            d_valid = lgb.Dataset(valid_x, valid_y)
    
            valid_sets = [d_train, d_valid]
            valid_names = ['train', 'valid']
            evals_result = {}
            
            lgb_model = lgb.train(lgb_params, d_train,
                                  num_boost_round = 10000,
                                  valid_sets = valid_sets, valid_names = valid_names,
                                  feval = _gini_lgb, evals_result = evals_result,
                                  early_stopping_rounds = 100, verbose_eval = 100)
            
            lgb_pred += lgb_model.predict(test_x, num_iteration = lgb_model.best_iteration)
            
            result_train_gini = evals_result['train']
            result_valid_gini = evals_result['valid']
            for j in range(lgb_model.best_iteration+1):
                train_gini = result_train_gini['gini'][j]
                valid_gini = result_valid_gini['gini'][j]
                ptr.print_log('round, train_gini, valid_gini: {0:04}, {1:0.6}, {2:0.6}'.format(j, train_gini, valid_gini), False)        
        
    lgb_pred = lgb_pred / kfold
    gc.collect()

def _gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def _gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', _gini(y, pred) / _gini(y, y)

def _gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = _gini(y, preds) / _gini(y, y)
    return 'gini', score, True
        

################################################################################    
## STEP4: generate submission    
def generate_submission():
    ptr.print_log('STEP4: generating submission ...')

    submission = pd.read_csv(submission_path)
    
    XGB_WEIGHT_LIST = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    LGB_WEIGHT_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for XGB_WEIGHT, LGB_WEIGHT in zip(XGB_WEIGHT_LIST, LGB_WEIGHT_LIST):
        submission['target'] = xgb_pred*XGB_WEIGHT + lgb_pred*LGB_WEIGHT
        submission.to_csv('sub{}_{}_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), XGB_WEIGHT, LGB_WEIGHT), 
                          index=False, float_format='%.5f')
    
    
################################################################################
## main
def main():
    process_data()
    if IS_PARAMS_TUNNING is False:
        build_model()
        train_predict()
        generate_submission()
    else:
        # xgboost parameters tuning
        xgboost_tuner.tune(data_x, data_y)
        
        # lightgbm parameters tuning
        lightgbm_tuner.tune(data_x, data_y)
        
        
################################################################################
if __name__ == "__main__":
    ptr.print_log('TRAINER')
    main()
    ptr.print_log('THE END.')
    