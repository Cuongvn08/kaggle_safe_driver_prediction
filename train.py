# -*- coding: utf-8 -*-

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import gc
from xgboost_tuner import XgboostTuner
from sklearn.model_selection import StratifiedKFold


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
    
    train_path      = 'C:/data/kaggle/safe_driver_prediction/train.csv'
    test_path       = 'C:/data/kaggle/safe_driver_prediction/test.csv'
    submission_path = 'C:/data/kaggle/safe_driver_prediction/sample_submission.csv'
    IS_PARAMS_TUNNING = False
    
    def __str__(self):
        return self.value
        

################################################################################    
## STEP1: process data
def process_data():
    print('\n\nSTEP1: processing data ...')
        
    global data_x
    global data_y
    global test_x
    
    # load data
    train_df, test_df = _load_data()
    
    # analyze
        
    # fill NA
    _fill_NA(train_df)
    _fill_NA(test_df)
    
    # encode features: TBD
        
    # add features: TBD
    
    # remove outliers: TBD
    
    # select features: TBD
    
    # prepare train and valid data: TBD
    print('\nPreparing train and test data ...')
    
    data_y = train_df['target']
    data_x = train_df.drop(['id', 'target'], axis=1)
    test_x = test_df[data_x.columns]

    data_x = data_x.values
    data_y = data_y.values
    test_x = test_x.values
    
    print('train x shape: ', data_x.shape)
    print('train y shape: ', data_y.shape)
    print('test x shape : ', test_x.shape)
    
    # release
    del train_df
    del test_df
    gc.collect()
    
    
def _load_data():
    print('\nLoading data ...')
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

def _analyze(df):
    print('\nAnalyzing data ...')
    
    # show correlation
    if(1):
        xgb_params = {
            'eta': 0.05,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'silent': 1,
            'seed' : 0
        }
        
        train_y = df['target'].values
        train_x = df.drop(['id', 'target'], axis=1)
        
        dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
        model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
        
        featureImportance = model.get_fscore()
        features = pd.DataFrame()
        features['features'] = featureImportance.keys()
        features['importance'] = featureImportance.values()
        features.sort_values(by=['importance'],ascending=False,inplace=True)
        fig,ax= plt.subplots()
        fig.set_size_inches(10,20)
        plt.xticks(rotation=90)
        sns.barplot(data=features,x="importance",y="features",ax=ax,orient="h",color="#34495e")      
        plt.show()
        
        del train_x
        del train_y
        gc.collect()
    
def _fill_NA(df):
    print('\nFilling data ...')
    
    na_ratio = ((df.isnull().sum() / len(df)) * 100).sort_values(ascending=False)
    print('NA ratio: ')
    print(na_ratio) 
    
    for feature in df:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].fillna("None")
        else:
            df[feature] = df[feature].fillna(-1)
    
def _encode_features(df):
    print('\nEncoding features ...')
        
def _add_features(df):
    print('\nAdding features ...')
    
def _remove_outliers(df):
    print('\nRemoving features ...')
    
def _select_features(df):
    print('\nSelecting features ...')
    
    
################################################################################        
## STEP2: build model
def build_model():
    print('\n\nSTEP2: building model ...')
    
    global xgb_params
    global lgb_params
    
    # xgboost params
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.025,
        'max_depth': 6, 
        'min_child_weight': 2,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'alpha': 1.6,
        'lambda': 10.0,
    }

    # lightgbm params
    lgb_params = {
        'objective': 'binary',
        'metric' : 'auc',
        'learning_rate' : 0.0025,
        'max_depth' : 100,
        'num_leaves' : 32,
        'feature_fraction' : .85,
        'bagging_fraction' : .95,
        'bagging_freq' : 8,
        'verbosity' : 0
    }

        
################################################################################    
## STEP3: train and predict with kfold   
def train_predict():
    print('\n\nSTEP3: training ...')

    global xgb_submission
    xgb_submission = pd.read_csv(submission_path)
    xgb_submission['target'] = 0
    
    kfold = 5
    
    # xgboost
    d_test = xgb.DMatrix(test_x)
    skf = StratifiedKFold(n_splits=kfold, random_state=0)
    
    for i, (train_index, valid_index) in enumerate(skf.split(data_x, data_y)):
        print('xgboost kfold:', i + 1)

        train_x, valid_x = data_x[train_index], data_x[valid_index]
        train_y, valid_y = data_y[train_index], data_y[valid_index]
        
        d_train = xgb.DMatrix(train_x, train_y) 
        d_valid = xgb.DMatrix(valid_x, valid_y)         
        
        evals = [(d_train, 'train'), (d_valid, 'valid')]
        xgb_model = xgb.train(xgb_params, d_train, 
                              num_boost_round = 2000, evals = evals, 
                              early_stopping_rounds = 100, feval = _gini_xgb, 
                              maximize = True, verbose_eval = 100)        
        
        xgb_submission['target'] += xgb_model.predict(d_test, ntree_limit = xgb_model.best_ntree_limit)
        
    xgb_submission = xgb_submission / kfold
    gc.collect()
    
    # lightgbm
    # TBD
        
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
    print('\n\nSTEP5: generating submission ...')

    xgb_submission.to_csv('sub{}.csv'.format(datetime.now().\
            strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.5f')
    
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
        #xgb_tuner = XgboostTuner(data_X, data_Y)
        #xgb_tuner.tune()
        
        # lightgbm parameters tuning
        #xgb_tuner = XgboostTuner(data_X, data_Y)
        #xgb_tuner.tune()        
        pass

################################################################################
if __name__ == "__main__":
    main()
    print('\n\n\nThe end.')
    