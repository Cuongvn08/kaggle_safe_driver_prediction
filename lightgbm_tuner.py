import lightgbm as lgb
import pytz
from datetime import datetime

num_boost_round = 5000
nfold = 5
metrics = {'mae'}
early_stopping_rounds = 10


def tune(data_x, data_y):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n\n{0} Tuning lightgbm parameters ...'.format(date_time))
    
    d_train = lgb.Dataset(data_x, label=data_y)
    
    # lightgbm params
    params = {
        'objective': 'binary',
        'metric' : 'mae',
        'silent' : True
    }
    
    # tune learning rate
    best_learning_rate = _tune_learning_rate(params, d_train)
    params['learning_rate'] = best_learning_rate
    
    # tune max_depth and num_samples_split
    best_max_depth, best_num_samples_split = \
                            _tune_max_depth__num_samples_split(params, d_train)
    params['max_depth'] = best_max_depth
    params['num_samples_split'] = best_num_samples_split
    
    # tune min_samples_leaf
    best_min_samples_leaf = _tune_min_samples_leaf(params, d_train)
    params['min_samples_leaf'] = best_min_samples_leaf
    
    # tune max_features
    best_max_features = _tune_max_features(params, d_train)
    params['max_features'] = best_max_features
    
    # tune subsample
    best_subsample = _tune_subsample(params, d_train)
    params['sub_sample'] = best_subsample
    
    # end
    print('The end.')
    
    
def _tune_learning_rate(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning learning rate ...'.format(date_time))
    
    learning_rate_list = [0.2, 0.1, 0.05, 0.025, 0.005, 0.0025]
    min_mae = float("Inf")
    best_learning_rate = learning_rate_list[0]

    for learning_rate in learning_rate_list:
        # update params
        params['learning_rate'] = learning_rate
        
        # run cv
        cv_results = lgb.cv(params,
                            d_train,
                            num_boost_round = num_boost_round,
                            folds = nfold,
                            metrics = metrics,
                            early_stopping_rounds = early_stopping_rounds)
        
        # print
        mae = cv_results['metric1-mean'].min()
        rounds = cv_results['metric1-mean'].argmin()
        print('learning_rate:', learning_rate, '; mae:',mae, '; rounds:', rounds)
        
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_learning_rate = learning_rate
        
    print('best learning rate:', best_learning_rate)
    print('min mae:', min_mae)
    return best_learning_rate


def _tune_max_depth__num_samples_split(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning max depth and num samples split ...'.format(date_time))


def _tune_min_samples_leaf(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning min samples leaf ...'.format(date_time))    


def _tune_max_features(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning max features ...'.format(date_time))
    

def _tune_subsample(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning subsample ...'.format(date_time))



    