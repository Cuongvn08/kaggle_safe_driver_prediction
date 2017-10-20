import xgboost as xgb
import pytz
from datetime import datetime

num_boost_round = 5000
nfold = 5


def tune(data_x, data_y):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n\n{0} Tuning xgboost parameters ...'.format(date_time))

    d_train = xgb.DMatrix(data_x, label=data_y)

    params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1
    }
    
    # tune eta
    if True:
        best_eta = _tune_eta(params, d_train)
        params['eta'] = best_eta
    
    # tune max_depth and min_child_weight
    if True:
        best_max_depth, best_min_child_weight = \
                        _tune_max_depth__min_child_weight(params, d_train)
        params['max_depth'] = best_max_depth
        params['min_child_weight'] = best_min_child_weight
    
    # tune subsample and colsample_bytree
    if True:
        best_subsample, best_colsample_bytree = \
                        _tune_subsample__colsample_bytree(params, d_train)
        params['subsample'] = best_subsample
        params['colsample_bytree'] = best_colsample_bytree
    
    # tune alpha and lambda
    if True:
        best_alpha, best_lambda = _tune_alpha_lambda(params, d_train)
        params['subsample'] = best_alpha
        params['colsample_bytree'] = best_lambda
        
    # end
    print('The end.')
    
    
def _tune_eta(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning eta ...'.format(date_time))
        
    eta_list = [0.2, 0.1, 0.05, 0.025, 0.005, 0.0025]
    min_mae = float("Inf")
    best_eta = eta_list[0]
    
    for eta in eta_list:
        # update params
        params['eta'] = eta

        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=nfold,
            metrics={'mae'},
            early_stopping_rounds=10
        )
    
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('eta:', eta, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_eta = eta
                    
    print('best eta:', best_eta)
    print('min mae:', min_mae)
    return best_eta


def _tune_max_depth__min_child_weight(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning max_depth and min_child_weight ...'.format(date_time))
    
    max_depth_list = list(range(5,10))
    min_child_weight_list = list(range(1,5))
    min_mae = float("Inf")
    best_max_depth = max_depth_list[0]
    best_min_child_weight = min_child_weight_list[0]
    
    for max_depth, min_child_weight in zip(max_depth_list, min_child_weight_list):
        # update params
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=nfold,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('max_depth:', max_depth, '; min_child_weight:', min_child_weight, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_max_depth = max_depth
            best_min_child_weight = min_child_weight
                    
    print('best max_depth:', best_max_depth)
    print('best min_child_weight:', best_min_child_weight)
    print('min mae:', min_mae)
    return best_max_depth, best_min_child_weight
        

def _tune_subsample__colsample_bytree(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning subsample and colsample_bytree ...'.format(date_time))
    
    subsample_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    colsample_bytree = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    min_mae = float("Inf")
    best_subsample = subsample_list[0]
    best_colsample_bytree = colsample_bytree[0]
    
    for subsample, colsample_bytree in zip(subsample_list, colsample_bytree):
        # update params
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=nfold,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('subsample:', subsample, '; colsample_bytree:', colsample_bytree, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_subsample = subsample
            best_colsample_bytree = colsample_bytree
                    
    print('best subsample:', best_subsample)
    print('best colsample_bytree:', best_colsample_bytree)
    print('min mae:', min_mae)
    return best_subsample, best_colsample_bytree


def _tune_alpha_lambda(params, d_train):
    date_time = datetime.now(tz=pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S')
    print('\n{0} Tuning alpha and lambda ...'.format(date_time))
    
    alpha_list = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    lambda_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    min_mae = float("Inf")
    best_alpha = alpha_list[0]
    best_lambda = lambda_list[0]
    
    for alpha, lambdaa in zip(alpha_list, lambda_list):
        # update params
        params['alpha'] = alpha
        params['lambda'] = lambdaa
        
        # run cv
        cv_results = xgb.cv(
            params,
            d_train,
            num_boost_round = num_boost_round,
            nfold=nfold,
            metrics={'mae'},
            early_stopping_rounds=10
        )
        
        # print
        mae = cv_results['test-mae-mean'].min()
        rounds = cv_results['test-mae-mean'].argmin()
        print('alpha:', alpha, '; lambda:', lambdaa, '; mae:',mae, '; rounds:', rounds)
    
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_alpha = alpha
            best_lambda = lambdaa
                    
    print('best alpha:', best_alpha)
    print('best lambda:', best_lambda)
    print('min mae:', min_mae)
    return alpha, lambdaa
