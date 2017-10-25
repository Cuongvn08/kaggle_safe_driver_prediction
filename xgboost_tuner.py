'''
http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html
'''

import xgboost as xgb
import printer as ptr


num_boost_round = 5000
nfold = 5
metrics = {'auc'}
early_stopping_rounds = 10


# tune
def tune(data_x, data_y):
    ptr.print_log('Tuning xgboost parameters ...')

    d_train = xgb.DMatrix(data_x, label=data_y)

    params = {
        'objective': 'binary:logistic',
        'silent': 1
    }
    
    # tune eta
    best_eta = _tune_eta(params, d_train)
    params['eta'] = best_eta
    
    # tune max_depth and min_child_weight
    best_max_depth, best_min_child_weight = _tune_max_depth__min_child_weight(params, d_train)
    params['max_depth'] = best_max_depth
    params['min_child_weight'] = best_min_child_weight
    
    # tune subsample and colsample_bytree
    best_subsample, best_colsample_bytree = _tune_subsample__colsample_bytree(params, d_train)
    params['subsample'] = best_subsample
    params['colsample_bytree'] = best_colsample_bytree
    
    # tune alpha and lambda
    best_alpha, best_lambda = _tune_alpha_lambda(params, d_train)
    params['subsample'] = best_alpha
    params['colsample_bytree'] = best_lambda
        
    # end
    ptr.print_log('XGBOOST TUNER was finished.')
    
    
# tune eta    
def _tune_eta(params, d_train):
    ptr.print_log('Tuning eta ...')
    
    eta_list = [0.2, 0.1, 0.05, 0.025, 0.005, 0.0025]
    
    max_auc = 0.0
    best_eta = eta_list[0]
    
    for eta in eta_list:
        # update params
        params['eta'] = eta

        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('eta: {}; auc: {}; rounds: {}'.format(eta, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_eta = eta
                    
    ptr.print_log('best eta: {0}'.format(best_eta))
    ptr.print_log('max auc: {0}'.format(max_auc))
    
    return best_eta


# tune max_depth and min_child_weight
def _tune_max_depth__min_child_weight(params, d_train):
    ptr.print_log('Tuning max_depth and min_child_weight ...')
    
    max_depth_list = list(range(5,10))
    min_child_weight_list = list(range(1,5))
    
    max_auc = 0.0
    best_max_depth = max_depth_list[0]
    best_min_child_weight = min_child_weight_list[0]
    
    for max_depth, min_child_weight in zip(max_depth_list, min_child_weight_list):
        # update params
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('max_depth: {}; min_child_weight: {}; auc: {}; rounds: {}'.\
                      format(max_depth, min_child_weight, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_depth = max_depth
            best_min_child_weight = min_child_weight
                    
    ptr.print_log('best max_depth: {0}'.format(best_max_depth))
    ptr.print_log('best min_child_weight: {0}'.format(best_min_child_weight))
    ptr.print_log('max auc: {0}'.format(max_auc))
    
    return best_max_depth, best_min_child_weight
        

# tune subsample and colsample_bytree
def _tune_subsample__colsample_bytree(params, d_train):
    ptr.print_log('Tuning subsample and colsample_bytree ...')
    
    subsample_list = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    colsample_bytree = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    max_auc = 0.0
    best_subsample = subsample_list[0]
    best_colsample_bytree = colsample_bytree[0]
    
    for subsample, colsample_bytree in zip(subsample_list, colsample_bytree):
        # update params
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('subsample: {}; colsample_bytree: {}; auc: {}; rounds: {}'.\
              format(subsample, colsample_bytree, auc, rounds))
                
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_subsample = subsample
            best_colsample_bytree = colsample_bytree
                    
    ptr.print_log('best subsample: {0}'.format(best_subsample))
    ptr.print_log('best colsample_bytree: {0}'.format(best_colsample_bytree))
    ptr.print_log('max auc: {0}'.format(max_auc))
            
    return best_subsample, best_colsample_bytree


# tune alpha and lambda
def _tune_alpha_lambda(params, d_train):
    ptr.print_log('Tuning alpha and lambda ...')
    
    alpha_list = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    lambda_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    
    max_auc = 0.0
    best_alpha = alpha_list[0]
    best_lambda = lambda_list[0]
    
    for alpha, lambdaa in zip(alpha_list, lambda_list):
        # update params
        params['alpha'] = alpha
        params['lambda'] = lambdaa
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('alpha: {}; lambdaa: {}; auc: {}; rounds: {}'.\
              format(alpha, lambdaa, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_alpha = alpha
            best_lambda = lambdaa
                    
    ptr.print_log('best alpha: {0}'.format(best_alpha))
    ptr.print_log('best lambda: {0}'.format(best_lambda))
    ptr.print_log('max auc: {0}'.format(max_auc))
    
    return alpha, lambdaa


def _run_cv(params, d_train):
    cv_results = xgb.cv(
        params,
        d_train,
        num_boost_round = int(num_boost_round),
        nfold = int(nfold),
        metrics = metrics,
        early_stopping_rounds = int(early_stopping_rounds)
    )
    
    auc = cv_results['test-auc-mean'].max()
    rounds = cv_results['test-auc-mean'].argmax()
    
    return auc, rounds
        
'''
2017-10-24 18:04:30: Tuning xgboost parameters ...
2017-10-24 18:04:30: Tuning eta ...
2017-10-24 18:05:48: eta: 0.2; auc: 0.6366822; rounds: 36
2017-10-24 18:08:11: eta: 0.1; auc: 0.6385632; rounds: 80
2017-10-24 18:13:00: eta: 0.05; auc: 0.6394542; rounds: 177
2017-10-24 18:21:35: eta: 0.025; auc: 0.6394; rounds: 332
2017-10-24 19:02:28: eta: 0.005; auc: 0.6394871999999999; rounds: 1630
2017-10-24 20:13:23: eta: 0.0025; auc: 0.6393380000000001; rounds: 2906
2017-10-24 20:13:23: best eta: 0.005
2017-10-24 20:13:23: max auc: 0.6394871999999999
2017-10-24 20:13:23: Tuning max_depth and min_child_weight ...
2017-10-24 20:56:11: max_depth: 5; min_child_weight: 1; auc: 0.6399486000000001; rounds: 2029
2017-10-24 21:33:46: max_depth: 6; min_child_weight: 2; auc: 0.639358; rounds: 1522
2017-10-24 22:16:31: max_depth: 7; min_child_weight: 3; auc: 0.6388712; rounds: 1510
2017-10-24 23:00:03: max_depth: 8; min_child_weight: 4; auc: 0.6369316; rounds: 1330
2017-10-24 23:00:03: best max_depth: 5
2017-10-24 23:00:03: best min_child_weight: 1
2017-10-24 23:00:03: max auc: 0.6399486000000001
2017-10-24 23:00:03: Tuning subsample and colsample_bytree ...
2017-10-24 23:02:09: subsample: 0.1; colsample_bytree: 0.1; auc: 0.6223552; rounds: 195
2017-10-24 23:02:44: subsample: 0.2; colsample_bytree: 0.2; auc: 0.6209662; rounds: 33
2017-10-24 23:04:47: subsample: 0.4; colsample_bytree: 0.4; auc: 0.6268124; rounds: 114
2017-10-24 23:08:24: subsample: 0.6; colsample_bytree: 0.6; auc: 0.6263224000000001; rounds: 173
2017-10-24 23:50:17: subsample: 0.8; colsample_bytree: 0.8; auc: 0.6413632; rounds: 2037
2017-10-25 00:33:25: subsample: 1.0; colsample_bytree: 1.0; auc: 0.6399486000000001; rounds: 2029
2017-10-25 00:33:25: best subsample: 0.8
2017-10-25 00:33:25: best colsample_bytree: 0.8
2017-10-25 00:33:25: max auc: 0.6413632
2017-10-25 00:33:25: Tuning alpha and lambda ...
2017-10-25 01:17:12: alpha: 0.0; lambdaa: 2.0; auc: 0.6413689999999999; rounds: 2120
2017-10-25 02:00:39: alpha: 0.4; lambdaa: 4.0; auc: 0.641401; rounds: 2108
2017-10-25 02:43:59: alpha: 0.8; lambdaa: 6.0; auc: 0.641376; rounds: 2108
2017-10-25 03:27:16: alpha: 1.2; lambdaa: 8.0; auc: 0.6412524000000001; rounds: 2105
2017-10-25 04:10:51: alpha: 1.6; lambdaa: 10.0; auc: 0.6412454000000001; rounds: 2105
2017-10-25 04:49:21: alpha: 2.0; lambdaa: 12.0; auc: 0.6407689999999999; rounds: 1886
2017-10-25 05:37:40: alpha: 2.4; lambdaa: 14.0; auc: 0.6414662; rounds: 2370
2017-10-25 06:26:36: alpha: 2.8; lambdaa: 16.0; auc: 0.6414618000000001; rounds: 2407
2017-10-25 06:26:36: best alpha: 2.4
2017-10-25 06:26:36: best lambda: 14.0
2017-10-25 06:26:36: max auc: 0.6414662
'''

