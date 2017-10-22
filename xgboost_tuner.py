import xgboost as xgb
import printer as ptr


num_boost_round = 5000
nfold = 5
metrics = {'auc'}
early_stopping_rounds = 10


def tune(data_x, data_y):
    ptr.print_log('Tuning xgboost parameters ...')

    d_train = xgb.DMatrix(data_x, label=data_y)

    params = {
        'objective': 'binary:logistic',
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
    ptr.print_log('XGBOOST TUNER was finished.')
    
    
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
        ptr.print_log('eta: {0}; auc: {1}; rounds: {2}'.\
                      format(eta, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_eta = eta
                    
    ptr.print_log('best eta: {0}'.format(best_eta))
    ptr.print_log('max auc: {0}'.format(max_auc))
    
    return best_eta


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
        ptr.print_log('max_depth: {0}; min_child_weight: {1}; auc: {2}; rounds: {3}'.\
                      format(max_depth, min_child_weight, auc, rounds))
        
        # check auc
        if auc < max_auc:
            max_auc = auc
            best_max_depth = max_depth
            best_min_child_weight = min_child_weight
                    
    ptr.print_log('best max_depth: {0}'.format(best_max_depth))
    ptr.print_log('best min_child_weight: {0}'.format(best_min_child_weight))
    ptr.print_log('max auc: {0}'.format(max_auc))
    
    return best_max_depth, best_min_child_weight
        

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
        ptr.print_log('subsample: {0}; colsample_bytree: {1}; auc: {2}; rounds: {3}'.\
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


def _tune_alpha_lambda(params, d_train):
    ptr.print_log('Tuning alpha and lambda ...')
    
    alpha_list = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    lambda_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    max_auc = float("Inf")
    best_alpha = alpha_list[0]
    best_lambda = lambda_list[0]
    
    for alpha, lambdaa in zip(alpha_list, lambda_list):
        # update params
        params['alpha'] = alpha
        params['lambda'] = lambdaa
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('alpha: {0}; lambdaa: {1}; auc: {2}; rounds: {3}'.\
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
        