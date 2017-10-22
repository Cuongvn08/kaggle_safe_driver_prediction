import numpy as np
import lightgbm as lgb
import printer as ptr


num_boost_round = 5000
nfold = 5
metrics = {'auc'}
early_stopping_rounds = 10


def tune(data_x, data_y):
    ptr.print_log('Tuning lightgbm parameters ...')
    
    d_train = lgb.Dataset(data_x, label=data_y)
    
    # lightgbm params
    params = {
        'objective': 'binary',
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
    ptr.print_log('LIGHTGBM TUNER was finished.')
    
    
def _tune_learning_rate(params, d_train):
    ptr.print_log('Tuning learning rate ...')
    
    learning_rate_list = [0.2, 0.1, 0.05, 0.025, 0.005, 0.0025]
    max_auc = 0.0
    best_learning_rate = learning_rate_list[0]

    for learning_rate in learning_rate_list:
        # update params
        params['learning_rate'] = learning_rate
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('learning_rate: {0}; auc: {1}; rounds: {2}'.\
                      format(learning_rate, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_learning_rate = learning_rate
        
    print('best learning rate:', best_learning_rate)
    print('max auc:', max_auc)
    
    return best_learning_rate


def _tune_max_depth__num_samples_split(params, d_train):
    ptr.print_log('Tuning max depth and num samples split ...')

    max_depth_list = list(range(5,10))
    num_samples_split_list = list(range(200,1001,200))
    
    max_auc = 0.0
    best_max_depth = max_depth_list[0]
    best_num_samples_split = num_samples_split_list[0]
    
    for max_depth, num_samples_split in zip(max_depth_list, num_samples_split_list):
        # update params
        params['max_depth'] = max_depth
        params['num_samples_split'] = num_samples_split
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('max_depth: {0}; num_sample_split: {1}; auc: {2}; rounds: {3}'.\
                      format(max_depth, num_samples_split, auc, rounds))

        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_depth = max_depth
            best_num_samples_split = num_samples_split

    print('best max depth:', best_max_depth)
    print('best num samples split:', best_num_samples_split)
    print('max auc:', max_auc)
    
    return best_max_depth, best_num_samples_split
        

def _tune_min_samples_leaf(params, d_train):
    ptr.print_log('Tuning min samples leaf ...')

    min_samples_leaf_list = list(range(30,71,10))
    
    max_auc = 0.0
    best_min_samples_leaf = min_samples_leaf_list[0]
    
    for min_samples_leaf in min_samples_leaf_list:
        # update params
        params['min_samples_leaf'] = min_samples_leaf
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('min_samples_leaf: {0}; auc: {1}; rounds: {2}'.\
                      format(min_samples_leaf, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_min_samples_leaf = min_samples_leaf
        
    print('best min samples leaf:', best_min_samples_leaf)
    print('max auc:', max_auc)
    
    return best_min_samples_leaf


def _tune_max_features(params, d_train):
    ptr.print_log('Tuning max features ...')

    max_features_list = list(range(7,20,2))
    
    max_auc = 0.0
    best_max_features = max_features_list[0]
    
    for max_features in max_features_list:
        # update params
        params['max_features'] = max_features
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('max_features: {0}; auc: {1}; rounds: {2}'.\
                      format(max_features, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_features = max_features
        
    print('best max features:', best_max_features)
    print('max auc:', max_auc)
    
    return best_max_features


def _tune_subsample(params, d_train):
    ptr.print_log('Tuning subsample ...')

    subsample_list = [0.6,0.7,0.75,0.8,0.85,0.9]
    
    max_auc = 0.0
    best_subsample = subsample_list[0]
    
    for subsample in subsample_list:
        # update params
        params['subsample'] = subsample
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('subsample: {0}; auc: {1}; rounds: {2}'.\
                      format(subsample, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_subsample = subsample
        
    print('best subsample:', best_subsample)
    print('max auc:', max_auc)
    
    return best_subsample


def _run_cv(params, d_train):
    cv_results = lgb.cv(params,
                        d_train,
                        num_boost_round = int(num_boost_round),
                        nfold = int(nfold),
                        metrics = metrics,
                        early_stopping_rounds = int(early_stopping_rounds))
        
    auc = max(cv_results['auc-mean'])
    rounds = np.argmax(cv_results['auc-mean'])
    
    return auc, rounds

        