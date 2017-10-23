'''
http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
'''

import numpy as np
import lightgbm as lgb
import printer as ptr


# default settings for cv()
num_boost_round = 5000
nfold = 5
metrics = {'auc'}
early_stopping_rounds = 10


# tune lightgbm parameters
def tune(data_x, data_y):
    ptr.print_log('\n')
    ptr.print_log('LIGHTGBM parameters are tuning ...')
    
    d_train = lgb.Dataset(data_x, label=data_y)
    
    # lightgbm params
    params = {
        'objective': 'binary',
    }
    
    # tune learning rate
    best_learning_rate = _tune_learning_rate(params, d_train)
    params['learning_rate'] = best_learning_rate
    
    # tune max_depth and num_leaves
    max_depth, num_leaves = _tune_max_depth__num_leaves(params, d_train)
    params['max_depth'] = max_depth
    params['num_leaves'] = num_leaves
    
    # tune min_data_in_leaf
    min_data_in_leaf = _tune_min_data_in_leaf(params, d_train)
    params['min_data_in_leaf'] = min_data_in_leaf

    # tune max_bin
    max_bin = _tune_max_bin(params, d_train)
    params['max_bin'] = max_bin
    
    # tune bagging_fraction and bagging_freq
    bagging_fraction, bagging_freq = _tune_bagging_fraction__bagging_freq(params, d_train)
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = bagging_freq
    
    # tune feature_fraction
    feature_fraction = _tune_feature_fraction(params, d_train)    
    params['feature_fraction'] = feature_fraction
    
    # end
    ptr.print_log('LIGHTGBM TUNER was finished.')
    ptr.print_log('\n')
    
    
# tune learning rate    
def _tune_learning_rate(params, d_train):
    ptr.print_log('Tuning learning_rate ...')
    
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
        
    ptr.print_log('best learning_rate: {0}'.format(best_learning_rate))
    ptr.print_log('max auc: {}'.format(max_auc))
        
    return best_learning_rate


# tune max_depth and num_leaves
def _tune_max_depth__num_leaves(params, d_train):
    ptr.print_log('Tuning max_depth and num_leaves ...')

    max_depth_list = list(range(4,9))
    num_leaves_list = list(range(30,121,10))
    
    max_auc = 0.0
    best_max_depth = max_depth_list[0]
    best_num_leaves = num_leaves_list[0]
    
    for max_depth, num_leaves in zip(max_depth_list, num_leaves_list):
        # update params
        params['max_depth'] = max_depth
        params['num_leaves'] = num_leaves
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('max_depth: {0}; num_leaves: {1}; auc: {2}; rounds: {3}'.\
                      format(max_depth, num_leaves, auc, rounds))

        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_depth = max_depth
            best_num_leaves = num_leaves

    ptr.print_log('best max_depth: {0}'.format(best_max_depth))
    ptr.print_log('best num_leaves: {0}'.format(best_num_leaves))
    ptr.print_log('max auc: {}'.format(max_auc))
    
    return best_max_depth, best_num_leaves
        

# tune min_data_in_leaf
def _tune_min_data_in_leaf(params, d_train):
    ptr.print_log('Tuning min_data_in_leaf...')

    min_data_in_leaf_list = list(range(100,1001,100))
    
    max_auc = 0.0
    best_min_data_in_leaf = min_data_in_leaf_list[0]
    
    for min_data_in_leaf in min_data_in_leaf_list:
        # update params
        params['min_data_in_leaf'] = min_data_in_leaf
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('min_data_in_leaf: {0}; auc: {1}; rounds: {2}'.\
                      format(min_data_in_leaf, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_min_data_in_leaf = min_data_in_leaf
        
    ptr.print_log('best min_data_in_leaf: {0}'.format(best_min_data_in_leaf))
    ptr.print_log('max auc: {}'.format(max_auc))
    
    return best_min_data_in_leaf


# tune max_bin
def _tune_max_bin(params, d_train):
    ptr.print_log('Tuning max_bin...')
    
    max_bin_list = list(range(50,301,50))
    
    max_auc = 0.0
    best_max_bin = max_bin_list[0]
    
    for max_bin in max_bin_list:
        # update params
        params['max_bin'] = max_bin
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('max_bin: {0}; auc: {1}; rounds: {2}'.\
                      format(max_bin, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_bin = max_bin
                    
    ptr.print_log('best max_bin: {0}'.format(best_max_bin))
    ptr.print_log('max auc: {}'.format(max_auc))
    
    return best_max_bin
    
  
# tune bagging_fraction and bagging_freq    
def _tune_bagging_fraction__bagging_freq(params, d_train):
    ptr.print_log('Tuning bagging_fraction and bagging_freq ...')
    
    bagging_fraction_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bagging_freq_list = list(range(0,51,10))
    
    max_auc = 0.0
    best_bagging_fraction = bagging_fraction_list[0]
    best_bagging_freq = bagging_freq_list[0]

    for bagging_fraction, bagging_freq in zip(bagging_fraction_list, best_bagging_freq):
        # update params
        params['bagging_fraction'] = bagging_fraction
        params['bagging_freq'] = bagging_freq
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('bagging_fraction: {0}; bagging_freq: {1}; auc: {2}; rounds: {3}'.\
                      format(bagging_fraction, bagging_freq, auc, rounds))

        # check auc
        if auc > max_auc:
            max_auc = auc
            best_bagging_fraction = bagging_fraction
            best_bagging_freq = bagging_freq
    
    ptr.print_log('best bagging_fraction: {0}'.format(best_bagging_fraction))
    ptr.print_log('best bagging_freq: {0}'.format(best_bagging_freq))
    ptr.print_log('max auc: {}'.format(max_auc))    
    
    return best_bagging_fraction, best_bagging_freq


# tune feature_fraction
def _tune_feature_fraction(params, d_train):
    ptr.print_log('Tuning feature_fraction ...')
    
    feature_fraction_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    max_auc = 0.0
    best_feature_fraction = feature_fraction_list[0]

    for feature_fraction in feature_fraction_list:
        # update params
        params['feature_fraction'] = feature_fraction
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('feature_fraction: {0}; auc: {1}; rounds: {2}'.\
                      format(feature_fraction, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_feature_fraction = feature_fraction
                    
    ptr.print_log('best feature_fraction: {0}'.format(best_feature_fraction))
    ptr.print_log('max auc: {}'.format(max_auc))
    
    return best_feature_fraction

   
# run cv with given parameters    
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

        