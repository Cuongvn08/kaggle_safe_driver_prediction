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
        ptr.print_log('learning_rate: {}; auc: {}; rounds: {}'.\
                      format(learning_rate, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_learning_rate = learning_rate
        
    ptr.print_log('best learning_rate: {}'.format(best_learning_rate))
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
        ptr.print_log('max_depth: {}; num_leaves: {}; auc: {}; rounds: {}'.\
                      format(max_depth, num_leaves, auc, rounds))

        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_depth = max_depth
            best_num_leaves = num_leaves

    ptr.print_log('best max_depth: {}'.format(best_max_depth))
    ptr.print_log('best num_leaves: {}'.format(best_num_leaves))
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
        ptr.print_log('min_data_in_leaf: {}; auc: {}; rounds: {}'.\
                      format(min_data_in_leaf, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_min_data_in_leaf = min_data_in_leaf
        
    ptr.print_log('best min_data_in_leaf: {}'.format(best_min_data_in_leaf))
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
        ptr.print_log('max_bin: {}; auc: {}; rounds: {}'.\
                      format(max_bin, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_max_bin = max_bin
                    
    ptr.print_log('best max_bin: {}'.format(best_max_bin))
    ptr.print_log('max auc: {}'.format(max_auc))
    
    return best_max_bin
    
  
# tune bagging_fraction and bagging_freq    
def _tune_bagging_fraction__bagging_freq(params, d_train):
    ptr.print_log('Tuning bagging_fraction and bagging_freq ...')
    
    bagging_fraction_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    bagging_freq_list = list(range(0,51,10))
    
    max_auc = 0.0
    best_bagging_fraction = bagging_fraction_list[0]
    best_bagging_freq = bagging_freq_list[0]

    for bagging_fraction, bagging_freq in zip(bagging_fraction_list, bagging_freq_list):
        # update params
        params['bagging_fraction'] = bagging_fraction
        params['bagging_freq'] = bagging_freq
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('bagging_fraction: {}; bagging_freq: {}; auc: {}; rounds: {}'.\
                      format(bagging_fraction, bagging_freq, auc, rounds))

        # check auc
        if auc > max_auc:
            max_auc = auc
            best_bagging_fraction = bagging_fraction
            best_bagging_freq = bagging_freq
    
    ptr.print_log('best bagging_fraction: {}'.format(best_bagging_fraction))
    ptr.print_log('best bagging_freq: {}'.format(best_bagging_freq))
    ptr.print_log('max auc: {}'.format(max_auc))    
    
    return best_bagging_fraction, best_bagging_freq


# tune feature_fraction
def _tune_feature_fraction(params, d_train):
    ptr.print_log('Tuning feature_fraction ...')
    
    feature_fraction_list = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    
    max_auc = 0.0
    best_feature_fraction = feature_fraction_list[0]

    for feature_fraction in feature_fraction_list:
        # update params
        params['feature_fraction'] = feature_fraction
        
        # run cv
        auc, rounds = _run_cv(params, d_train)
        ptr.print_log('feature_fraction: {}; auc: {}; rounds: {}'.\
                      format(feature_fraction, auc, rounds))
        
        # check auc
        if auc > max_auc:
            max_auc = auc
            best_feature_fraction = feature_fraction
                    
    ptr.print_log('best feature_fraction: {}'.format(best_feature_fraction))
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


'''
2017-10-23 13:47:17: LIGHTGBM parameters are tuning ...
2017-10-23 13:47:17: Tuning learning_rate ...
2017-10-23 13:47:29: learning_rate: 0.2; auc: 0.6366819364859776; rounds: 41
2017-10-23 13:48:09: learning_rate: 0.1; auc: 0.6384235009174468; rounds: 74
2017-10-23 13:49:12: learning_rate: 0.05; auc: 0.6385787795241494; rounds: 156
2017-10-23 13:50:55: learning_rate: 0.025; auc: 0.6390406168332312; rounds: 336
2017-10-23 14:02:12: learning_rate: 0.005; auc: 0.6393072737979867; rounds: 1724
2017-10-23 14:20:35: learning_rate: 0.0025; auc: 0.6393976431274061; rounds: 3531
2017-10-23 14:20:35: best learning_rate: 0.0025
2017-10-23 14:20:35: max auc: 0.6393976431274061
2017-10-23 14:20:35: Tuning max_depth and num_leaves ...
2017-10-23 14:21:26: max_depth: 4; num_leaves: 30; auc: 0.6113147808895177; rounds: 164
2017-10-23 14:22:39: max_depth: 5; num_leaves: 40; auc: 0.6175308707774295; rounds: 258
2017-10-23 14:39:24: max_depth: 6; num_leaves: 50; auc: 0.6391120213984274; rounds: 3064
2017-10-23 14:40:58: max_depth: 7; num_leaves: 60; auc: 0.6226079437501216; rounds: 247
2017-10-23 14:43:09: max_depth: 8; num_leaves: 70; auc: 0.6231282279219362; rounds: 255
2017-10-23 14:43:09: best max_depth: 6
2017-10-23 14:43:09: best num_leaves: 50
2017-10-23 14:43:09: max auc: 0.6391120213984274
2017-10-23 14:43:09: Tuning min_data_in_leaf...
2017-10-23 15:00:45: min_data_in_leaf: 100; auc: 0.6394701011093902; rounds: 3189
2017-10-23 15:20:32: min_data_in_leaf: 200; auc: 0.6397383698465186; rounds: 3671
2017-10-23 15:40:20: min_data_in_leaf: 300; auc: 0.6395242149781659; rounds: 3150
2017-10-23 15:42:03: min_data_in_leaf: 400; auc: 0.6213934418020457; rounds: 265
2017-10-23 16:03:28: min_data_in_leaf: 500; auc: 0.6395343643872469; rounds: 3069
2017-10-23 16:25:50: min_data_in_leaf: 600; auc: 0.6393861984715724; rounds: 3151
2017-10-23 16:27:44: min_data_in_leaf: 700; auc: 0.6218681595000501; rounds: 328
2017-10-23 16:48:00: min_data_in_leaf: 800; auc: 0.639275923883251; rounds: 3157
2017-10-23 17:06:21: min_data_in_leaf: 900; auc: 0.6391862835628281; rounds: 3113
2017-10-23 17:09:03: min_data_in_leaf: 1000; auc: 0.6221987618837161; rounds: 421
2017-10-23 17:09:03: best min_data_in_leaf: 200
2017-10-23 17:09:03: max auc: 0.6397383698465186
2017-10-23 17:09:03: Tuning max_bin...
2017-10-23 17:34:07: max_bin: 50; auc: 0.6397383698465186; rounds: 3671
2017-10-23 17:54:04: max_bin: 100; auc: 0.6397383698465186; rounds: 3671
2017-10-23 18:13:48: max_bin: 150; auc: 0.6397383698465186; rounds: 3671
2017-10-23 18:33:39: max_bin: 200; auc: 0.6397383698465186; rounds: 3671
2017-10-23 18:53:55: max_bin: 250; auc: 0.6397383698465186; rounds: 3671
2017-10-23 19:15:10: max_bin: 300; auc: 0.6397383698465186; rounds: 3671
2017-10-23 19:15:10: best max_bin: 50
2017-10-23 19:15:10: max auc: 0.6397383698465186
2017-10-24 09:03:40 Tuning bagging_fraction and bagging_freq ...
2017-10-24 09:27:13 bagging_fraction: 0.1; bagging_freq: 0; auc: 0.6397383698465186; rounds: 3671
2017-10-24 09:28:00 bagging_fraction: 0.2; bagging_freq: 10; auc: 0.6266593219938851; rounds: 222
2017-10-24 09:29:15 bagging_fraction: 0.4; bagging_freq: 20; auc: 0.6254059028392025; rounds: 290
2017-10-24 09:29:50 bagging_fraction: 0.6; bagging_freq: 30; auc: 0.6211882682958355; rounds: 135
2017-10-24 09:31:42 bagging_fraction: 0.8; bagging_freq: 40; auc: 0.6230909700241403; rounds: 297
2017-10-24 09:32:32 bagging_fraction: 0.9; bagging_freq: 50; auc: 0.62012790306674; rounds: 135
2017-10-24 09:32:32 best bagging_fraction: 0.1
2017-10-24 09:32:32 best bagging_freq: 0
2017-10-24 09:32:32 max auc: 0.6397383698465186
2017-10-24 09:39:17 Tuning feature_fraction ...
2017-10-24 09:39:24 feature_fraction: 0.1; auc: 0.619373763834435; rounds: 14
2017-10-24 09:39:36 feature_fraction: 0.2; auc: 0.6279830909765094; rounds: 39
2017-10-24 09:39:50 feature_fraction: 0.4; auc: 0.6275969300877133; rounds: 46
2017-10-24 09:40:03 feature_fraction: 0.6; auc: 0.626701054710648; rounds: 34
2017-10-24 09:40:12 feature_fraction: 0.8; auc: 0.6220311678737613; rounds: 14
2017-10-24 09:40:21 feature_fraction: 0.9; auc: 0.62182441870044; rounds: 14
2017-10-24 09:40:21 best feature_fraction: 0.2
2017-10-24 09:40:21 max auc: 0.6279830909765094
'''      