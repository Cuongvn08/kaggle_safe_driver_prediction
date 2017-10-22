import lightgbm as lgb
import printer as ptr


num_boost_round = 5000
nfold = 5
metrics = {'mae'}
early_stopping_rounds = 10


def tune(data_x, data_y):
    ptr.print_log('Tuning lightgbm parameters ...')
    
    d_train = lgb.Dataset(data_x, label=data_y)
    
    # lightgbm params
    params = {
        'objective': 'binary',
        'metric' : 'mae',
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
    min_mae = float("Inf")
    best_learning_rate = learning_rate_list[0]

    for learning_rate in learning_rate_list:
        # update params
        params['learning_rate'] = learning_rate
        
        # run cv
        cv_results = lgb.cv(params,
                            d_train,
                            num_boost_round = int(num_boost_round),
                            nfold = int(nfold),
                            metrics = metrics,
                            early_stopping_rounds = int(early_stopping_rounds))
        
        # print
        mae = cv_results['metric1-mean'].min()
        rounds = cv_results['metric1-mean'].argmin()
        ptr.print_log('learning_rate: {0}; mae: {1}; rounds: {2}'.\
                      format(learning_rate, mae, rounds))
        
        # check min mae
        if mae < min_mae:
            min_mae = mae
            best_learning_rate = learning_rate
        
    print('best learning rate:', best_learning_rate)
    print('min mae:', min_mae)
    return best_learning_rate


def _tune_max_depth__num_samples_split(params, d_train):
    ptr.print_log('Tuning max depth and num samples split ...')

def _tune_min_samples_leaf(params, d_train):
    ptr.print_log('Tuning min samples leaf ...')

def _tune_max_features(params, d_train):
    ptr.print_log('Tuning max features ...')

def _tune_subsample(params, d_train):
    ptr.print_log('Tuning subsample ...')


    