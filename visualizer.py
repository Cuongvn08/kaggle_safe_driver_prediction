import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gc

np.set_printoptions(threshold = np.nan)


def visualize():
    # config
    train_path = 'C:/data/kaggle/safe_driver_prediction/train.csv'

    # load data
    train_df = pd.read_csv(train_path)
    train_df.drop(['id'], axis = 1, inplace = True)
    print('\ntrain shape: ', train_df.shape)
    print('\nfeature types: ', Counter(train_df.dtypes.values))
    print('\nfeatures: ')
    for feature in train_df:
        print(' ', feature)

    # check NA ratio
    # replace -1 by NA because -1 in data indicate that the feature was missing    
    train_df_copied = train_df
    train_df_copied = train_df_copied.replace(-1, np.NaN) # Values of -1 indicate that the feature was missing from the observation"

    na_ratio = (train_df_copied.isnull().sum() / len(train_df_copied)).sort_values(ascending=False)
    print('\nNA ratio: ')
    print(na_ratio)

    del train_df_copied
    gc.collect()
    
    # show the target feature
    zero_count = (train_df['target']==0).sum()
    one_count = (train_df['target']==1).sum()
    plt.bar(np.arange(2), [zero_count, one_count])
    plt.show()
    
    print('target 0: ', zero_count)
    print('target 1: ', one_count)
    
    # correlation between continuous features
    f, ax = plt.subplots(figsize = (15, 15))
    plt.title('correlation of continuous features')
    sns.heatmap(train_df.corr(), ax = ax)
    plt.show()
    
    corr_values = train_df.corr().unstack().sort_values(ascending=False)
    print(type(corr_values))
    for pair, value in corr_values.iteritems():
        if float(value) != 1.0:
            print(pair, value)
    
    # binary features inpection
    bin_cols = [col for col in train_df.columns if '_bin' in col]
    zero_list = []
    one_list = []
    for col in bin_cols:
        zero_list.append((train_df[col]==0).sum())
        one_list.append((train_df[col]==1).sum())
    
    fig, ax = plt.subplots(figsize = (10, 10))
    p1 = ax.bar(range(len(bin_cols)), zero_list, width = 0.5, color=(1.0,0.0,0.0))
    p2 = ax.bar(range(len(bin_cols)), one_list, width = 0.5, color=(0.0,1.0,0.0))
    ax.set_xticklabels(bin_cols, rotation=40)
    red_patch = mpatches.Patch(color='red', label='zero count')
    green_patch = mpatches.Patch(color='green', label='one count')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    
    
    # release
    del train_df
    gc.collect()
    
    
if __name__ == "__main__":
    visualize()
    print('\n\n\nThe end.')
    
    
''' 
Notes from visualization:

train shape:  (595212, 58)

feature types:  Counter({dtype('int64'): 48, dtype('float64'): 10})

features: 
  target
  ps_ind_01
  ps_ind_02_cat
  ps_ind_03
  ps_ind_04_cat
  ps_ind_05_cat
  ps_ind_06_bin
  ps_ind_07_bin
  ps_ind_08_bin
  ps_ind_09_bin
  ps_ind_10_bin
  ps_ind_11_bin
  ps_ind_12_bin
  ps_ind_13_bin
  ps_ind_14
  ps_ind_15
  ps_ind_16_bin
  ps_ind_17_bin
  ps_ind_18_bin
  ps_reg_01
  ps_reg_02
  ps_reg_03
  ps_car_01_cat
  ps_car_02_cat
  ps_car_03_cat
  ps_car_04_cat
  ps_car_05_cat
  ps_car_06_cat
  ps_car_07_cat
  ps_car_08_cat
  ps_car_09_cat
  ps_car_10_cat
  ps_car_11_cat
  ps_car_11
  ps_car_12
  ps_car_13
  ps_car_14
  ps_car_15
  ps_calc_01
  ps_calc_02
  ps_calc_03
  ps_calc_04
  ps_calc_05
  ps_calc_06
  ps_calc_07
  ps_calc_08
  ps_calc_09
  ps_calc_10
  ps_calc_11
  ps_calc_12
  ps_calc_13
  ps_calc_14
  ps_calc_15_bin
  ps_calc_16_bin
  ps_calc_17_bin
  ps_calc_18_bin
  ps_calc_19_bin
  ps_calc_20_bin
  
NA ratio: 
    ps_car_03_cat     0.690898
    ps_car_05_cat     0.447825
    ps_reg_03         0.181065
    ps_car_14         0.071605
    ps_car_07_cat     0.019302
    ps_ind_05_cat     0.009760
    ps_car_09_cat     0.000956
    ps_ind_02_cat     0.000363
    ps_car_01_cat     0.000180
    ps_ind_04_cat     0.000139
    ps_car_02_cat     0.000008
    ps_car_11         0.000008
    ps_car_12         0.000002

Distribution of target:
    0 count:  573518
    1 count:  21694
    ==> imbalanced distribution of the target feature

Pairs of high correlation (>0.3):
    ('ps_ind_12_bin', 'ps_ind_14') 0.890127252659
    ('ps_ind_14', 'ps_ind_12_bin') 0.890127252659
    ('ps_car_13', 'ps_car_12') 0.671720256523
    ('ps_car_12', 'ps_car_13') 0.671720256523
    ('ps_reg_01', 'ps_reg_03') 0.637034533469
    ('ps_reg_03', 'ps_reg_01') 0.637034533469
    ('ps_car_13', 'ps_car_04_cat') 0.595173174013
    ('ps_car_04_cat', 'ps_car_13') 0.595173174013
    ('ps_car_04_cat', 'ps_car_12') 0.570027983489
    ('ps_car_12', 'ps_car_04_cat') 0.570027983489
    ('ps_ind_14', 'ps_ind_11_bin') 0.564902973936
    ('ps_ind_11_bin', 'ps_ind_14') 0.564902973936
    ('ps_car_15', 'ps_car_13') 0.529518555917
    ('ps_car_13', 'ps_car_15') 0.529518555917
    ('ps_reg_02', 'ps_reg_03') 0.516457187021
    ('ps_reg_03', 'ps_reg_02') 0.516457187021
    ('ps_car_03_cat', 'ps_car_05_cat') 0.489789427405
    ('ps_car_05_cat', 'ps_car_03_cat') 0.489789427405
    ('ps_reg_01', 'ps_reg_02') 0.4710270697
    ('ps_reg_02', 'ps_reg_01') 0.4710270697
    ('ps_ind_14', 'ps_ind_13_bin') 0.426399883628
    ('ps_ind_13_bin', 'ps_ind_14') 0.426399883628
    ('ps_ind_07_bin', 'ps_car_13') 0.347763731292
    ('ps_car_13', 'ps_ind_07_bin') 0.347763731292
    ('ps_ind_15', 'ps_ind_16_bin') 0.312449503103
    ('ps_ind_16_bin', 'ps_ind_15') 0.312449503103
    ('ps_ind_04_cat', 'ps_ind_07_bin') 0.304949378335
    ('ps_ind_07_bin', 'ps_ind_04_cat') 0.304949378335
        
Pairs of low correlation (<-0.3):
    ('ps_car_09_cat', 'ps_car_05_cat') -0.327750534826
    ('ps_car_05_cat', 'ps_car_09_cat') -0.327750534826
    ('ps_ind_06_bin', 'ps_ind_08_bin') -0.356838332745
    ('ps_ind_08_bin', 'ps_ind_06_bin') -0.356838332745
    ('ps_car_13', 'ps_car_08_cat') -0.370008769992
    ('ps_car_08_cat', 'ps_car_13') -0.370008769992
    ('ps_car_08_cat', 'ps_car_15') -0.37426557911
    ('ps_car_15', 'ps_car_08_cat') -0.37426557911
    ('ps_ind_06_bin', 'ps_ind_09_bin') -0.384345140139
    ('ps_ind_09_bin', 'ps_ind_06_bin') -0.384345140139
    ('ps_ind_15', 'ps_ind_18_bin') -0.451689409973
    ('ps_ind_18_bin', 'ps_ind_15') -0.451689409973
    ('ps_car_02_cat', 'ps_car_12') -0.469345054468
    ('ps_car_12', 'ps_car_02_cat') -0.469345054468
    ('ps_ind_06_bin', 'ps_ind_07_bin') -0.474009041853
    ('ps_ind_07_bin', 'ps_ind_06_bin') -0.474009041853
    ('ps_car_02_cat', 'ps_car_13') -0.482962718789
    ('ps_car_13', 'ps_car_02_cat') -0.482962718789
    ('ps_ind_16_bin', 'ps_ind_17_bin') -0.518076359741
    ('ps_ind_17_bin', 'ps_ind_16_bin') -0.518076359741
    ('ps_ind_18_bin', 'ps_ind_16_bin') -0.594265432744
    ('ps_ind_16_bin', 'ps_ind_18_bin') -0.594265432744    
    
'''






