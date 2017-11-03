# https://www.kaggle.com/bertcarremans/data-preparation-exploration?scriptVersionId=1676042

'''
This notebook aims at getting a good insight in the data for the PorteSeguro competition. 
Besides that, it gives some tips and tricks to prepare your data for modeling. 
The notebook consists of the following main sections:

1. [Visual inspection of your data]
2. [Defining the metadata]
3. [Descriptive statistics]
4. [Handling imbalanced classes]
5. [Data quality checks]
6. [Exploratory data visualization]
7. [Feature engineering]
8. [Feature selection]
9. [Feature scaling]
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', 100)

train = pd.read_csv('D:/data/safe_driver_prediction/train.csv')
test = pd.read_csv('D:/data/safe_driver_prediction/test.csv')


################################################################################
## DATA AT FIRST SIGHT

'''
Here is an except of the the data description for the competition:
* Features that belong to **similar groupings are tagged** as such in the feature names (e.g.,  ind, reg, car, calc).
* Feature names include the postfix **bin** to indicate binary features and **cat** to  indicate categorical features. 
* Features **without these designations are either continuous or ordinal**. 
* Values of **-1**  indicate that the feature was **missing** from the observation. 
* The **target** columns signifies whether or not a claim was filed for that policy holder.
'''

#print(train.head())
#print(train.tail())


'''
We indeed see the following:
* binary variables
* categorical variables of which the category values are integers
* other variables with integer or float values
* variables with -1 representing missing values
* the target variable and an ID variable
'''

print('train shape: ', train.shape) # (595212, 59)
print('test shape: ', test.shape) # (892816, 58)

print(train.info())


################################################################################
## METADATA
'''
To facilitate the data management, we'll store meta-information about the variables 
in a DataFrame. This will be helpful when we want to select specific variables 
for analysis, visualization, modeling, ...

Concretely we will store:
* role: input, ID, target
* level: nominal, interval, ordinal, binary
* keep: True or False
* dtype: int, float, str
'''

data = []
for f in train.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train[f].dtype == np.float64:
        level = 'interval'
    elif train[f].dtype == np.int64:
        level = 'ordinal'
            
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)

print(meta)
#print(meta[(meta.level == 'nominal') & (meta.keep)].index)
#print(pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index())


################################################################################
## Descriptive statistics
'''
We can also apply the describe method on the dataframe. 
However, it doesn't make much sense to calculate the mean, std, ... on categorical 
variables and the id variable. We'll explore the categorical variables visually later.
'''

# Interval variables
v = meta[(meta.level == 'interval') & (meta.keep)].index
print(train[v].describe())

# Ordinal variables
v = meta[(meta.level == 'ordinal') & (meta.keep)].index
print(train[v].describe())

# Binary variables
v = meta[(meta.level == 'binary') & (meta.keep)].index
print(train[v].describe())


################################################################################
# Handling imbalanced classes
'''
As we mentioned above the proportion of records with target=1 is far less than target=0 (0.036448)
This can lead to a model that has great accuracy but does have any added value in practice. 
Two possible strategies to deal with this problem are:
* oversampling records with target=1
* undersampling records with target=0
There are many more strategies of course and MachineLearningMastery.com gives a nice overview. 
As we have a rather large training set, we can go for undersampling.
'''

desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)
print('train shape: ', train.shape)
print('target 0 count: ', (train['target'] == 0).sum())
print('target 1 count: ', (train['target'] == 1).sum())


################################################################################
## Data Quality Checks

# Checking missing values: Missings are represented as -1
vars_with_missing = []

for f in train.columns:
    missings = train[train[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/train.shape[0]
        
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
        
print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
'''
ps_car_03_cat and ps_car_05_cat have a large proportion of records with missing values. Remove these variables.
For the other categorical variables with missing values, we can leave the missing value -1 as such.
ps_reg_03 (continuous) has missing values for 18% of all records. Replace by the mean.
ps_car_11 (ordinal) has only 5 records with misisng values. Replace by the mode.
ps_car_12 (continuous) has only 1 records with missing value. Replace by the mean.
ps_car_14 (continuous) has missing values for 7% of all records. Replace by the mean.
'''

# Dropping the variables with too many missing values
vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']
train.drop(vars_to_drop, inplace=True, axis=1)
meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta

# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()


# Checking the cardinality of the categorical variables
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

v = meta[(meta.level == 'nominal') & (meta.keep)].index
for f in v:
    dist_values = train[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))
'''
Only ps_car_11_cat has many distinct values, although it is still reasonable.
'''

train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 
                             test["ps_car_11_cat"], 
                             target=train.target, 
                             min_samples_leaf=100,
                             smoothing=10,
                             noise_level=0.01)
    
train['ps_car_11_cat_te'] = train_encoded
train.drop('ps_car_11_cat', axis=1, inplace=True)
meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta
test['ps_car_11_cat_te'] = test_encoded
test.drop('ps_car_11_cat', axis=1, inplace=True)


################################################################################
## Exploratory Data Visualization
    
# Categorical variables
v = meta[(meta.level == 'nominal') & (meta.keep)].index
for f in v:
    plt.figure()
    fig, ax = plt.subplots(figsize=(6,3))
    
    # Calculate the percentage of target=1 per category value
    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()
    cat_perc.sort_values(by='target', ascending=False, inplace=True)
    
    # Bar plot
    # Order the bars descending on target mean
    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])
    plt.ylabel('% target', fontsize=10)
    plt.xlabel(f, fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.show();
'''
As we can see from the variables with missing values, 
it is a good idea to keep the missing values as a separate category value, 
instead of replacing them by the mode for instance. 
The customers with a missing value appear to have a much higher (in some cases much lower) 
probability to ask for an insurance claim.
'''

# Checking the correlations between Interval variables
'''
Checking the correlations between interval variables. A heatmap is a good way 
to visualize the correlation between variables.
'''

def corr_heatmap(v):
    correlations = train[v].corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show();
    
v = meta[(meta.level == 'interval') & (meta.keep)].index
corr_heatmap(v)
'''
There are a strong correlations between the variables:
* ps_reg_02 and ps_reg_03 (0.7)
* ps_car_12 and ps_car13 (0.67)
* ps_car_12 and ps_car14 (0.58)
* ps_car_13 and ps_car15 (0.67)
'''

s = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()

sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()

sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()

sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()

# Checking the correlations between ordinal variables
v = meta[(meta.level == 'ordinal') & (meta.keep)].index
corr_heatmap(v)












