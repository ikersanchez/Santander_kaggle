#Bayesian optimization script

import gc
import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import cross_val_score
from xgboost import XGBClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
features = train.columns[1:-1]


train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))




remove = []
cols = train.columns
for i in range(len(cols)-1):
        v = train[cols[i]].values
        for j in range(i+1, len(cols)):
            if np.array_equal(v, train[cols[j]].values):
                remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

remove = []
for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)




test_id = test.ID
test = test.drop(["ID"],axis=1)
labels = train.TARGET.values

train = train.drop(["TARGET","ID"],axis=1)
features = train.columns[1:-1]


def xgboostcv(max_depth,
              learning_rate,
              n_estimators,
              gamma,
              min_child_weight,
              max_delta_step,
              subsample,
              colsample_bytree,
              silent = False,
              nthread = -1,
              seed = 1234,
              verbose = True):
    return cross_val_score(XGBClassifier(max_depth = int(max_depth),
                                         learning_rate = learning_rate,
                                         n_estimators = int(n_estimators),
                                         silent = silent,
                                         nthread = nthread,
                                         gamma = gamma,
                                         min_child_weight = min_child_weight,
                                         max_delta_step = max_delta_step,
                                         subsample = subsample,
                                         colsample_bytree = colsample_bytree,
                                         seed = seed,
                                         objective = "binary:logistic"),
                           train,
                           labels,
                           "roc_auc",
                           cv=5).mean()



xgboostBO = BayesianOptimization(xgboostcv,
                                     {'max_depth': (5, 8),
                                      'learning_rate': (0.01,0.05),
                                      'n_estimators': (450, 1500),
                                      'gamma': (0,0),
                                      'min_child_weight': (0, 2),
                                      'max_delta_step': (0, 0),
                                      'subsample': (0.7, 1),
                                      'colsample_bytree' :(0.4, 0.8)
                                     })

xgboostBO.maximize()
print('-'*53)

print('Final Results')
print('XGBOOST: %f' % xgboostBO.res['max']['max_val'])
print('XGBOOST: %s' % xgboostBO.res['max']['max_params'])

#Best round
Step |   Time |      Value |   colsample_bytree |     gamma |   learning_rate |   max_delta_step |   max_depth |   min_child_weight |   n_estimators |   subsample | 
 20 | 16m43s |    0.84083 |             0.8000 |    0.0000 |          0.0100 |           0.0000 |      5.0000 |             2.0000 |       691.8857 |      0.7000 | 
 
