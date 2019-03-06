
import matplotlib.pyplot as plt
import os
import seaborn as sns
from bld.project_paths import project_paths_join as ppj
import numpy as np
import pandas as pd
# Modelling algorithms.
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.model_code.module import RMSE

### Split both independent variable set and dependent variable set into training set and
### validation set with specific proportion.
X = pd.read_csv(ppj("OUT_DATA", "clean_X.csv"))
y = pd.read_csv(ppj("OUT_DATA", "clean_y.csv"))
X_test = pd.read_csv(ppj("OUT_DATA", "clean_X_test.csv"))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

# Modeling

## Linear Models:

### OLS
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
pred_ols = ols.predict(X_val)

#rmse_ols = np.sqrt(mean_squared_error(y_val, pred_ols))
rmse_ols = RMSE(y_val, pred_ols)


###  Ridge
model = linear_model.Ridge(random_state=1)
param_grid = {'alpha': [0.01, 0.2, 0.250, 0.3]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
pred_ridge = grid.predict(X_val)

rmse_ridge = np.sqrt(mean_squared_error(y_val, pred_ridge))


### Lasso
model = linear_model.Lasso(random_state=1)
param_grid = {'alpha': [0.001, 0.0001], 'max_iter': [600, 700]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)
pred_lasso = grid.predict(X_val)

rmse_lasso = np.sqrt(mean_squared_error(y_val, pred_lasso))

## Tree-Based Models:

### Decision Tree
dt = tree.DecisionTreeRegressor()
dt.fit(X_train, y_train)
pred_decision_tree = dt.predict(X_val)

rmse_decision_tree = np.sqrt(mean_squared_error(y_val, pred_decision_tree))


### RandomForest (Ensemble Techniques)
rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=32)
rf.fit(X_train,y_train)
pred_rf = rf.predict(X_val)

rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))


### XGBRegressor (Ensemble Techniques)

xgb = XGBRegressor(max_depth=15, n_jobs=32, n_estimators=100)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_val)

#t1 = time.time()
#time_xgb = t1-t0
#print(time_xgb)

rmse_xgb = np.sqrt(mean_squared_error(y_val,pred_xgb))


#### XGBoost Regressor achieved a considerably low RMSE!!!

## Non-cross-validated model:
### rmse_ols = 0.3731654202814775
### rmse_decision_tree = 0.1833295729393064
### rmse_rf = 0.23658979449357592
### rmse_xgb = 0.11862260744764404

## Cross-validated model:
### rmse_lasso = 0.3731772910822148
### rmse_ridge = 0.3731654091471227





# XGBRegression check cross validation score
## xgb = XGBRegressor(max_depth=15,n_jobs=32,n_estimators=100,subsample=0.7)
# t0 = time.time()

scores = cross_val_score(xgb, X, y, scoring='neg_mean_squared_error', cv=5)  # using full dataset

#t1 = time.time()
#time_xgb_cv = t1-t0
#print(time_xgb_cv)

# Above all XGBoost Regressor achieved lowest RMSE!!!
rmse_xgb_cv = np.sqrt(-scores)     # rmse_xgb_cv = 0.3475436
