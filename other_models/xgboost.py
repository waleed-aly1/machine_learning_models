import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid, GridSearchCV
import xgboost as xgb
import pickle


# import data
data = pd.read_csv('../data/MarketData_ClosesOnly.csv', index_col=0)

# fill non-traded bars with previous data
data.ffill(axis=0, inplace=True)

# separate the independent variables from the dependent variable in X and Y
X = data.iloc[:, 0:8]
y = data.iloc[:, 8:9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=False)

xgb_model = xgb.XGBRegressor(n_estimators=7000, learning_rate=.08, max_depth=3, min_child_weight=3, gamma=0,
                                colsample_bytree=.9, subsample=.6, reg_alpha=.1, objective='reg:squarederror')
xgb_model.fit(X,y)


filename = 'XGB_Model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

print('done')



