import matplotlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import pickle



# import data
data = pd.read_csv('MarketData_ClosesOnly.csv', index_col=0)

# fill non-traded bars with previous data
data.ffill(axis=0, inplace=True)

# separate the independent variables from the dependent variable in X and Y
X = data.iloc[:, 0:7]
y = data.iloc[:, 8:9]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, shuffle=False)


test_index = X_test.index.values.tolist()


scalarX = StandardScaler()
scalary = StandardScaler()

scalarX.fit(X_train)
scalary.fit(y_train)

X_train, X_test = scalarX.transform(X_train), scalarX.transform(X_test)
y_train, y_test = scalary.transform(y_train), scalary.transform(y_test)


def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=512, input_dim=7, activation='relu'))
    regressor.add(Dense(units=256, activation='relu'))
    regressor.add(Dense(units=128, activation='relu'))
    regressor.add(Dense(units=64, activation='relu'))
    regressor.add(Dense(units=1))
    regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'accuracy'])
    return regressor


regressor = KerasRegressor(build_fn=build_regressor)


ann_params = {'batch_size': [128],
              'nb_epoch': [500]
              }

gsc = GridSearchCV(regressor, param_grid=ann_params,
                   cv=TimeSeriesSplit(n_splits=10).split(X_train), verbose=10, n_jobs=-1, refit=True)


gsc.fit(X_train, y_train)

gsc_dataframe = pd.DataFrame(gsc.cv_results_)



y_pred = gsc.predict(X_test)
y_pred = y_pred.reshape(-1, 1)
y_pred = scalary.inverse_transform(y_pred)
y_test = scalary.inverse_transform(y_test)

mae = round(metrics.mean_absolute_error(y_test, y_pred), 2)
mse = round(metrics.mean_squared_error(y_test, y_pred), 2)
y_df = pd.DataFrame(index=pd.to_datetime(test_index))
y_pred = y_pred.reshape(len(y_pred), )
y_test = np.asarray(y_test).reshape(len(y_test), )
y_df['Model'] = y_pred
y_df['Actual'] = y_test
y_df.plot(title='{}'.format(gsc.cv_results_['params'][gsc.best_index_]))
at = matplotlib.offsetbox.AnchoredText('MAE:{}\nMSE:{}'.format(mae, mse), loc='lower left', frameon=True)
plt.gca().add_artist(at)
plt.tight_layout()
plt.show(block=False)
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 16)
gsc_dataframe.to_csv('ann_gridsearch.csv')
plt.show()

