import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, ParameterGrid, GridSearchCV
from sklearn.svm import SVR
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from tqdm import tqdm
from math import sqrt
from datetime import datetime

# import data
data = pd.read_csv('MarketData_ClosesOnly.csv', index_col=0)

# fill non-traded bars with previous data
data.ffill(axis=0, inplace=True)

# separate the independent variables from the dependent variable in X and Y
X = data.iloc[:, 0:8]
y = data.iloc[:, 8:9]


class ModelRun():
    def __init__(self, X, y, regressor, parameters, no_splits=5, scalar=StandardScaler):
        self.X = X
        self.y = y
        self.regressor = regressor
        self.parameters = parameters
        self.no_splits = no_splits
        self.scalar = scalar

    def find_optimal_paramters(self, X, y, regressor, parameters, scoring_metric='MAE', greater_is_better=False):
        score_methods = {'MAE': metrics.mean_absolute_error,
                         'MSE': metrics.mean_squared_error,
                         'MSLE': metrics.mean_squared_log_error,
                         'r_qsuqred': metrics.r2_score}

        scoring_metric = score_methods[scoring_metric]

        # check if parameter list is empty and run return default params if so
        if not parameters:
            best_params = regressor.get_params()
            best_score = 0


        for e, p in enumerate(ParameterGrid(parameters)):
            regressor.set_params(**p)
            regressor.fit(X, y.ravel())
            score = scoring_metric(regressor.predict(X), y)

            if e == 0:
                best_score = score
                best_params = p
            elif greater_is_better:
                if score > best_score:
                    best_score = score
                    best_params = p
            elif score < best_score:
                best_score = score
                best_params = p

        return best_score, best_params

    def foward_chain_cv(self, scoring_metric, greater_is_better=False):
        i = 1

        MAE = []
        Exp_var = []
        MSE = []
        r_squared = []
        params_used = {}


        y_pred_cont = []
        y_test_cont = []
        y_pred_cont_index = []
        split_dates = []

        fig = plt.figure()


        tscv = TimeSeriesSplit(n_splits=self.no_splits)
        for train_index, test_index in tqdm(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_test_index = X_test.index.values.tolist()


            if self.scalar is not None:
                # Scale Data
                scaler_X = self.scalar()
                scaler_y = self.scalar()
                scaler_X.fit(X_train)
                scaler_y.fit(y_train)
                X_train, X_test = scaler_X.transform(X_train), scaler_X.transform(X_test)
                y_train, y_test = scaler_y.transform(y_train), scaler_y.transform(y_test)
            else:
                X_train, X_test = np.asarray(X_train), np.asarray(X_test)
                y_train, y_test = np.asarray(y_train), np.asarray(y_test)

            # Find Best Params
            best_score, best_params = self.find_optimal_paramters(
                X_train, y_train, self.regressor, self.parameters, scoring_metric, greater_is_better)

            self.regressor.set_params(**best_params)
            self.regressor.fit(X_train, y_train.ravel())


            # predict y values
            y_pred = self.regressor.predict(X_test)

            if self.scalar is not None:
                # transform y values back to real scale for assessment
                y_pred = scaler_y.inverse_transform(y_pred)
                y_test = scaler_y.inverse_transform(y_test)

            # compute error metrics
            params_used[i] = best_params
            MAE.append(metrics.mean_absolute_error(y_test, y_pred))
            Exp_var.append(metrics.explained_variance_score(y_test, y_pred))
            MSE.append(metrics.mean_squared_error(y_test, y_pred))
            r_squared.append(metrics.r2_score(y_test, y_pred))

            # plot y_pred vs y_test
            y_df = pd.DataFrame(index=pd.to_datetime(X_test_index))
            y_pred = y_pred.reshape(len(y_pred), )
            y_test = y_test.reshape(len(y_test), )
            y_df['y_pred'] = y_pred
            y_df['y_test'] = y_test

            # plot the subplots
            ax = fig.add_subplot(int(sqrt(self.no_splits)), int(sqrt(self.no_splits)+1), i)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%y'))
            y_df.plot(title = 'Split{}'.format(i), ax=ax, legend=False)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if i == 1:
                fig.legend(loc=4)

            # convert arrays to list and append continuous y_pred vs y_test
            y_pred_cont_index = y_pred_cont_index + X_test_index
            split_dates.append(y_pred_cont_index[-1])
            y_pred_list = y_pred.tolist()
            y_test_list = y_test.tolist()
            y_pred_cont = y_pred_cont + y_pred_list
            y_test_cont = y_test_cont + y_test_list

            i += 1

        # Plot the continuous chart
        y_continuous_df = pd.DataFrame(index=pd.to_datetime(y_pred_cont_index))
        y_pred_cont = np.asarray(y_pred_cont)
        y_test_cont = np.asarray(y_test_cont)
        y_continuous_df['Model'] = y_pred_cont
        y_continuous_df['Actual'] = y_test_cont
        y_continuous_df.plot(title='Running Performance')

        # add verticle lines to the running total output
        del split_dates[-1]
        for date in split_dates:
            date = datetime.strptime(date, '%m/%d/%Y %H:%M')
            plt.axvline(x=date, linestyle=':', color='red', linewidth=1, alpha=.8)

        # Calculate average metrics
        no_splits = tscv.get_n_splits()
        avg_mae = sum(MAE) / no_splits
        avg_exp_var = sum(Exp_var) / no_splits
        avg_mse = sum(MSE) / no_splits
        avg_rsquared = sum(r_squared) / no_splits

        print('\nMAE:{} \nMSE:{} \nExp Var Explained: {}\nr^2: {}\nParams:{}'.format(MAE, MSE, Exp_var, r_squared,
                                                                                     params_used))
        print('\nAvg MAE:', avg_mae,
              '\nAverage Explained Variance:', avg_exp_var,
              '\nAvg MSE:', avg_mse,
              '\nAvg r^2:', avg_rsquared)
        print('end')
        fig.tight_layout()
        plt.show()


SVR_parameters = [{'kernel': ['rbf'],
               'gamma': [.01],
               'C': [1]}]

RF_params = [{'n_estimators': [10,20,50],
              'max_features':[2,6,8]
              }]
ModelRun(X, y, SVR(), SVR_parameters, no_splits=10, scalar=StandardScaler).foward_chain_cv(scoring_metric='MAE')
