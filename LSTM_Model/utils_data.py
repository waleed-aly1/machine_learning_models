import numpy as np
import scipy.io as sc
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


class DataManager:
    def __init__(self, params):
        self.params = params
        self.load_data()
        self.create_data()

    def load_data(self):


        df_temp = pd.read_csv(self.params['file_name'])
        df_features_target = df_temp.loc[:, 'CLFH':'CL']
        df_features_target.fillna(method='ffill', inplace=True)

        self.df_features_target = df_features_target

    def create_data(self):

        lag_lstm = self.params['lag_lstm']
        lag_forward_target = self.params['lag_forward_target']

        target = self.df_features_target.iloc[lag_forward_target:, -1]
        features = self.df_features_target.iloc[0:-lag_forward_target, :]


        self.target_lstm = target.iloc[lag_lstm-1:]
        #  (T, lag_lstm, num_feature)
        T = self.target_lstm.shape[0]
        num_feature = features.shape[1]
        self.features_lstm = np.zeros([T , lag_lstm, num_feature  ])

        for time in range(lag_lstm):
            self.features_lstm[:,time,:] = np.asarray(features.iloc[time:time+T, :])



        self.features_lstm = np.asarray(self.features_lstm)
        self.target_lstm = np.asarray(self.target_lstm)



    def normalize_data(self, data_array, test_array):
        scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
        yy = np.reshape(data_array, (-1, 1))
        yy = scalar.fit_transform(yy)
        yy = np.reshape(yy, data_array.shape)

        zz = np.reshape(test_array, (-1, 1))
        zz = scalar.transform(zz)
        zz = np.reshape(zz, test_array.shape)
        return yy, zz

    def normalize_data_robust(self, data_array, test_array):
        scalar = RobustScaler()
        yy = np.reshape(data_array, (-1, 1))
        yy = scalar.fit_transform(yy)
        yy = np.reshape(yy, data_array.shape)

        zz = np.reshape(test_array, (-1, 1))
        zz = scalar.transform(zz)
        zz = np.reshape(zz, test_array.shape)
        return yy, zz
