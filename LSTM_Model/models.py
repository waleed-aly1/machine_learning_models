from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten, RepeatVector, Masking, TimeDistributed, Input
import keras.losses as losses
from keras.models import Model
from keras.regularizers import l1_l2




def lstm_auto_encoder(lag, features_no, encoder_size):
    input_layer_0 = Input(shape=(lag, features_no))
    hidden_layer_0 = LSTM(3, return_sequences=True, kernel_regularizer=l1_l2(0.0001, 0.0001))(input_layer_0)
    hidden_layer_1 = LSTM(encoder_size, return_sequences=True)(hidden_layer_0)
    hidden_layer_2 = LSTM(3, return_sequences=True)(hidden_layer_1)
    hidden_layer_3 = (TimeDistributed(Dense(features_no)))(hidden_layer_2)

    new_model = Model(inputs=input_layer_0, outputs=hidden_layer_3)
    new_model.compile(loss=losses.mean_absolute_error, optimizer='adam')

    new_encoder = Model(inputs=input_layer_0, outputs=hidden_layer_1)

    return new_model, new_encoder



def lstm_predictor(lag, feature_size):
    input_layer_0 = Input(shape=(lag, feature_size))
    hidden_layer_0 = LSTM(5, return_sequences=True, kernel_regularizer=l1_l2(0.00002, 0.00002))(input_layer_0)
    hidden_layer_0 = Dropout(0.2)(hidden_layer_0)
    hidden_layer_1 = LSTM(2, return_sequences=True, kernel_regularizer=l1_l2(0.00002, 0.00002))(hidden_layer_0)
    hidden_layer_1 = Dropout(0.2)(hidden_layer_1)
    hidden_layer_2 = LSTM(3, return_sequences=False, kernel_regularizer=l1_l2(0.00005, 0.00005))(hidden_layer_1)
    hidden_layer_3 = Dense(1)(hidden_layer_2)

    new_model = Model(inputs=input_layer_0, outputs=hidden_layer_3)
    new_model.compile(loss=losses.mean_squared_error, optimizer='adam')

    return new_model
