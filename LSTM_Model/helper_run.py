from models import lstm_auto_encoder, lstm_predictor
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import numpy as np
from utils_data import DataManager
import plotly
import plotly.graph_objs as go
import pickle
import os
from datetime import datetime
import pandas as pd


def create_predictor(params, encoder, train_x, train_y, eval_x, eval_y, test_x, test_y):

    if params['use_encoder']:
        name = 'predictor_' + params['encoder_type'] + '_' + params['predictor_type']
    else:
        name = 'predictor_' + params['predictor_type']

    if params['train_predictor']:
        if params['use_encoder']:
            train_x = encoder.predict(train_x)
            test_x = encoder.predict(test_x)
            eval_x = encoder.predict(eval_x)

        model = lstm_predictor(params['lag_lstm'], train_x.shape[2])
        run_history = model.fit(train_x, train_y, validation_data=(eval_x, eval_y), verbose=2, epochs=params['epochs'])
        predicted = model.predict(test_x)

        predicted_train = model.predict(train_x)
        plotter_plot(train_y, predicted_train, 'train')

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        report_results(test_y, predicted, name, params, 'predictor', run_history.history['loss'][-1],
                       run_history.history['val_loss'][-1], short_model_summary)

        model.save('models\\' + name + '.h5')

    else:
        model = load_model('models\\' + name + '.h5')
        predicted = model.predict(test_x)
        predicted_train = model.predict(train_x)
        plotter_plot(train_y, predicted_train, 'train')
        plotter_plot(test_y, predicted, name)


    return model


def create_encoder(params, train_x, eval_x, test_x):
    name = 'encoder_' + params['encoder_type']
    if params['train_autoencoder']:
        # auto encoder
        if params['encoder_type'] == 'lstm':
            model, encoder = lstm_auto_encoder(params['lag_lstm'], train_x.shape[2], params['encoder_size'])
        else:
            raise Exception('Unknown Encoder Type!')

        print(model.summary())
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        run_history = model.fit(x=train_x, y=train_x, validation_data=(eval_x, eval_x), verbose=2, epochs=5,
                                callbacks=[early_stopping, reduce_lr])

        predicted = model.predict(test_x)

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)

        report_results(test_x, predicted, name, params, 'encoder', run_history.history['loss'][-1],
                       run_history.history['val_loss'][-1], short_model_summary)

        encoder.save('models\\' + name + '.h5')
    else:
        encoder = load_model('models\\' + name + '.h5')
    return encoder


def report_results(target, estimate, name, params, mode, loss, val_loss, summary):
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    mae = np.mean(np.abs((target.ravel() - estimate.ravel())))
    if mode == 'predictor':
        file_name = 'results\\results_predictor.pkl'
    elif mode == 'encoder':
        file_name = 'results\\results_encoder.pkl'

    if os.path.isfile(file_name):
        results = pd.read_pickle(file_name)
    else:
        results = pd.DataFrame(columns=['exec_time', 'name', 'loss', 'val_loss', 'mae_test', 'params', 'summary'])
    results.loc[len(results)] = [now, name, loss, val_loss, mae, params, summary]
    results.to_pickle(file_name)
    print("total mean absolute error: " + str(mae))
    plotter_plot(target, estimate, name)
    return 0


def plotter_plot(target, estimate, name):
    #ind_plot = np.random.randint(target.ravel().shape[0], size=10000)
    test_plot = target.ravel()[:]
    predict_plot = estimate.ravel()[:]


    trace0 = go.Scatter(
        x=np.arange(test_plot.shape[0]),
        y=predict_plot,
        mode='lines+markers',
        name='predicted'
    )
    trace1 = go.Scatter(
        x=np.arange(test_plot.shape[0]),
        y=test_plot,
        mode='lines+markers',
        name='original'
    )
    data0 = [trace0, trace1]
    plotly.offline.plot(data0, filename='results\\' + name + '.html')


def prepare_data(params):
    if params['update_data']:
        data_man = DataManager(params)
        data_x = data_man.features_lstm
        data_y = data_man.target_lstm

        # Non-causal normalizing
        #for ind_normalize in range(data_x.shape[2]):
         #  data_x[:, :, ind_normalize], yy = data_man.normalize_data( data_x[:, :, ind_normalize], data_x[:, :, ind_normalize])

        #data_y , yy= data_man.normalize_data(data_y, data_y)

        ind_train = int( np.round(data_y.shape[0] *0.8))


        train_x = data_x[:ind_train, :, :]
        train_y = data_y[:ind_train]
        test_x = data_x[ind_train:, :, :]
        test_y = data_y[ind_train:]

        # Causal normalization of target or features can be considered here
        for ind_normalize in range(train_x.shape[2]):
           train_x[:, :, ind_normalize], test_x[:, :, ind_normalize] = data_man.normalize_data( train_x[:, :, ind_normalize],
           test_x[:, :, ind_normalize])

        train_y, test_y = data_man.normalize_data(train_y, test_y)

        ind_eval =  int( np.round(data_y.shape[0] *0.7))
        eval_x = train_x[ind_eval:, :, :]
        eval_y = train_y[ind_eval:]
        train_x = train_x[:ind_eval, :, :]
        train_y = train_y[:ind_eval]



        data = {'train_x': train_x,
                'train_y': train_y,
                'test_x': test_x,
                'test_y': test_y,
                'eval_x': eval_x,
                'eval_y': eval_y,
                }

        with open('data\\data.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('data\\data.pickle', 'rb') as handle:
            data = pickle.load(handle)

    return data['train_x'], data['train_y'], data['eval_x'], data['eval_y'], \
           data['test_x'], data['test_y']
