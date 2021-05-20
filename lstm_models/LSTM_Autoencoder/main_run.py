from helper_run import create_predictor, create_encoder, prepare_data

import tensorflow as tf

with tf.device('/cpu:0'):
#with tf.device('/gpu:0'):

    params = {
        'file_name':'MarketData_ClosesOnly.csv',
        'lag_lstm': 12,
        'lag_forward_target': 1,
        'update_data': True,
        'train_predictor': True,
        'predictor_type': 'lstm',
        'epochs': 50,
        'use_encoder': False,
        'encoder_size': 6,
        'train_autoencoder': True,
        'encoder_type': 'lstm'
    }

    train_x, train_y, eval_x, eval_y, test_x, test_y = prepare_data(params)

    if params['use_encoder']:
        encoder = create_encoder(params, train_x, eval_x, test_x)
    else:
        encoder = None

    predictor = create_predictor(params, encoder, train_x, train_y, eval_x, eval_y, test_x, test_y)

    print('')


def place_on_locations(y, params, output_matrix):
    for i in range(len(y)):
        output_matrix[params[i, 0], params[i, 1]] = y[i]

    return output_matrix
