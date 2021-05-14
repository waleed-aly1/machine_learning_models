import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
from sklearn.decomposition import PCA
from model import MLP


def running_view(arr, window, axis=0):
    """
    return a running view of length 'window' over 'axis'
    the returned array has an extra last dimension, which spans the window
    """
    shape = list(arr.shape)
    shape[axis] -= (window - 1)
    assert (shape[axis] > 0)
    return np.lib.index_tricks.as_strided(
        arr,
        shape + [window],
        arr.strides + (arr.strides[axis],))


def wavelet_denoise(arr):
    level = 0
    wav = pywt.Wavelet("coif5")
    coeffs = pywt.wavedec(arr, wav, level=level, mode="per")
    sigma = robust.mad(coeffs[-1], center=0)
    uthresh = sigma * np.sqrt(2 * np.log(len(arr)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeffs[1:])
    y = pywt.waverec(coeffs, wav, mode="per")
    return y


data = pd.read_csv('MarketData_ClosesOnly.csv')  # , index=0)#['CL']
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.fillna(method='ffill', inplace=True)

print(data.head())

multi = False
if multi:
    data.drop('Date', 1, inplace=True)
    x = data.values
else:
    x = data.CL
y = data.CL

# print(x_rec)

# Wavelets
x_rec = wavelet_denoise(x)
#x_rec = np.array(x_rec)
#print(x_rec.shape)
#plt.plot(x)
plt.plot(x_rec)
plt.show()

x = x_rec

# PCA
if multi:
    x = PCA(1).fit_transform(x)
    # x = np.concatenate([x, ind], 1)
else:
    # x = np.concatenate([x[:, np.newaxis], ind], 1)
    pass


# Bootstrap sampling (2 years, 2 months)
train_len = 25000
test_len = 2000
# splits_train = np.arange(0, len(x) - test_len, train_len)[1:]
# print(splits_train)


horizon = 1  # 48 1/2 hours per day
hist_horizon = 1
if multi:
    n_features = hist_horizon * x.shape[1]
else:
    n_features = hist_horizon

X_windows = running_view(x, window=hist_horizon).reshape(-1, n_features)
Y_windows = running_view(y[hist_horizon:], window=horizon).reshape(-1, horizon)

# split = int(.5 * len(data))
for split in range(train_len, len(x) - test_len, test_len):
    # training_data = data.iloc[:split]
    # test_data = data.iloc[split:split+test_len]
    print(split)

    errors = []

    split_tr = split  # + hist_horizon

    x_train = X_windows[split_tr - train_len:split_tr - horizon + 1]
    x_test = X_windows[split_tr:split_tr + test_len]
    y_train = Y_windows[split_tr - train_len:split_tr - horizon + 1]
    y_test = Y_windows[split_tr:split_tr + test_len]
    prices = y[split_tr + hist_horizon:split_tr + test_len].values

    model = MLP(n_features, horizon)
    model.fit(x_train, y_train, 1)
    # print(x_train[-1], y_train[-1])

    preds = np.zeros(len(prices))
    # preds[0] = np.mean(model.predict([x_test[0]]))
    for i in range(horizon, len(y_test) - horizon + 1, horizon):
        # print([x_test[i - horizon]], [y_test[i - horizon]])

        model.fit([x_test[i - horizon]], [y_test[i - horizon]], 5)
        # print(np.mean(model.predict([x_test[i]])), model.predict([x_test[i]]))
        # print(model.predict([x_test[i]]))

        preds[i:i + horizon] = model.predict([x_test[i]])

    preds = preds[horizon:]
    last_price = prices[horizon - 1:-1]
    prices = prices[horizon:]

    mape_base = np.mean(np.abs((prices - last_price) / prices)) * 100
    print("MAPE baseline:", mape_base)
    mape = np.mean(np.abs((prices - preds) / prices)) * 100
    print("MAPE:", mape)
    errors.append(mape)

    plt.plot(prices)
    plt.plot(preds)
    plt.show()
