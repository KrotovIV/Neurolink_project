import tensorflow as tf
import time
import numpy as np
import pandas_datareader.data as pdr
import pandas as pd


spisok = []
for i in range(10):
    def get_stock_data(ticker, start_date, end_date):
        i = 1 #сколько раз попытался подключиться к данным
        try:
            all_data = pdr.get_data_yahoo(ticker, start_date, end_date)
        except ValueError:
            print("ValueError, trying again")
            i += 1
            if i < 5:
                time.sleep(10)
                get_stock_data(ticker, start_date, end_date)
            else:
                print("Tried 5 times, Yahoo error. Trying after 2 minutes")
                time.sleep(120)
                get_stock_data(ticker, start_date, end_date)
        all_data.to_csv("stock_prices.csv")

    start = "2003-01-01"
    end = "2022-03-30"
    get_stock_data('AAPL', start_date=start, end_date=end)
    filename = 'stock_prices.csv'
    df = pd.read_csv(filename, sep=',')


    #убираем сегодняшнюю дату если скачали сегодня
    from datetime import datetime
    today = ''.join(str(datetime.today()).split()[0].split('-'))
    last_day = str(df.at[df.shape[0]-1, 'Date'])
    if today == last_day: df = df[:-1]


    #подготовка traindata и testdata
    spl = 0.85
    i_spl = int(len(df) * spl)
    cols = ['Close', 'Volume']
    data_train = df.get(cols).values[:i_spl]
    data_test = df.get(cols).values[i_spl:]
    len_train = len(data_train)
    len_test = len(data_test)
    print(len(df), len_train, len_test)


    #нейросеть
    sequence_lnegth = 50
    input_dim = 2
    batch_size = 64
    epochs = 5

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(sequence_lnegth-1, input_dim), return_sequences=True),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100, return_sequences=False),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.summary() #print нейросети

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])


    def normalise_windows(window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def _next_window(i, seq_len, normalise):
        window = data_train[i:i+seq_len]
        window = normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def get_train_data(seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(len_train - seq_len + 1):
            x, y = _next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)


    x, y = get_train_data(
        seq_len=sequence_lnegth,
        normalise=True
    )

    print(x, y, x.shape, y.shape)

    import math
    steps_per_epoch = math.ceil((len_train - sequence_lnegth) / batch_size)
    print(steps_per_epoch)
    from keras.callbacks import EarlyStopping
    callbacks = [
        EarlyStopping(monitor='accuracy', patience=2)
    ]
    #обучение
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)


    def get_test_data(seq_len, normalise):
        data_windows = []
        for i in range(len_test - seq_len):
            data_windows.append(data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    x_test, y_test = get_test_data(
        seq_len=sequence_lnegth,
        normalise=True
    )

    print('test data shapes: ', x_test.shape, y_test.shape)

    model.evaluate(x_test, y_test, verbose=2)

    def get_last_data(seq_len, normalise):
        last_data = data_test[seq_len:]
        data_windows = np.array(last_data).astype(float)
        data_windows = normalise_windows(data_windows, single_window=True) if normalise else data_windows
        return data_windows



    last_data_2_predict_prices = get_last_data(-(sequence_lnegth-1), False)
    last_data_2_predict_prices_1st_place = last_data_2_predict_prices[0][0]
    last_data_2_predict = get_last_data(-(sequence_lnegth-1), True)
    print('***', -(sequence_lnegth-1), last_data_2_predict.size, '***')

    predictions2 = model.predict(last_data_2_predict)
    print(predictions2, predictions2[0][0])

    def de_normalise_predicted(price_1st, _data):
        return (_data + 1) * price_1st
    predicted_price = de_normalise_predicted(last_data_2_predict_prices_1st_place, predictions2[0][0])
    print(predicted_price)
    spisok.append(predicted_price)
print('-'*20)
print(sum(spisok)/len(spisok))