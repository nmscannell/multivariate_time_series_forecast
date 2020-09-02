from keras.models import Sequential
from keras.layers import Dense, LSTM

def Model(input):
    model = Sequential()
    model.add(LSTM(50, input_shape=(input.shape[1], input.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model