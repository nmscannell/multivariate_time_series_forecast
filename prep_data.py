"""
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
Reframe time series data as supervised learning dataset for time series forecasting.
Other ideas to explore:
--One-hot encoding wind direction
--Making all series stationary with differencing and seasonal adjustment
--Providing more than 1 hour of input time steps **
"""

from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# convert series to supervised learning problem--shift forecasts
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    :param data: Sequence of observations as a list or numpy array
    :param n_in: Number of lag observations as input (X)
    :param n_out: Number of observations as output (y)
    :param dropnan: Boolean whether or not to dro rows with NaN values
    :return: DataFrame of series framed for supervised learning
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # inputs
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecasts
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg