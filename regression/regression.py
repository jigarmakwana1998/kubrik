import requests
import pandas
import scipy
import numpy 
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area):# float
    from scipy import stats

    file = open('linreg_train.csv', 'r')
    for each in file:
        data_x = (each.split(','))
        if data_x[0] == 'area':
            X = data_x[1:]
        elif data_x[0] == 'price':
            Y = data_x[1:]

    pre_X = numpy.array([float(i) for i in X])
    pre_Y = numpy.array([float(i) for i in Y])
    slope, intercept, r_value, p_value, std_err = stats.linregress(pre_X, pre_Y)
    print(slope, intercept, r_value, p_value, std_err)
    return (slope * area) + intercept
    # YOUR IMPLEMENTATION HERE
    # ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
