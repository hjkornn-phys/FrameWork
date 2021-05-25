import numpy as np
import pandas as pd


def target_function(x1, x2, x3):
    y = np.cos(x1) / (np.sin(x2) + 2) + x3
    return y


def create_dataset():
    x_data = np.random.uniform(low=-10, high=10, size=(1000, 3))
    noise = np.random.normal(0, 0.1, size=(1000))
    y_data = target_function(x_data[:, 0], x_data[:, 1], x_data[:, 2]) + noise
    return x_data, y_data


def train_test_split(x, y, split_ratio=0.2, shuffle=True):
    assert x.shape[0] == y.shape[0]
    split_num = int(split_ratio * x.shape[0])

    if shuffle:
        shuffler = np.random.permutation(x.shape[0])
        x = x[shuffler]
        y = y[shuffler]
    x_train = x[:-split_num]
    y_train = y[:-split_num]
    x_test = x[-split_num:]
    y_test = y[-split_num:]
    return (x_train, y_train), (x_test, y_test)


def test_model_result(model, x_data, y_data):
    y_pred = model.forward(x_data)
    for x, y, p in zip(x_data, y_data, y_pred):
        print(f"Input: {tuple(x)} / Answer: {y} / Pred: {p}")


def load_dataset(filename):
    # iris.csv를 불러와서 x_data, y_data 형식으로 만들어줍니다.
    data = pd.read_csv(filename)
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1:]
    y_data = y_data.replace({"Setosa": 0, "Versicolor": 1, "Virginica": 2})
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    return x_data, y_data