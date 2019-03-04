import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
import matplotlib
import matplotlib.pyplot as plt

def preProcessData():
    train = pd.read_csv("/home/kaiyuewang/Machine-Learning/fetal-hr.csv")
    train = train.drop(
        ['Unnamed: 9', 'Unnamed: 31', 'Unnamed: 42', 'Unnamed: 44', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS',
         'SUSP', 'CLASS'], axis=1)
    train = train.drop(train.index[2126:])
    train['Time'] = train['e'].sub(train['b'], axis=0)
    train = train.drop(['b', 'e'], axis=1)  # instead of using start and end feature, we can red

    list_to_normalize = list(set(train.columns.values) - set(['NSP']))
    y = train['NSP'].values
    X = train[list_to_normalize].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # scale the data for neural network input
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    std_scale = StandardScaler().fit(X_train)
    X_train = std_scale.transform(X_train)
    X_test = std_scale.transform(X_test)

    return X_train, X_test, y_train, y_test

def plot_graph(yDimension, xDimension, ylabel, xlabel):
    fig = plt.figure()
    plt.plot(xDimension, yDimension)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

