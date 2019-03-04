import mlrose
import numpy
import util
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = util.preProcessData()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [16, 8], activation ='relu',
                                     algorithm ='simulated_annealing',  clip_max=5,
                                     max_iters = 10000, bias = True, is_classifier = True,
                                     early_stopping = True,
                                     max_attempts = 10000)

    nn_model1.fit(X_train, y_train)

    y_pred = nn_model1.predict(X_test)



    accuracy = accuracy_score(y_test, y_pred)
    print (accuracy)