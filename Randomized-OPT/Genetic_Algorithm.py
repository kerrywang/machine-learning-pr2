import mlrose
import numpy
import util
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = util.preProcessData()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes = [16, 8], activation ='relu',
                                     algorithm ='genetic_alg',
                                     max_iters = 1000, bias = True, is_classifier = True,
                                     learning_rate = 1, early_stopping = True,
                                     clip_max = 1, max_attempts = 5000, mutation_prob=0.1)

    nn_model1.fit(X_train, y_train)

    y_pred = nn_model1.predict(X_test)
    print (y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print (accuracy)