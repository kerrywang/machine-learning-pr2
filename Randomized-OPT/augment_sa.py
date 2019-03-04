import mlrose
import numpy as np
import time

length = 100
fitness = mlrose.OneMax()
# fitness = mlrose.MaxKColor([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (5, 8), (6, 9), (6, 8), (7, 9)])
# fitness = mlrose.Knapsack([10, 5, 2, 8, 15], [1, 2, 3, 4, 5])
problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
start = time.time()
best_state, best_fitness = mlrose.mimic(problem, max_iters=12, max_attempts=100000)
end = time.time()

print ("the time used is: {}".format(end - start))
print ('the best state found is: {}'.format(best_state))
print ('the fitness at the best state is {}'.format(best_fitness))