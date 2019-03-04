import mlrose
import numpy as np
import time
def random_graph_generator(nodeCount):
    edge = []
    numEdge = np.random.randint(nodeCount * 2, nodeCount ** 2, 1)[0]
    print (numEdge)
    for _ in range(numEdge):
        node1, node2 = np.random.randint(0, nodeCount, 2)
        edge.append((node1, node2))
    return edge
length = 30
graph = random_graph_generator(length)

for opt, maxIter in [(mlrose.mimic, 3), (mlrose.random_hill_climb, 1900), (mlrose.genetic_alg, 10), (mlrose.simulated_annealing, 1800)]:
    fitness = mlrose.MaxKColor(graph)
    # fitness = mlrose.MaxKColor([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 7), (5, 8), (6, 9), (6, 8), (7, 9)])
    # fitness = mlrose.Knapsack([10, 5, 2, 8, 15], [1, 2, 3, 4, 5])
    problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=False, max_val=length)
    start = time.time()
    best_state, best_fitness = opt(problem, max_iters=maxIter, max_attempts=10000)
    end = time.time()

    print ("the time used is: {}".format(end - start))
    print ('the best state found is: {}'.format(best_state))
    print ('the fitness at the best state is {}'.format(best_fitness))
