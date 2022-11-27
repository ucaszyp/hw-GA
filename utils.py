import numpy as np
import matplotlib.pyplot as plt
import random


def init_graph(args):
    cities = np.random.randint(0 , args.n, size=(args.n, 2))
    dist = np.zeros([args.n, args.n])

    for i in range(args.n):
        for j in range(args.n):
            if i != j:
                dist[i,j] = get_dist(cities, i, j)
    return cities, dist

def init_pop(args, dist):
    pop = []
    for _ in range(args.pn):
        ind = {}
        gene = [i for i in range(args.n)]
        random.shuffle(gene)
        ind['gene'] = gene
        ind['fit'] = compute_fitness(args, gene, dist)
        pop.append(ind)
    return pop

def compute_fitness(args, gene, dist):
    fitness = 0

    for i in range(args.n - 1):
        fitness += dist[gene[i], gene[i + 1]]
    fitness += dist[gene[0], gene[-1]]

    return fitness

def get_dist(cities, i, j):
    return np.sqrt(np.square(cities[i][0] - cities[j][0]) + np.square(cities[i][1] - cities[j][1]))

def draw_cities(args, cities):
    plt.figure()
    plt.scatter(cities[:, 0], cities[:, 1])
    plt.title("Cityes")
    plt.legend()
    plt.savefig("city_{}.jpg".format(args.pn))
    plt.close()

def draw_route(args, result):
    plt.figure()
    plt.plot(result[:, 0], result[:, 1])
    plt.title("Route Result")
    plt.legend()
    plt.savefig("route_{}_1.jpg".format(args.pn))
    plt.close()


def draw_fitness(args, fitness_list):
    plt.figure()
    plt.ylim(130, 350)
    plt.plot(fitness_list)
    plt.title("Fitness Result")
    plt.legend()
    plt.savefig("fitness_{}.jpg".format(args.pn))
    plt.close()
