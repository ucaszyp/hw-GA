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

def get_sort(group):
    for i in range(1, len(group)):
        for j in range(0, len(group) - i):
            if group[j]['fit'] > group[j + 1]['fit']:
                group[j], group[j + 1] = group[j + 1], group[j]
    return group

def draw_cities(args, cities):
    plt.figure()
    plt.scatter(cities[:, 0], cities[:, 1])
    plt.title("Cityes")
    plt.legend()
    plt.savefig("vis/city_{}_{}.jpg".format(args.algorithm, args.n))
    plt.close()

def draw_route(args, result):
    plt.figure()
    plt.plot(result[:, 0], result[:, 1])
    plt.title("Route Result")
    plt.legend()
    plt.savefig("vis/route_{}_{}.jpg".format(args.algorithm, args.n))
    plt.close()


def draw_fitness(args, fitness_list):
    max_fit = max(fitness_list)
    min_fit = min(fitness_list)
    plt.figure()
    plt.ylim(min_fit - 10, max_fit + 10)
    plt.plot(fitness_list)
    if args.algorithm == 'ga':
        plt.title("Fitness Result")
    elif args.algorithm == 'hm':
        plt.title("Energy Result")
    plt.legend()
    if args.algorithm == 'ga':
        plt.savefig("vis/fitness_{}_{}.jpg".format(args.algorithm, args.n))
    elif args.algorithm == 'hm':
        plt.savefig("vis/energy_{}_{}.jpg".format(args.algorithm, args.n))
    plt.close()
