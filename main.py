from ga import GA
from hm import HM
import argparse
from utils import *
import numpy as np
import random




def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['ga', 'hm'])
    parser.add_argument('--n', type=int, default=30, help='the amount of cities')
    parser.add_argument('--pn', type=int, default=30, help='the amount of individual in population')
    parser.add_argument('--iters', type=int, default=1000, help='generation num')
    parser.add_argument('--variation_prob', type=float, default=0.5, help='probability of mutate')
    parser.add_argument('--cross_prob', type=float, default=0.99, help='probability of cross')
    parser.add_argument('--choice', type=str, default="championship", help='roulette of championship')
    parser.add_argument('--gn', type=int, default=10, help='group size for championship')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    assert args.n >= 10

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # init cities
    cities, dist = init_graph(args)
    print(cities)
    draw_cities(args, cities)

    pop = init_pop(args, dist)

    result = 0
    result_pos_list = []
    if args.algorithm == "ga":
        ga = GA(args, pop, dist)
        result, fitness = ga.train()
        result = result[-1]
        result_pos_list = cities[result, :]

    elif args.algorithm == "hm":
        hm = HM(args, dist)
    print(result)
    print(result_pos_list)

    draw_route(args, result_pos_list)
    draw_fitness(args, fitness)



