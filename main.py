import os
import numpy as np
import random

from ga import GA
from hnn import HM
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['ga', 'hm'], default='hm')
    parser.add_argument('--n', type=int, default=25, help='the amount of cities')
    parser.add_argument('--pn', type=int, default=40, help='the amount of individual in population')
    parser.add_argument('--iters', type=int, default=100000, help='generation num')
    parser.add_argument('--variation_prob', type=float, default=0.8, help='probability of mutate')
    parser.add_argument('--cross_prob', type=float, default=0.999, help='probability of cross')
    parser.add_argument('--choice', type=str, default="championship", help='roulette of championship')
    parser.add_argument('--gn', type=int, default=10, help='group size for championship')
    parser.add_argument('--u0', type=float, default=0.0009)
    parser.add_argument('--eta', type=float, default=0.0001, help='step for update u')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    assert args.n >= 10

    random.seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists("./vis"):
        os.makedirs("./vis")

    # init cities
    cities, dist = init_graph(args)
    print(cities)
    draw_cities(args, cities)

    result = 0
    result_pos_list = []

    if args.algorithm == "ga":
        pop = init_pop(args, dist)
        ga = GA(args, pop, dist)
        result, fitness = ga.train()
        result = result[-1]
        draw_fitness(args, fitness)

    elif args.algorithm == "hm":
        hm = HM(args, dist)
        energy, result = hm.train()
        draw_fitness(args, energy)
    
    result_pos_list = cities[result, :]
    draw_route(args, result_pos_list)
    



