from ga import GA
import argparse
from utils import *
import numpy as np
import random



def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--n', type=int, default=30, help='the amount of cities')
    parser.add_argument('--pn', type=int, default=30, help='the amount of individual in population')
    parser.add_argument('--iters', type=int, default=1000, help='generation num')
    parser.add_argument('--variation_prob', type=float, default=0.3, help='probability of mutate')
    parser.add_argument('--cross_prob', type=float, default=0.99, help='probability of cross')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    assert args.n >= 10

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # init cities
    citys, dist = init_graph(args)
    draw_cities(args, citys)

    pop = init_pop(args, dist)


    ga = GA(args, pop, dist)
    result, fitness = ga.train()
    result = result[-1]
    result_pos_list = citys[result, :]
    print(result)
    print(result_pos_list)

    draw_route(args, result_pos_list)
    draw_fitness(args, fitness)



