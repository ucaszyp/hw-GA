from ga1 import GA
import argparse
from utils import *
import numpy as np
import random



def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--n', type=int, default=30, help='the amount of cities')  # 城市数量
    parser.add_argument('--individual_num', type=int, default=40, help='the amount of population')  # 个体数
    parser.add_argument('--gen_num', type=int, default=500, help='generation num')  # 迭代轮数
    parser.add_argument('--mutate_prob', type=float, default=0.9999, help='probability of mutate')  # 变异概率
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

    # pop = init_pop(args)


    ga = GA(args, dist)
    result_list, fitness_list = ga.train()
    result = result_list[-1]
    result_pos_list = citys[result, :]
    print(result)
    print(result_pos_list)

    draw_route(args, result_pos_list)
    draw_fitness(args, fitness_list)



