import random

city_dist_mat = None

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类
class Individual:
    def __init__(self, args, genes=None):
        # 随机生成序列
        self.args = args
        if genes is None:
            genes = [i for i in range(self.args.n)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0.0
        for i in range(self.args.n - 1):
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx]
        # 连接首尾
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness


class GA:
    def __init__(self, args, input_):
        global city_dist_mat
        city_dist_mat = input_
        self.best = None  # 每一代的最佳个体
        self.individual_list = []  # 每一代的个体列表
        self.result_list = []  # 每一代对应的解
        self.fitness_list = []  # 每一代对应的适应度
        self.args = args

    def cross(self):
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, self.args.individual_num - 1, 2):
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, self.args.n - 2)
            index2 = random.randint(index1, self.args.n - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # 交叉
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(self.args, genes1))
            new_gen.append(Individual(self.args, genes2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < self.args.mutate_prob:
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, self.args.n - 2)
                index2 = random.randint(index1, self.args.n - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        # 两代合并
        self.individual_list += new_gen

    def select(self):
        # 锦标赛
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.args.individual_num // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(self.args, player.genes)
                group.append(player)
            group = GA.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners


    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # 获得这一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # 初代种群
        self.individual_list = [Individual(self.args) for _ in range(self.args.individual_num)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(self.args.gen_num):
            self.next_gen()
            # 连接首尾
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list
