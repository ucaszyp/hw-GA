import random

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
    def __init__(self, args, pop, dist):
        
        self.args = args
        self.pop = pop      
        self.dist = dist
        self.best = None  # 每一代的最佳个体
        self.result_list = []  # 每一代对应的解
        self.fitness_list = []  # 每一代对应的适应度


    def cross(self):
        new_gen = []
        random.shuffle(self.pop)
        for i in range(0, self.args.pn - 1, 2):
            # 父代基因
            genes1 = copy_list(self.pop[i])
            genes2 = copy_list(self.pop[i + 1])
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
            new_gen.append(genes1)
            new_gen.append(genes2)
        return new_gen

    def mutate(self, new_gen):
        for gene in new_gen:
            if random.random() < self.args.mutate_prob:
                # 翻转切片
                old_genes = copy_list(gene)
                index1 = random.randint(0, self.args.n - 2)
                index2 = random.randint(index1, self.args.n - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                gene = old_genes[:index1] + genes_mutate + old_genes[index2:]
        # 两代合并
        self.pop += new_gen

    def select(self):
        # 锦标赛
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.args.pn // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.pop)
                player = Individual(self.args, player.genes)
                group.append(player)
            group = GA.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.pop = winners


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
        for individual in self.pop:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # # 初代种群
        # self.pop = [Individual(self.args) for _ in range(self.args.pn)]
        # init population
        self.best = self.pop[0]
        # 迭代
        for i in range(self.args.iters):
            self.next_gen()
            # 连接首尾
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)

        return self.result_list, self.fitness_list
