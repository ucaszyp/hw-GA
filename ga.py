import random

class GA:
    def __init__(self, args, pop, dist):
        
        self.args = args
        self.pop = pop      
        self.dist = dist
        self.best = None
        self.result = [] 
        self.fitness = []
        self.new_ind = []


    def cross(self):
        random.shuffle(self.pop)
        
        for i in range(0, self.args.pn - 1, 2):

            gene1 = self.pop[i]['gene'].copy()
            gene2 = self.pop[i + 1]['gene'].copy()

            begin_idx = random.randint(0, self.args.n - 2)
            end_idx = random.randint(begin_idx, self.args.n - 1)

            pos1_recorder = {value: idx for idx, value in enumerate(gene1)}
            pos2_recorder = {value: idx for idx, value in enumerate(gene2)}


            if random.random() < self.args.cross_prob:
                for j in range(begin_idx, end_idx):
                    pos1, pos2 = pos1_recorder[gene2[j]], pos2_recorder[gene1[j]]
                    gene1[j], gene1[pos1] = gene1[pos1], gene1[j]
                    gene2[j], gene2[pos2] = gene2[pos2], gene2[j]
                    
                    pos1_recorder[gene1[j]], pos1_recorder[gene2[j]] = pos1, j
                    pos2_recorder[gene1[j]], pos2_recorder[gene2[j]] = j, pos2

            fit1 = self.compute_fitness(gene1)
            fit2 = self.compute_fitness(gene2)

            new_gene1 = {}
            new_gene1['fit'] = fit1
            new_gene1['gene'] = gene1

            new_gene2 = {}
            new_gene2['fit'] = fit2
            new_gene2['gene'] = gene2

            self.new_ind.append(new_gene1)
            self.new_ind.append(new_gene2)
            

    def variation(self):
        for ind in self.new_ind:
            
            if random.random() < self.args.variation_prob:
                old_gene = ind['gene'].copy()
                
                begin_idx = random.randint(0, self.args.n - 2)
                end_idx = random.randint(begin_idx, self.args.n - 1)
                
                gene_variation = old_gene[begin_idx: end_idx]
                gene_variation.reverse()
                
                ind["gene"] = old_gene[:begin_idx] + gene_variation + old_gene[end_idx:]

        self.pop += self.new_ind
        

    def select(self):

        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.args.pn // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.pop)
                group.append(player)
            group = GA.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.pop = winners
        self.new_ind = []


    def compute_fitness(self, gene):
        fitness = 0

        for i in range(self.args.n - 1):
            fitness += self.dist[gene[i], gene[i + 1]]
        fitness += self.dist[gene[0], gene[-1]]

        return fitness


    @staticmethod
    def rank(group):
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j]['fit'] > group[j + 1]['fit']:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        self.cross()
        self.variation()
        self.select()

        for ind in self.pop:
            if ind['fit'] < self.best['fit']:
                self.best = ind

    def train(self):
        # init population
        self.best = self.pop[0]

        # train
        for i in range(self.args.iters):
            
            self.next_gen()
            result = self.best['gene'].copy()
            result.append(result[0])
            self.result.append(result)
            self.fitness.append(self.best['fit'])

        return self.result, self.fitness
