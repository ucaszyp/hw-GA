import random
from utils import *
from tqdm import tqdm

class GA:
    def __init__(self, args, pop, dist):
        
        self.args = args
        self.pop = pop      
        self.dist = dist
        self.best = self.pop[0]
        self.result = [] 
        self.fitness = []
        self.new_ind = []
        self.cities_prob = []

    def gen(self):

        # random cross
        random.shuffle(self.pop)
        for i in range(0, self.args.pn - 1, 2):

            gene1 = self.pop[i]['gene'].copy()
            gene2 = self.pop[i + 1]['gene'].copy()

            begin_idx = random.randint(0, self.args.n - 2)
            end_idx = random.randint(begin_idx, self.args.n - 1)

            pos1_r = {value: idx for idx, value in enumerate(gene1)}
            pos2_r = {value: idx for idx, value in enumerate(gene2)}


            if random.random() < self.args.cross_prob:
                for j in range(begin_idx, end_idx):
                    pos1, pos2 = pos1_r[gene2[j]], pos2_r[gene1[j]]
                    gene1[j], gene1[pos1] = gene1[pos1], gene1[j]
                    gene2[j], gene2[pos2] = gene2[pos2], gene2[j]
                    
                    pos1_r[gene1[j]], pos1_r[gene2[j]] = pos1, j
                    pos2_r[gene1[j]], pos2_r[gene2[j]] = j, pos2

            fit1 = compute_fitness(self.args, gene1, self.dist)
            fit2 = compute_fitness(self.args, gene2, self.dist)

            new_gene1 = {}
            new_gene1['fit'] = fit1
            new_gene1['gene'] = gene1

            new_gene2 = {}
            new_gene2['fit'] = fit2
            new_gene2['gene'] = gene2

            self.new_ind.append(new_gene1)
            self.new_ind.append(new_gene2)

        # random variation
        for ind in self.new_ind:
            if random.random() < self.args.variation_prob:
                old_gene = ind['gene'].copy()
                
                begin_idx = random.randint(0, self.args.n - 2)
                end_idx = random.randint(begin_idx, self.args.n - 1)
                
                gene_variation = old_gene[begin_idx: end_idx]
                gene_variation.reverse()
                
                ind["gene"] = old_gene[:begin_idx] + gene_variation + old_gene[end_idx:]

        self.pop += self.new_ind

        # select
        if self.args.choice == "championship":
            group_winner = self.args.pn // self.args.gn
            winners = []
            for i in range(self.args.gn):
                group = []
                for j in range(self.args.gn):
                    player = random.choice(self.pop)
                    group.append(player)
                group = get_sort(group)
                winners += group[:group_winner]
            self.pop = winners

        elif self.args.choice == 'roulette':
            self.cities_prob = [(1 / self.pop[i]["fit"]) for i in range(len(self.pop))]

            total_prob = sum(self.cities_prob)
            roulette = []
            new_pop = []

            while len(new_pop) < 30:
                temp_prob = random.uniform(0.0, total_prob)
                for j in range(len(self.pop)):
                    temp_prob -= self.cities_prob[j]
                    if temp_prob < 0 and j not in roulette:  
                        roulette.append(j)
                        new_pop.append(self.pop[j])
                        break

            self.pop = new_pop

        # update
        for ind in self.pop:
            if ind['fit'] < self.best['fit']:
                self.best = ind
        
        self.new_ind = []

    def train(self):
        # gen
        for i in tqdm(range(self.args.iters)):
            self.gen()
            result = self.best['gene'].copy()
            result.append(result[0])
            self.result.append(result)
            self.fitness.append(self.best['fit'])

        return self.result, self.fitness
