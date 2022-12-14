import numpy as np
from tqdm import tqdm

class HM:
    def __init__(self, args, dist):
        self.args = args
        self.dist = dist
        self.energy = []
        self.best_dist = 1e8
        self.u0 = args.u0
        self.eta = args.eta

    def get_dist(self, path):
        dist = 0.
        for i in range(len(path) - 1):
            dist += self.dist[path[i]][path[i+1]]
        if dist < self.best_dist:
            self.best_dist = dist
            return 1
        else:
            return 0

    def calc_du(self, v, distance):
        a = np.sum(v, axis=0) - 1
        b = np.sum(v, axis=1) - 1
        A = self.args.n * self.args.n
        D = self.args.n / 2
        t1 = np.zeros((self.args.n, self.args.n))
        t2 = np.zeros((self.args.n, self.args.n))
        for i in range(self.args.n):
            for j in range(self.args.n):
                t1[i, j] = a[j]
        for i in range(self.args.n):
            for j in range(self.args.n):
                t2[j, i] = b[j]
        c_1 = v[:, 1:self.args.n]
        c_0 = np.zeros((self.args.n, 1))
        c_0[:, 0] = v[:, 0]
        c = np.concatenate((c_1, c_0), axis=1)
        c = np.dot(distance, c)
        return -A * (t1 + t2) - D * c

    def update_u(self, u, du):
        return u + du * self.eta

    def update_v(self, u):
        return 1 / 2 * (1 + np.tanh(u / self.u0))

    def calc_energy(self, v, distance):
        A = self.args.n * self.args.n
        D = self.args.n / 2
        t1 = np.sum(np.power(np.sum(v, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(v, axis=1) - 1, 2))
        idx = [i for i in range(1, self.args.n)]
        idx = idx + [0]
        Vt = v[:, idx]
        t3 = distance * Vt
        t3 = np.sum(np.sum(np.multiply(v, t3)))
        e = 0.5 * (A * (t1 + t2) + D * t3)
        self.energy.append(e)

    def check_path(self, v):
        newV = np.zeros([self.args.n, self.args.n])
        route = []
        for i in range(self.args.n):
            mm = np.max(v[:, i])
            for j in range(self.args.n):
                if v[j, i] == mm:
                    newV[j, i] = 1
                    route += [j]
                    break
        return route, newV

    def train(self):

        distance = self.dist
        
        u = 1 / 2 * self.u0 * np.log(self.args.n - 1) + (2 * (np.random.random((self.args.n, self.args.n))) - 1)
        v = self.update_v(u)
        H_path = []
        for n in tqdm(range(self.args.iters)):
            du = self.calc_du(v, distance)
            u = self.update_u(u, du)
            v = self.update_v(u)
            self.calc_energy(v, distance)
            route, newV = self.check_path(v)
            
            if len(np.unique(route)) == self.args.n:
                route.append(route[0])
                is_best = self.get_dist(route)
                if is_best:
                    H_path = []
                    print(route)
                    [H_path.append((route[i], route[i + 1])) for i in range(len(route) - 1)]

        if len(H_path) > 0:
            final_path = []
            for h_path in H_path:
                final_path.append(h_path[0])
            final_path.append(H_path[0][0])
            return self.energy, final_path
        else:
            print('没有找到最优解')
            return None