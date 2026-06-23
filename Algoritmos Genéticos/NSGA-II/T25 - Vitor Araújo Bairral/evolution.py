import random as rd

class Specimen:
    def __init__(self, genome, fitness):
        self.genome = genome
        self.fitness = fitness
    
    def __gt__(self, other):
        """Afirma se um indivíduo domina sobre o outro. Um indivíduo domina sobre outro se todas suas funções objetivos são iguais ou maiores 
        do que as do segundo e pelo menos uma função objetivo é maior do que a do segundo"""
        a = self.fitness
        b = other.fitness
        first = True
        second = False
        for i, j in zip(a, b):
            if j > i:
                first = False
        for i, j in zip(a, b):
            if i > j:
                second = True
        return first and second
    

class Evolution:
    def __init__(self, pool_size, n_batches, algorithm, mutation, config):
        self.pool_size = pool_size
        self.n_batches = n_batches
        self.algorithm = algorithm
        self.mutation = mutation
        self.config = config
    
    def generate_pool(self, config):
        pool = []
        for j in range(self.pool_size):
            genome = dict()
            genes = config.keys()
            for i in genes:
                if config[i]["type"] == "int" or config[i]["type"] == "float":
                    if config[i]["type"] == "float":
                        if len(config[i]["interval"]) == 1:
                            genome[i] = config[i]["interval"][0]
                        else:
                            genome[i] = rd.uniform(*config[i]["interval"])
                    else:
                        if len(config[i]["interval"]) == 1:
                            genome[i] = config[i]["interval"][0]
                        else:
                            genome[i] = rd.randint(*config[i]["interval"])
                elif config[i]["type"] == "cathegorical":
                    if len(config[i]["interval"]) == 1:
                            genome[i] = config["interval"][0]
                    else:
                        genome[i] = rd.choice(config[i]["interval"])
                elif config[i]["type"] == "bool":
                    if len(config[i]["interval"]) == 1:
                        genome[i] = config[i]["interval"][0]
                    else:
                        genome[i] = rd.randint(0, 1)
            pool.append(genome)
        return pool
    
    def pass_through(self, pool):
        return list()

    def step(self, pool):
        batch = self.pass_through(pool)
        print("Passed through")
        optim = self.algorithm()
        optim.config_params(batch, self.mutation, self.config)
        result = optim.evolve()
        print("Evolved batch")
        return result
    
    def optimizate(self):
        parents = self.generate_pool(self.config)
        pool = self.step(parents)
        print("Geração 0 concluída")
        for _ in range(1, self.n_batches):
            pool = self.step(pool)
            print(f"Geração {_} concluída")
        return pool





