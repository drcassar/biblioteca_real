import random as rd
from evolution import Specimen
class NSGAII:
    def __init__(self):
        pass

    def config_params(self, parents, m_rate, config, elitist:bool=True):
        self.parents = parents
        self.m_rate = m_rate
        self.config = config
        self.elitist = elitist
    
    def selection(self, pool):
        """"Seleciona os indivíduos de uma pool usando o Non-Dominated-Sorting (NDS) e Crowding-Distance-Sorting (CDS)"""
        Rt = []
        fronts = self.non_dominated_sort(pool) # Executa a NDS
        print(f"Generated {len(fronts)} fronts")
        while len(Rt) < len(self.parents) / 2: # Preenche a lista de selecionados até que o tamanho máximo de 0.5 * parentes seja atingido
            f = fronts.pop(0)
            rt = Rt.copy()
            rt.extend(f) # Tenta colocar uma frente inteira dentro da lista de selecionados
            if len(rt) > len(self.parents)/2: # Se der errado,
                sorted_front = self.crowding_distance_sorting(f) # Executa o CDS na frente que não pôde ser colocada inteira
                
                while len(Rt) < len(self.parents)/2: # Preenche a lista de selecionados item a item com os indivíduos do front ordenado por CDS até que a lista esteja cheia.
                    ind = sorted_front.pop(0)
                    Rt.append(ind)
            else: 
                Rt = rt # Mas se o front couber, beleza
            
        return Rt # Retorna os melhores indivíduos.

    
    def matchmaker(self, selected:list):
        """Pareia os indivíduos selecionados"""
        pool = selected.copy()
        rd.shuffle(pool) # Emba
        assert len(pool) % 2 ==0, "There must be a even number of parents in the matchmaker"
        matches = []
        while len(pool) != 0:
            ind1 = pool.pop()
            ind2 = pool.pop()
            match = (ind1, ind2)
            matches.append(match)
        return matches
            
    
    def crossing_over(self, parent_a:Specimen, parent_b:Specimen):
        """Pega dois indivíduos e retorna os descendentes do cruzamento entre eles"""
        a_genome = parent_a.genome # Guarda os genomas dos indivíduos
        b_genome = parent_b.genome
        s1_genome = dict() 
        s2_genome = dict()
        loci = a_genome.keys() # Guarda os nomes dos genes
        for locus in loci:
            lista = [a_genome[locus], b_genome[locus]] # Emparelha os valores de um mesmo gene de cada pai 
            rd.shuffle(lista) # Embaralha
            s1_genome[locus] = lista[0]
            s2_genome[locus] = lista[1] # Distribui os genes embaralhados entre os filhos
        return [s1_genome, s2_genome]
    
    def mutation(self, genome:dict, config: dict):
        """Aplica mutação em um genoma segundo as regras de configuração passadas no ambiente Evolution"""
        genes = genome.copy()
        dice = int(1 / self.m_rate) # Gera um dado de 1/m_rate lados

        def generate_gene(config, gene):
            """Tem o mesmo comportamento da função generate_pool no ambiente Evolution, a única diferença é que esta retorna 
            um único gene em vez de um genoma inteiro"""
            if config[gene]["type"] == "int" or config[gene]["type"] == "float":
                if config[gene]["type"] == "float":
                    if len(config[gene]["interval"]) == 1:
                        return config[gene]["interval"][0]
                    else:
                        return rd.uniform(*config[gene]["interval"])
                else:
                    if len(config[gene]["interval"]) == 1:
                        return config[gene]["interval"][0]
                    else:
                        return rd.randint(*config[gene]["interval"])
            elif config[gene]["type"] == "cathegorical":
                if len(config[gene]["interval"]) == 1:
                        return config[gene]["interval"][0]
                else:
                    return rd.choice(config[gene]["interval"])
            elif config[gene]["type"] == "bool":
                if len(config[gene]["interval"]) == 1:
                    return config[gene]["interval"][0]
                else:
                    return rd.randint(0, 1)

        for gene in genes.keys():
            throw = rd.randint(0, dice) # Se tirar o número máximo no dado gerado anteriormente, 
            if throw == dice:
                genes[gene] = generate_gene(config, gene) # O gene sofre mutação.
        return genes
    
    def evolve(self):
        """Aplica seleção, cruzamento e mutação em uma população"""
        selected = self.selection(self.parents) # Seleção
        print(f"Selected {len(selected)} genomes")
        matches = self.matchmaker(selected) # Pareamento
        offspring = []
        for match in matches:
            offspring.extend(self.crossing_over(*match)) # Crossing-Over
        mutated_offspring = []
        for genome in offspring:
            mutated_offspring.append(self.mutation(genome, self.config)) #Mutação
        
        output = mutated_offspring # Genomas dos filhos
        if self.elitist: # Caso queira usar os genomas dos pais
            for parent in selected:
                output.append(parent.genome) 
        return output
    
    def non_dominated_sort(self, pool:list):
        """Aplica o non-dominated-sorting em uma pool. Retorna a pool dividida em Frentes."""
        output_fronts = []
        fronts = []
        pool_exp = []
        c_pool = pool.copy()

        for individual in c_pool:
            # Conta a quantidade de indivíduos que dominam sobre o indivíduo e lista os indivíduos sobre os quais ele domina
            cc_pool = c_pool.copy()
            cc_pool.remove(individual)
            ser = {"individuo": individual, "n_rank": 0, "domain": []}
            for colega in cc_pool:
                if individual > colega:
                    ser["domain"].append(colega)
                elif colega > individual:
                    ser["n_rank"] = ser["n_rank"] + 1
            pool_exp.append(ser)

        print('Started NDS')
        cpool_exp = pool_exp.copy()
        while len(pool_exp) != 0:
            if len(fronts) != 0: # segundo front e além:
                for i in fronts[-1]:
                    if len(i["domain"]) != 0:
                        # retira 1 do n_rank de cada colega dominado pelo indivíduo i (puxa os colegas para a próxima frente)
                        for dominated in i["domain"]:
                            # Acessar o indivíduo em pool_exp
                            indice = c_pool.index(dominated)
                            cpool_exp[indice]["n_rank"] = cpool_exp[indice]["n_rank"] - 1

                
            front_n = []
            for ss in range(len(pool_exp)):
                print(len(pool_exp))
                print(f"ss={ss}")
                print(pool_exp[ss])
                # Retira os elementos com n_rank = 0 e adiciona a um novo front
                indice = cpool_exp.index(pool_exp[ss])
                if cpool_exp[indice]["n_rank"] == 0:
                    ind = pool_exp[ss]
                    front_n.append(ind)
            for ind in front_n:
                pool_exp.remove(ind)
            if len(front_n) != 0:
                fronts.append(front_n)

            print(f"Remaining {len(pool_exp)} individuals")
        
        for front in fronts:
            # Retorna apenas os indivíduos, sem n_rank e a lista de dominados.
            o_front = []
            for i in front:
                o_front.append(i["individuo"])
            output_fronts.append(o_front)
        return output_fronts

    def crowding_distance_sorting(self, front):
        sorted_list = []
        for i in range(0, len(front)):
            # Calcula o i_distance de um elemento
            if i == 0:
                anterior = front[i].fitness
                posterior = front[i+1].fitness
            elif i == len(front) - 1:
                anterior = front[i-1].fitness
                posterior = front[i].fitness
            else: 
                anterior = front[i-1].fitness
                posterior = front[i+1].fitness
            i_distance = 0
            for k, j in zip(anterior, posterior):
                i_distance += abs(abs(k) - abs(j))
            i_distance = i_distance / len(anterior)
            ser = {"specimen": front[i], "i_distance": i_distance}

            # Posiciona o elemento em uma posição na lista em ordem decrescente de i_distances
            if len(sorted_list) == 0:
                sorted_list.append(ser)
            else:
                for l in range(0, len(sorted_list)):
                    if sorted_list[l]["i_distance"] < i_distance:
                        sorted_list.insert(l, ser)
                        break
                    elif l == len(sorted_list) - 1:
                        sorted_list.append(ser)

        sorted_front = []
        # Retira o valor i_distance.
        for i in sorted_list:
            sorted_front.append(i["specimen"])
        return sorted_front
