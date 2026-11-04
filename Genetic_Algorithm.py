import random
import time
import operator
import matplotlib.pyplot as plt
import service_simulator
import numpy as np

'''
class Knapsack(object):

    def __init__(self, populationSize, numGenerations, mutationRate,
                 patience=20, verbose=False):

        self.itemList = 50
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.mutationRate = mutationRate
        self.verbose = verbose

        self.population = []
        self.fitness = []
        self.parents = []

        self.patience = patience
        self.patienceCounter = 0
        self.bestFitness = float("inf")
        self.bestfoundAt = -1
        self.bestIndividual = None
        self.listOfBestFitness = []

        self.initialisePopulation()



    def initialisePopulation(self):

        for _ in range(self.populationSize):

            individual = [1 for _ in range(50)]

            remaining_employees = 30

            for _ in range(remaining_employees):
                department = random.randrange(len(individual))
                individual[department] += 1

            self.population.append(individual)

        print("First population initialisation done")


    def calcFitness(self, employee_setting):

        means = []
        maxs = []

        # CHANGE 4: save RNG state so seeding doesn't affect the GA's randomness
        state = random.getstate()
        np_state = np.random.get_state()

        for rep in range(3):

            # CHANGE 4: fixed seed per replication -> every allocation
            # is evaluated on the same scenarios
            random.seed(rep)
            np.random.seed(rep)

            _x, wait_times_per_service = service_simulator.init_and_simulate(
                employee_setting
            )

            sums = [
                np.sum(wait_times_per_service[j])
                for j in range(len(wait_times_per_service))
            ]

            counts = [
                len(wait_times_per_service[j])
                for j in range(len(wait_times_per_service))
            ]

            maximums = [
                np.max(wait_times_per_service[j])
                for j in range(len(wait_times_per_service))
                if len(wait_times_per_service[j]) > 0
            ]

            total_count = np.sum(counts)

            mean_wait = np.sum(sums) / total_count if total_count > 0 else 0

            max_wait = np.max(maximums) if maximums else 0

            means.append(mean_wait)
            maxs.append(max_wait)

        # CHANGE 4: restore RNG state
        random.setstate(state)
        np.random.set_state(np_state)

        fitness_score = np.mean(means) * 0.89 + np.mean(maxs) * 0.11

        return fitness_score

    def evaluation(self, generationNumber):

        self.fitness = []

        for individual in self.population:

            ft = self.calcFitness(individual)

            self.fitness.append((ft, individual))

        self.fitness.sort(key=operator.itemgetter(0))

        if self.fitness[0][0] < self.bestFitness:

            self.bestFitness = self.fitness[0][0]
            self.bestIndividual = self.fitness[0][1].copy()
            self.bestfoundAt = generationNumber
            self.patienceCounter = 0

        # CHANGE 3: count generations without improvement here
        else:
            self.patienceCounter += 1

        print("Evaluation complete")

    def selectParents(self):

        numElites = max(2, self.populationSize // 4)

        numParents = min(numElites * 2, self.populationSize)

        self.parents = [
            x[1].copy() for x in self.fitness[:numElites]
        ]

        otherParentsIndices = random.sample(
            range(numElites, len(self.fitness)),
            numParents - numElites
        )

        for i in otherParentsIndices:

            self.parents.append(self.fitness[i][1].copy())

        print("Parents selected")

    # CHANGE 1: repair() restores the 80-employee total, keeping every gene >= 1
    def repair(self, ind):

        ind = ind.copy()
        diff = sum(ind) - 80

        while diff != 0:

            i = random.randrange(len(ind))

            if diff > 0 and ind[i] > 1:
                ind[i] -= 1
                diff -= 1

            elif diff < 0:
                ind[i] += 1
                diff += 1

        return ind

    def crossover(self, sack1, sack2):

        cut = random.randrange(1, len(sack1))

        c1 = sack1[:cut] + sack2[cut:]
        c2 = sack2[:cut] + sack1[cut:]

        return self.repair(c1), self.repair(c2)


    def mutation(self, employee_setting):

        employee_setting = employee_setting.copy()

        for i in range(len(employee_setting)):

            if random.random() < self.mutationRate:

                possible_departments = [
                    x for x in range(len(employee_setting)) if x != i
                ]

                j = random.choice(possible_departments)

                if employee_setting[i] > 1:

                    employee_setting[i] -= 1
                    employee_setting[j] += 1

        return employee_setting

    def validateIndividual(self, individual):

        assert len(individual) == 50
        assert sum(individual) == 80
        assert min(individual) >= 1

    def run(self):

        for g in range(self.numGenerations):

            generation_start = time.time()

            self.evaluation(g)

            self.selectParents()

            newPopulation = []


            newPopulation.append(self.bestIndividual.copy())


            while len(newPopulation) < self.populationSize:

                p1, p2 = random.sample(self.parents, 2)

                child1, child2 = self.crossover(p1, p2)

                newPopulation.append(self.mutation(child1))
                newPopulation.append(self.mutation(child2))

            newPopulation = newPopulation[:self.populationSize]

            for individual in newPopulation:
                self.validateIndividual(individual)

            # CHANGE 3: removed `self.patienceCounter += 1` (now in evaluation)

            self.listOfBestFitness.append(self.bestFitness)

            generation_time = time.time() - generation_start

            print(
                "Generation {} | Best fitness: {:.4f} | Time: {:.2f}s".format(
                    g, self.bestFitness, generation_time
                )
            )

            if self.patienceCounter >= self.patience:

                print("Stopped after {} generations".format(g))
                break

            self.population = newPopulation

        print(
            "Best individual is {} ; fitness value is {} ; "
            "found at generation {}".format(
                self.bestIndividual,
                self.bestFitness,
                self.bestfoundAt
            )
        ) 

ga = Knapsack(
    populationSize=50,
    numGenerations=10,
    mutationRate=0.05
)

ga.run()  

''' 

employee_setting = [1, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 6, 2, 1, 2, 1, 1, 1, 1, 1, 1, 3, 2, 3, 1, 1, 1, 1, 1, 2, 6, 1, 2, 1, 1, 1, 2, 4, 1, 1, 2, 2]
means = [0 for _ in range(10)]
maxs = [0 for _ in range(10)]
for _ in range(10):
  _x, wait_times_per_service = service_simulator.init_and_simulate(employee_setting)
  sums = [np.sum(wait_times_per_service[j]) for j in range(len(wait_times_per_service))]
  counts = [len(wait_times_per_service[j]) for j in range(len(wait_times_per_service))]
  maximums= [np.max(wait_times_per_service[j]) for j in range(len(wait_times_per_service))]
  means[_] =  np.sum(sums)/np.sum(counts)
  maxs[_] = np.max(maximums)

print(means)
print(maxs)
print("Mean waiting time averaged over 10 simulations : ", np.mean(means))
print("Max waiting time  averaged over 10 simulations : ", np.mean(maxs))