
import random
import time
import operator
import matplotlib.pyplot as plt
import service_simulator
import numpy as np


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

        for _ in range(3):

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

    def crossover(self, sack1, sack2, threshold):

        while threshold <= len(sack1):

            tmp_1 = sack1[threshold:]
            tmp_2 = sack2[threshold:]

            sack_1 = sack1[:threshold]
            sack_2 = sack2[:threshold]

            if sum(tmp_1 + sack_2) == 80:

                sack_1.extend(tmp_2)
                sack_2.extend(tmp_1)

                return sack_1, sack_2

            threshold += 1

        return False


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

            for i in range(len(self.parents)):

                parent1 = self.parents[i]

                for j in range(i + 1, len(self.parents)):

                    parent2 = self.parents[j]

                    result = self.crossover(parent1, parent2, 1)

                    if result is not False:

                        child1, child2 = result

                        newPopulation.append(self.mutation(child1))
                        newPopulation.append(self.mutation(child2))

                    if len(newPopulation) >= self.populationSize:
                        break

                if len(newPopulation) >= self.populationSize:
                    break

            while len(newPopulation) < self.populationSize:

                newPopulation.append(
                    random.choice(self.parents).copy()
                )

            newPopulation = newPopulation[:self.populationSize]

            for individual in newPopulation:
                self.validateIndividual(individual)

            self.patienceCounter += 1

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
    populationSize=20,
    numGenerations=10,
    mutationRate=0.05
)

ga.run()
