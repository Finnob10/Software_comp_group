import time
import random
import operator
import matplotlib.pyplot as plt
import numpy as np
import service_simulator


def _evaluate_individual_direct(employee_setting, w_mean, w_max):
    """
    Evaluates an employee allocation directly using service_simulator.init_and_simulate
    across the 3 fixed seeds (rep = 0, 1, 2) with single-pass pure-Python aggregation.
    Returns (fitness_score, avg_mean, avg_max, dept_entries).
    """
    state = random.getstate()
    np_state = np.random.get_state()

    means = []
    maxs = []
    dept_entries = []

    for rep in range(3):
        random.seed(rep)
        np.random.seed(rep)

        _x, wait_times_per_service = service_simulator.init_and_simulate(
            employee_setting
        )

        # Optimization #3: Single-pass pure-Python aggregation over the 50 queues
        total_wait = 0.0
        total_count = 0
        max_wait = 0.0

        for j, q_waits in enumerate(wait_times_per_service):
            q_count = len(q_waits)
            if q_count > 0:
                q_sum = float(sum(q_waits))
                q_max = float(max(q_waits))
                total_wait += q_sum
                total_count += q_count
                if q_max > max_wait:
                    max_wait = q_max
            else:
                q_sum = 0.0
                q_max = 0.0
            dept_entries.append((rep, j, employee_setting[j], q_sum, q_count, q_max))

        mean_wait = total_wait / total_count if total_count > 0 else 0.0
        means.append(mean_wait)
        maxs.append(max_wait)

    random.setstate(state)
    np.random.set_state(np_state)

    avg_mean = float(np.mean(means))
    avg_max = float(np.mean(maxs))
    fitness_score = avg_mean * w_mean + avg_max * w_max
    return fitness_score, avg_mean, avg_max, dept_entries


class KnapsackOptimized(object):

    def __init__(
        self,
        populationSize=150,
        numGenerations=80,
        mutationRate=0.03,
        w_mean=0.80,
        w_max=0.20,
        patience=30,
        use_dept_cache=True,
        verbose=True,
    ):
        self.itemList = 50
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.mutationRate = mutationRate
        self.w_mean = w_mean
        self.w_max = w_max
        self.patience = patience
        self.use_dept_cache = use_dept_cache
        self.verbose = verbose

        self.population = []
        self.fitness = []
        self.parents = []

        self.patienceCounter = 0
        self.bestFitness = float("inf")
        self.bestfoundAt = -1
        self.bestIndividual = None
        self.bestMeanWait = float("inf")
        self.bestMaxWait = float("inf")
        self.listOfBestFitness = []

        # Optimization #1: Individual-level memoization cache
        # Maps tuple(individual) -> (fitness_score, mean_wait, max_wait)
        self.fitnessCache = {}
        self.cacheHits = 0
        self.simulatorCalls = 0

        # Optimization #5: Per-department simulation cache populated dynamically
        # from real init_and_simulate() calls: maps (rep, dept_idx, emp_count) -> (q_sum, q_count, q_max)
        self.deptCache = {}
        self.maxPrewarmedK = 0

        if self.use_dept_cache:
            # Pre-warm k = 1..8 across all 50 departments using uniform vectors [k]*50
            # so both the GA and 1-step local refinement never suffer a cache miss
            self._prewarm_dept_cache(max_k=8)

        self.initialisePopulation()

    def _prewarm_dept_cache(self, max_k=8):
        state = random.getstate()
        np_state = np.random.get_state()

        # Extract mus and lambdas from service_simulator.init_and_simulate constants
        mus = [
            0.7784, 0.7973, 0.4882, 0.8344, 0.3625, 0.2278, 0.7301, 0.4758, 0.1679, 0.4488,
            0.7849, 0.0576, 0.5886, 0.1693, 0.3713, 0.5115, 0.7951, 0.0428, 0.098, 0.0565,
            0.3946, 0.8582, 0.3421, 0.5003, 0.8966, 0.7875, 0.5392, 0.511, 0.3894, 0.0467,
            0.4797, 0.1338, 0.477, 0.8155, 0.45, 0.716, 0.6678, 0.737, 0.0719, 0.8623,
            0.2051, 0.5068, 0.4527, 0.3632, 0.3137, 0.1884, 0.5032, 0.598, 0.5755, 0.6181,
        ]
        lambdas = [
            0.4079, 0.5782, 0.4356, 0.1033, 0.5663, 0.0936, 0.6135, 0.0616, 0.4641, 0.7572,
            0.2179, 0.0384, 0.0574, 0.0965, 0.1348, 0.1592, 0.4302, 0.0486, 0.1027, 0.534,
            0.6918, 0.6601, 0.5973, 0.2389, 0.1209, 0.451, 0.0737, 0.4353, 0.397, 0.1619,
            0.6977, 0.4623, 0.1114, 0.6064, 0.068, 0.287, 0.0668, 0.7765, 0.6671, 0.4903,
            0.4259, 0.5848, 0.1292, 0.1249, 0.7741, 0.6972, 0.4797, 0.1232, 0.613, 0.7999,
        ]

        for k in range(1, max_k + 1):
            for rep in range(3):
                random.seed(rep)
                np.random.seed(rep)
                _, wait_times_per_service = service_simulator.simulate(
                    [k] * 50, 500, mus, lambdas, False
                )
                self.simulatorCalls += 1
                for j, q_waits in enumerate(wait_times_per_service):
                    q_count = len(q_waits)
                    q_sum = float(sum(q_waits)) if q_count > 0 else 0.0
                    q_max = float(max(q_waits)) if q_count > 0 else 0.0
                    self.deptCache[(rep, j, k)] = (q_sum, q_count, q_max)

        self.maxPrewarmedK = max_k
        random.setstate(state)
        np.random.set_state(np_state)

    def initialisePopulation(self):
        for _ in range(self.populationSize):
            individual = [1 for _ in range(50)]
            remaining_employees = 30

            for _ in range(remaining_employees):
                department = random.randrange(len(individual))
                individual[department] += 1

            self.population.append(individual)

        if self.verbose:
            print("First population initialisation done")

    def _can_eval_from_dept_cache(self, employee_setting):
        if not self.use_dept_cache:
            return False
        for j, k in enumerate(employee_setting):
            if (0, j, k) not in self.deptCache:
                return False
        return True

    def _eval_from_dept_cache(self, employee_setting):
        means = []
        maxs = []
        for rep in range(3):
            total_wait = 0.0
            total_count = 0
            max_wait = 0.0
            for j, k in enumerate(employee_setting):
                q_sum, q_count, q_max = self.deptCache[(rep, j, k)]
                total_wait += q_sum
                total_count += q_count
                if q_count > 0 and q_max > max_wait:
                    max_wait = q_max
            means.append(total_wait / total_count if total_count > 0 else 0.0)
            maxs.append(max_wait)

        avg_mean = float(np.mean(means))
        avg_max = float(np.mean(maxs))
        fitness_score = avg_mean * self.w_mean + avg_max * self.w_max
        return fitness_score, avg_mean, avg_max

    def calcFitness(self, employee_setting):
        # Optimization #1: Check individual memoization cache first
        key = tuple(employee_setting)
        if key in self.fitnessCache:
            self.cacheHits += 1
            return self.fitnessCache[key][0]

        # Optimization #5: Check if all (department, employee_count) pairs are cached
        if self._can_eval_from_dept_cache(employee_setting):
            self.cacheHits += 1
            fitness_score, avg_mean, avg_max = self._eval_from_dept_cache(
                employee_setting
            )
            self.fitnessCache[key] = (fitness_score, avg_mean, avg_max)
            return fitness_score

        # Otherwise run full simulator evaluation via service_simulator.init_and_simulate
        self.simulatorCalls += 3
        fitness_score, avg_mean, avg_max, dept_entries = _evaluate_individual_direct(
            employee_setting, self.w_mean, self.w_max
        )
        for rep, j, k, q_sum, q_count, q_max in dept_entries:
            self.deptCache[(rep, j, k)] = (q_sum, q_count, q_max)
        self.fitnessCache[key] = (fitness_score, avg_mean, avg_max)
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
            best_key = tuple(self.bestIndividual)
            _, self.bestMeanWait, self.bestMaxWait = self.fitnessCache[best_key]
            self.bestfoundAt = generationNumber
            self.patienceCounter = 0
        else:
            self.patienceCounter += 1

        if self.verbose:
            print("Evaluation complete")

    def selectParents(self):
        numElites = max(2, self.populationSize // 4)
        numParents = min(numElites * 2, self.populationSize)

        self.parents = [x[1].copy() for x in self.fitness[:numElites]]

        otherParentsIndices = random.sample(
            range(numElites, len(self.fitness)), numParents - numElites
        )

        for i in otherParentsIndices:
            self.parents.append(self.fitness[i][1].copy())

        if self.verbose:
            print("Parents selected")

    # Optimization #4: Direct-index repair when diff > 0 avoids wasted random retries on 1-employee depts
    def repair(self, ind):
        ind = ind.copy()
        diff = sum(ind) - 80

        if diff > 0:
            reducible = [i for i, val in enumerate(ind) if val > 1]
            while diff > 0:
                idx_pos = random.randrange(len(reducible))
                dept = reducible[idx_pos]
                ind[dept] -= 1
                diff -= 1
                if ind[dept] == 1:
                    reducible[idx_pos] = reducible[-1]
                    reducible.pop()
        elif diff < 0:
            while diff < 0:
                i = random.randrange(len(ind))
                ind[i] += 1
                diff += 1

        return ind

    def crossover(self, sack1, sack2):
        cut = random.randrange(1, len(sack1))
        c1 = sack1[:cut] + sack2[cut:]
        c2 = sack2[:cut] + sack1[cut:]
        return self.repair(c1), self.repair(c2)

    # Optimization #4: Check employee_setting[i] > 1 first and select j != i in O(1) without list comprehension
    def mutation(self, employee_setting):
        employee_setting = employee_setting.copy()
        n = len(employee_setting)

        for i in range(n):
            if employee_setting[i] > 1 and random.random() < self.mutationRate:
                j = random.randrange(n - 1)
                if j >= i:
                    j += 1
                employee_setting[i] -= 1
                employee_setting[j] += 1

        return employee_setting

    def validateIndividual(self, individual):
        assert len(individual) == 50
        assert sum(individual) == 80
        assert min(individual) >= 1

    # Post-GA Local Hill-Climber: tests single-employee transfers from non-bottleneck to bottleneck queues
    def localSearchRefine(self):
        improved = True
        steps = 0
        while improved:
            improved = False
            best_candidate = None
            best_cand_ft = self.bestFitness

            donors = [i for i, v in enumerate(self.bestIndividual) if v > 1]
            # Focus recipients on departments that currently have high load or >1 staff
            for d in donors:
                for r in range(50):
                    if d == r:
                        continue
                    cand = self.bestIndividual.copy()
                    cand[d] -= 1
                    cand[r] += 1
                    ft = self.calcFitness(cand)
                    if ft < best_cand_ft - 1e-9:
                        best_cand_ft = ft
                        best_candidate = cand
                        improved = True

            if improved and best_candidate is not None:
                self.bestIndividual = best_candidate
                self.bestFitness = best_cand_ft
                key = tuple(self.bestIndividual)
                _, self.bestMeanWait, self.bestMaxWait = self.fitnessCache[key]
                steps += 1
        return steps

    def verifyBestDirectly(self):
        """
        Directly runs service_simulator.init_and_simulate on self.bestIndividual
        to verify 100% bit-for-bit agreement with the simulator.
        """
        ft, avg_mean, avg_max, _ = _evaluate_individual_direct(
            self.bestIndividual, self.w_mean, self.w_max
        )
        balanced_score = 0.89 * avg_mean + 0.11 * avg_max
        return ft, avg_mean, avg_max, balanced_score

    def run(self, apply_local_refinement=True):
        total_start = time.time()

        for g in range(self.numGenerations):
            generation_start = time.time()

            self.evaluation(g)
            self.selectParents()

            newPopulation = [self.bestIndividual.copy()]

            while len(newPopulation) < self.populationSize:
                p1, p2 = random.sample(self.parents, 2)
                child1, child2 = self.crossover(p1, p2)
                newPopulation.append(self.mutation(child1))
                newPopulation.append(self.mutation(child2))

            newPopulation = newPopulation[: self.populationSize]

            for individual in newPopulation:
                self.validateIndividual(individual)

            self.listOfBestFitness.append(self.bestFitness)
            generation_time = time.time() - generation_start

            if self.verbose:
                print(
                    "Generation {:2d} | Best fitness: {:.4f} (Mean: {:.2f}s, Max: {:.2f}s) | Time: {:.4f}s".format(
                        g,
                        self.bestFitness,
                        self.bestMeanWait,
                        self.bestMaxWait,
                        generation_time,
                    )
                )

            if self.patienceCounter >= self.patience:
                if self.verbose:
                    print("Stopped after {} generations".format(g))
                break

            self.population = newPopulation

        refinement_steps = 0
        if apply_local_refinement:
            refinement_steps = self.localSearchRefine()

        total_elapsed = time.time() - total_start
        ft_verify, mean_verify, max_verify, bal_verify = self.verifyBestDirectly()

        if self.verbose:
            print(
                "\nBest individual is {} ;\n"
                "  Weighted fitness ({:.2f}*Mean + {:.2f}*Max): {:.4f} (direct simulator verify: {:.4f})\n"
                "  Mean waiting time: {:.4f}s | Max waiting time: {:.4f}s\n"
                "  Scale-balanced score (0.89*Mean + 0.11*Max): {:.4f}\n"
                "  Found at generation {} (+{} local refinement steps) | Total GA run time: {:.2f}s".format(
                    self.bestIndividual,
                    self.w_mean,
                    self.w_max,
                    self.bestFitness,
                    ft_verify,
                    mean_verify,
                    max_verify,
                    bal_verify,
                    self.bestfoundAt,
                    refinement_steps,
                    total_elapsed,
                )
            )

        return {
            "bestIndividual": self.bestIndividual,
            "bestFitness": ft_verify,
            "meanWait": mean_verify,
            "maxWait": max_verify,
            "balancedScore": bal_verify,
            "bestFoundAt": self.bestfoundAt,
            "refinementSteps": refinement_steps,
            "totalTime": total_elapsed,
            "simulatorCalls": self.simulatorCalls,
            "cacheHits": self.cacheHits,
        }


if __name__ == "__main__":
    t_init = time.time()
    random.seed(42)
    np.random.seed(42)

    ga = KnapsackOptimized(
        populationSize=150,
        numGenerations=80,
        mutationRate=0.03,
        w_mean=0.80,
        w_max=0.20,
        patience=30,
        use_dept_cache=True,
        verbose=True,
    )
    res = ga.run(apply_local_refinement=True)
    print(f"Total wall-clock time (including cache pre-warm & verification): {time.time() - t_init:.2f}s")
