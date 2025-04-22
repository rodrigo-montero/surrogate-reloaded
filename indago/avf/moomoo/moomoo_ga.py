from datetime import datetime
import random
from typing import Callable, List

import numpy as np
import autograd.numpy as anp
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from indago.avf.moomoo.chromosome import Chromosome
# import indago.avf.moomoo.mutations as mutation
import indago.avf.moomoo.crossovers as crossover
import indago.avf.moomoo.sampling as sampling
from indago.avf.moomoo.termination import SingleObjectiveSpaceTermination
import indago.avf.moomoo.selection as selection
import indago.avf.moomoo.crossmutation as mutation

# NOTE: Replace this with a better stopping criterian for Pymoo
from indago.avf.ga.stopping_criterion import StoppingCriterion

from pymoo.problems import get_problem

from indago.utils.torch_utils import to_numpy

from indago.avf.avf_policy import AvfPolicy
from indago.avf.dataset import Dataset

from log import Log


class PMGAProblem(Problem):
    
    def __init__(
        self,
        fitness_fn,
        avf_test_policy: str,
        chromosome_factory: Callable[[], Chromosome],
        preprocessed_dataset: Dataset,
        initial_population: List,            # Similar to INDAGO GA- Provide a small sample of existing failures from training to help boost the population
        # saved_environments: List,           # The list of saved environments for diversity fitness check
        trained_avf_policy: AvfPolicy,
        avf_train_policy: str,
        regression: bool,
        crossover_rate: float,
        minimize: bool = True,
        # minimize_attribution: bool = False
    ):
        super().__init__(
            n_var=1,
            n_obj=1
        )
        self.minimize = minimize
        # self.minimize_attribution = minimize_attribution
        self.population: List[Chromosome] = initial_population
        self.fitness_values = []
        self.fitness_fn = fitness_fn
        self.fitness_evaluations = 0
        self.chromosome_factory = chromosome_factory
        self.avf_test_policy = avf_test_policy

        self.preprocessed_dataset = preprocessed_dataset
        self.initial_population = initial_population
        # self.saved_environments = saved_environments
        self.avf_train_policy = avf_train_policy
        self.trained_avf_policy = trained_avf_policy
        self.regression = regression

        self.crossover_rate = crossover_rate

        self.failure_env_configurations_pop = []

        self.logger = Log("PMGAProblem")

    """
    _evaluate - Simple at the moment, gets the latest predictions and stores them.
    return - Returns "F" - top value and "G" Array of all predicted values.
    """
    def _evaluate(self, children, out, *args, **kwargs):
        objectives = []
        for child in children:
            chromosome = child[0]
            if chromosome.fitness is None:
                chromosome.compute_fitness(fitness_fn=self.fitness_fn)
            objectives.append(chromosome.fitness)

        out["F"] = np.array(objectives)

class PMGA():

    def __init__(
            self,
            population_size: int,
            generations: int,
            avf_test_policy: str,
            stopping_criterion_factory: Callable[[], StoppingCriterion],
            chromosome_factory: Callable[[], Chromosome],
            preprocessed_dataset: Dataset,
            trained_avf_policy: AvfPolicy,
            fitness_fn,
            avf_train_policy: str,
            regression: bool,
            crossover_rate: float,
            minimize: bool = True,
            # minimize_attribution: bool = False
        ):
        self.population_size = population_size
        self.initial_population = []
        self.generations = generations
        self.minimize = minimize
        # self.minimize_attribution = minimize_attribution
        self.fitness_fn = fitness_fn
        self.fitness_evaluations = 0
        self.chromosome_factory = chromosome_factory
        self.stopping_criterion_factory = stopping_criterion_factory
        self.avf_test_policy = avf_test_policy

        self.preprocessed_dataset = preprocessed_dataset
        self.avf_train_policy = avf_train_policy
        self.trained_avf_policy = trained_avf_policy
        self.regression = regression

        self.crossover_rate = crossover_rate

        self.failure_env_configurations_pop = []

        self.logger = Log("Pymoo Genetic Algorithm")
    
    def run_mutations(
            self,
            num_generations=50,
            only_best=True,
            fitness_fn=None,
            budget=None,
        ):

        algorithm = GA(pop_size=self.population_size,
                          sampling=sampling.InitialImageSampling(),
                          crossover=crossover.SinglePointFixedCrossover(),
                          mutation=mutation.CombinedCrossoverMutation(),
                          eliminate_duplicates=False
                        )

        # Define the problem
        problem = PMGAProblem(
            fitness_fn=fitness_fn,
            avf_test_policy=self.avf_test_policy,
            chromosome_factory=self.chromosome_factory,
            preprocessed_dataset=self.preprocessed_dataset,
            initial_population=self.initial_population,
            trained_avf_policy=self.trained_avf_policy,
            avf_train_policy=self.avf_train_policy,
            regression=self.regression,
            crossover_rate=self.crossover_rate,
            minimize=True,
            # minimize_attribution=self.minimize_attribution
        )

        # Generate a random seed
        random.seed(int(datetime.now().timestamp()))
        
        try:
            # Run the optimization
            res = minimize(
                problem,
                algorithm,
                termination=SingleObjectiveSpaceTermination(tol=0.0005, n_last=5, n_gen=num_generations),
                # seed=42, # seed=seed if seed else random.randint(0, 999999999),
                seed=random.randint(0, 999999999),
                verbose=True
            )
        except Exception as e:
            print(e)

        # Extract the best solution
        best_solution = res.X  # Decision variable(s) of the best solution
        best_f = res.F  # Objective value of the best solution

        print("Best Solution:", best_solution)
        print("Best Objective Value:", best_f)

        return best_solution[0]
