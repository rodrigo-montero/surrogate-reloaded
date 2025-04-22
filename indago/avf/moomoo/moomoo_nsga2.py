from datetime import datetime
import random
from typing import Callable, List

import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt
from math import sqrt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from indago.avf.moomoo.chromosome import Chromosome
import indago.avf.moomoo.crossmutation as mutation
import indago.avf.moomoo.crossovers as crossover
import indago.avf.moomoo.sampling as sampling
import indago.avf.moomoo.diversity as Diversity
from indago.avf.moomoo.termination import MultiObjectiveHypervolumeTermination

# NOTE: Replace this with a better stopping criterian for Pymoo
from indago.avf.ga.stopping_criterion import StoppingCriterion

from pymoo.problems import get_problem

from indago.utils.torch_utils import to_numpy

from indago.avf.avf_policy import AvfPolicy
from indago.avf.dataset import Dataset

from concurrent.futures import ThreadPoolExecutor, as_completed

from log import Log

class PMNSGA2Problem(Problem):
    
    def __init__(
        self,
        fitness_fn,
        avf_test_policy: str,
        chromosome_factory: Callable[[], Chromosome],
        preprocessed_dataset: Dataset,
        initial_population: List,           # Similar to INDAGO GA- Provide a small sample of existing failures from training to help boost the population
        saved_environments: List,           # The list of saved environments for diversity fitness check
        trained_avf_policy: AvfPolicy,
        avf_train_policy: str,
        regression: bool,
        crossover_rate: float,
        minimize: bool = True,
        # minimize_attribution: bool = False
    ):
        super().__init__(
            n_var=1,
            n_obj=2
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
        self.process_name = type(self.preprocessed_dataset).__name__
        self.initial_population = initial_population
        self.saved_environments = saved_environments
        self.avf_train_policy = avf_train_policy
        self.trained_avf_policy = trained_avf_policy
        self.regression = regression

        self.crossover_rate = crossover_rate

        self.failure_env_configurations_pop = []

        self.logger = Log("PMNSGA2Problem")

    def _evaluate(self, children, out, *args, **kwargs):
    
        f1 = []
        f2 = []

        # if self.process_name == 'ParkingDataset':
        #     diversity = Diversity.ParkingDiversity()
            # args = (diversity.evaluate_child, self.fitness_fn, self.saved_environments, diversity.weights, diversity.parked_weights)
        if self.process_name == 'ParkingDataset':
            diversity = Diversity.ParkingDiversitySimple()
            args = (diversity.evaluate_child, self.fitness_fn, self.preprocessed_dataset, self.saved_environments)
        elif self.process_name == 'DonkeyDataset':
            diversity = Diversity.DonkeyDiversitySimple()
            args = (diversity.evaluate_child, self.fitness_fn, self.preprocessed_dataset, self.saved_environments)
        elif self.process_name == 'HumanoidDataset':
            diversity = Diversity.HumanoidDiversitySimple()
            args = (diversity.evaluate_child, self.fitness_fn, self.preprocessed_dataset, self.saved_environments)

        # Attempt to speed up diversity by pooling evaluation code
        # As saved environments grows, so does the time of the diversity check
        with ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            future_to_child = {
                executor.submit(*args, child): i for i, child in enumerate(children)
            }

            # Collect results while maintaining order
            results = sorted(
                ((future_to_child[future], future.result()) for future in as_completed(future_to_child)),
                key=lambda x: x[0],  # Sort by original index
            )

            # Unpack results into f1 and f2
            for _, (fitness, diversity_score) in results:
                f1.append(fitness)
                f2.append(diversity_score)

        # Ensure f2 is initialized to zeros if no saved environments
        if not self.saved_environments:
            f2 = np.zeros(len(children)).tolist()

        # Calculate diversity of internal population
        out["F"] = anp.column_stack([f1, f2])

class PMNSGA2():

    def __init__(
            self,
            population_size: int,
            saved_environments: List,
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
        self.saved_environments = saved_environments
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

        self.logger = Log("MultiObjAlgorithm")
    
    def run_mutations(
            self,
            num_generations=50,
            only_best=True,
            fitness_fn=None,
            budget=None,
        ):

        algorithm = NSGA2(pop_size=self.population_size,
                          sampling=sampling.InitialImageSampling(),
                          crossover=crossover.SinglePointFixedCrossover(),
                          mutation=mutation.CombinedCrossoverMutation(),
                          eliminate_duplicates=False
                        )

        # Define the problem
        problem = PMNSGA2Problem(
            fitness_fn=fitness_fn,
            avf_test_policy=self.avf_test_policy,
            chromosome_factory=self.chromosome_factory,
            preprocessed_dataset=self.preprocessed_dataset,
            initial_population=self.initial_population,
            saved_environments=self.saved_environments,
            trained_avf_policy=self.trained_avf_policy,
            avf_train_policy=self.avf_train_policy,
            regression=self.regression,
            crossover_rate=self.crossover_rate,
            minimize=True,
        )

        # Generate a random seed
        random.seed(int(datetime.now().timestamp()))
        
        # Define termination with a reference point
        ref_point = np.array([1.1, 20.1])  # Adjust based on problem scale

        try:
            # Run the optimization
            res = minimize(
                problem,
                algorithm,
                # ("n_gen", 50),
                # seed=42, # seed=seed if seed else random.randint(0, 999999999),
                termination=MultiObjectiveHypervolumeTermination(ref_point=ref_point, tol=5e-6, n_last=10, n_gen=50),
                seed=random.randint(0, 999999999),
                verbose=True
            )
        except Exception as e:
            print(e)

        # Extract all solutions and their objectives
        solutions = res.X  # Decision variables (candidate solutions)
        objectives = res.F  # Objective values (multi-objective results)

        # Find the index of the solution with the best first objective (minimum value)
        best_index = min(enumerate(objectives), key=lambda x: x[1][0])[0]  # x[1][0] is the first objective

        # Get the corresponding solution and objective values
        best_solution = solutions[best_index]
        best_f = objectives[best_index]

        print("Best Solution (Optimizing First Objective):", best_solution)
        print("Best Objective Value (First Objective):", best_f)
        
        plt.scatter(res.F[:, 0], res.F[:, 1], c='blue', label="Pareto Front", edgecolor='black')

        # Labels and title
        plt.xlabel("Failure Probability")
        plt.ylabel("Diversity")
        plt.title("Pareto Front")
        plt.legend()

        # Show the plot
        # plt.show()

        return best_solution[0]