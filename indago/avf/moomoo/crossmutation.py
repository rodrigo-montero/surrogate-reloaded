from pymoo.core.mutation import Mutation
from typing import List
import numpy as np
import copy

from indago.avf.moomoo.chromosome import Chromosome
from indago.utils import randomness
import torch
from captum.attr import Saliency
from indago.avf.config import ELITISM_PERCENTAGE
from indago.utils.torch_utils import to_numpy

def fitness_sorting(chromosome: Chromosome):
    return chromosome.fitness


def keep_offspring(
    parent_1: Chromosome, parent_2: Chromosome, offspring_1: Chromosome, offspring_2: Chromosome, minimize: bool = True
) -> bool:

    assert parent_1.fitness is not None, "First parent fitness not computed"
    assert parent_2.fitness is not None, "Second parent fitness not computed"
    assert offspring_1.fitness is not None, "First offspring fitness not computed"
    assert offspring_2.fitness is not None, "Second offspring fitness not computed"

    if minimize:
        return (
            compare_best_offspring_to_best_parent(
                parent_1=parent_1, parent_2=parent_2, offspring_1=offspring_1, offspring_2=offspring_2, minimize=minimize
            )
            <= 0
        )
    return (
        compare_best_offspring_to_best_parent(
            parent_1=parent_1, parent_2=parent_2, offspring_1=offspring_1, offspring_2=offspring_2, minimize=minimize
        )
        >= 0
    )


def compare_best_offspring_to_best_parent(
    parent_1: Chromosome, parent_2: Chromosome, offspring_1: Chromosome, offspring_2: Chromosome, minimize: bool = True
) -> int:

    best_parent = get_best(c1=parent_1, c2=parent_2, minimize=minimize)
    best_offspring = get_best(c1=offspring_1, c2=offspring_2, minimize=minimize)

    if best_offspring.fitness > best_parent.fitness:
        return 1
    if best_offspring.fitness < best_parent.fitness:
        return -1
    return 0


def get_best(c1: Chromosome, c2: Chromosome, minimize: bool = True) -> Chromosome:
    if minimize:
        return c1 if c1.fitness < c2.fitness else c2
    return c1 if c1.fitness < c2.fitness else c2

class CombinedCrossoverMutation(Mutation):
    def _do(self, problem, mutations, **kwargs):

        new_generation: List[Chromosome] = self.elitism(problem.population)

        while len(new_generation) < len(problem.population):
            parent_1 = self.roulette_wheel_selection(population=problem.population, minimize=problem.minimize)
            parent_2 = self.roulette_wheel_selection(population=problem.population, minimize=problem.minimize)

            offspring_1 = copy.deepcopy(parent_1)
            offspring_2 = copy.deepcopy(parent_2)
            # if offspring_1.fitness is None:
            #     self.fitness_evaluations += 1
            # if offspring_2.fitness is None:
            #     self.fitness_evaluations += 1
            offspring_1.compute_fitness(fitness_fn=problem.fitness_fn)
            offspring_2.compute_fitness(fitness_fn=problem.fitness_fn)

            if randomness.get_random_float(low=0, high=1) < problem.crossover_rate:

                crossover = self.single_point_fixed_crossover(c1=offspring_1, c2=offspring_2)
                if crossover:
                    # if offspring_1.fitness is None:
                    #     self.fitness_evaluations += 1
                    # if offspring_2.fitness is None:
                    #     self.fitness_evaluations += 1
                    offspring_1.compute_fitness(fitness_fn=problem.fitness_fn)
                    offspring_2.compute_fitness(fitness_fn=problem.fitness_fn)

            if "saliency" in problem.avf_test_policy:
                mutated_offsprings = []
                for offspring in [offspring_1, offspring_2]:

                    env_config_transformed = problem.preprocessed_dataset.transform_env_configuration(
                        env_configuration=offspring.env_config, policy=problem.avf_train_policy,
                    )
                    saliency = Saliency(forward_func=problem.trained_avf_policy.get_model().forward)
                    env_config_tensor = torch.tensor(env_config_transformed, dtype=torch.float32, requires_grad=True)
                    env_config_tensor = env_config_tensor.view(1, -1)
                    if not problem.regression:
                        attributions = saliency.attribute(env_config_tensor, abs=False, target=1)
                    else:
                        attributions = saliency.attribute(env_config_tensor, abs=False)
                    mapping = problem.preprocessed_dataset.get_mapping_transformed(env_configuration=offspring.env_config)
                    attributions = to_numpy(attributions).squeeze()
                    mutated_offsprings.append(offspring.mutate_hot(attributions=attributions, mapping=mapping))

                mutated_offspring_1 = mutated_offsprings[0]
                mutated_offspring_2 = mutated_offsprings[1]
            else:
                mutated_offspring_1 = offspring_1.mutate()
                mutated_offspring_2 = offspring_2.mutate()

            if mutated_offspring_1 is not None:
                offspring_1.set_env_config(env_config=mutated_offspring_1.env_config)
                # if offspring_1.fitness is None:
                #     self.fitness_evaluations += 1
                offspring_1.compute_fitness(fitness_fn=problem.fitness_fn)
            if mutated_offspring_2 is not None:
                offspring_2.set_env_config(env_config=mutated_offspring_2.env_config)
                # if offspring_2.fitness is None:
                #     self.fitness_evaluations += 1
                offspring_2.compute_fitness(fitness_fn=problem.fitness_fn)

            # The two offsprings replace the parents if and only if one of the
            # offspring is not worse than the best parent.
            if keep_offspring(
                parent_1=parent_1, parent_2=parent_2, offspring_1=offspring_1, offspring_2=offspring_2, minimize=problem.minimize
            ):
                new_generation.append(offspring_1)
                new_generation.append(offspring_2)
            else:
                new_generation.append(parent_1)
                new_generation.append(parent_2)

        problem.population = copy.deepcopy(new_generation)

        new_mutants = np.full((len(new_generation), 1), None, dtype=object)

        for i in range(len(new_generation)):
            new_mutants[i, 0] = new_generation[i]

        return new_mutants

    def sort_population(self, population) -> None:
        population.sort(key=fitness_sorting)

    def elitism(self, population) -> List[Chromosome]:
        self.sort_population(population)
        new_population: List[Chromosome] = []
        for i in range(int(len(population) * ELITISM_PERCENTAGE / 100)):
            new_population.append(population[i])
        return new_population

    def roulette_wheel_selection(self, population: List[Chromosome], minimize: bool = True) -> Chromosome:
        sum_of_fitnesses = (
            sum([c.fitness for c in population]) if not minimize else sum([(1 / (c.fitness + 1)) for c in population])
        )

        if sum_of_fitnesses == 0.0:
            return population[randomness.get_random_int(low=0, high=len(population) - 1)]

        rnd = randomness.get_random_float(low=0, high=1) * sum_of_fitnesses

        for i in range(len(population)):
            fitness = population[i].fitness

            if minimize:
                fitness = 1 / (fitness + 1)

            if fitness >= rnd:
                return population[i]
            rnd = rnd - fitness

        return population[randomness.get_random_int(low=0, high=len(population))]
    
    def single_point_fixed_crossover(self, c1: Chromosome, c2: Chromosome, trials: int = 20) -> bool:
        assert c1.length == c2.length, "The length of the two chromosome must be the same: {} != {}".format(c1.length, c2.length)

        if c1.length < 2:
            return False

        for _ in range(trials):
            point = randomness.get_random_int(low=0, high=c1.length)
            mixed_chromosome_1 = c1.crossover(c=c2, pos1=point, pos2=point)
            mixed_chromosome_2 = c2.crossover(c=c1, pos1=point, pos2=point)
            if mixed_chromosome_1 is not None and mixed_chromosome_2 is not None:
                c1.set_env_config(env_config=mixed_chromosome_1.env_config)
                c2.set_env_config(env_config=mixed_chromosome_2.env_config)
                return True

        return False