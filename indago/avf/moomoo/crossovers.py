from indago.avf.ga.chromsome import Chromosome
from pymoo.core.crossover import Crossover
from indago.utils import randomness
import copy
import numpy as np


def single_point_fixed_crossover(c1: Chromosome, c2: Chromosome, trials: int = 20) -> bool:
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

# Currently commented out so we can just pass through
# Crossover is applied during the mutation code.
class SinglePointFixedCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, children, **kwargs):
        # # The input of has the following shape (n_parents, n_matings, n_var)
        # _, n_matings, n_var = children.shape

        # if randomness.get_random_float(low=0, high=1) < problem.crossover_rate:
        #     return children
        
        # # The output with the shape (n_offsprings, n_matings, n_var)
        # # Because there the number of parents and offsprings are equal it keeps the shape of X
        # crossovers = np.full_like(children, None, dtype=object)

        # if len(children[0]) == 1:
        #     return children

        # # for each mating provided
        # for k in range(n_matings):
        #     ori_c1, ori_c2 = children[0, k, 0], children[1, k, 0]

        #     # prepare the offsprings by copying the change matrix
        #     c1 = copy.deepcopy(ori_c1)
        #     c2 = copy.deepcopy(ori_c2)

        #     if c1.length == c2.length:
        #         # Set the range to -1 so that if we crossover goal_idx, we also crossover parking positions.
        #         # For now we don't want to crossover parking positions alone as this breaks the goal_idx in places (goal_idx could be in parking_locations)
        #         #TODO: Perhaps expand this later so we separately crossover the parking positions array.
        #         point = randomness.get_random_int(low=0, high=c1.length - 1)
        #         mixed_chromosome_1 = c1.crossover(c=c2, pos1=point, pos2=point)
        #         mixed_chromosome_2 = c2.crossover(c=c1, pos1=point, pos2=point)

        #         crossovers[0, k, 0] = mixed_chromosome_1 if mixed_chromosome_1 is not None else c1
        #         crossovers[1, k, 0] = mixed_chromosome_2 if mixed_chromosome_2 is not None else c2

        return children