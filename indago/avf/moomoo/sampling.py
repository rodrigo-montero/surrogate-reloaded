from pymoo.core.sampling import Sampling
import numpy as np
# import de_mutations as mutation
from typing import Callable, List, Optional, Tuple
from indago.avf.moomoo.chromosome import Chromosome
# Environment specific imports
from indago.avf.env_configuration import EnvConfiguration
from indago.envs.park.parking_env_configuration import ParkingEnvConfiguration
from indago.avf.factories import (
    get_avf_policy,
    get_chromosome_factory,
    get_stopping_criterion_factory,
)
from indago.config import (
    DONKEY_ENV_NAME,
    ENV_NAMES,
    HUMANOID_ENV_NAME,
    PARK_ENV_NAME,
)

# InitialImageSampling - Generates our base samples. Uses the given image_array to generate a new seed.
# Modifies one pixel in this seed and adds it to our sample array along with the initial prediction value.
class InitialImageSampling(Sampling):
    
    def _do(self, problem, n_samples, **kwargs):
        population = np.full((n_samples, 1), None, dtype=object)

        # If we generated an initial population of failures
        # Then we want to first append these
        for i in range(len(problem.initial_population)):
            population[i, 0] = problem.initial_population[i]
        # Generate the rest of our population
        for i in range(len(problem.initial_population), n_samples):
            env_config = self.generate_random_env_configuration(problem)
            population[i, 0] = Chromosome(env_config=env_config)

        return population
    
    def generate_random_env_configuration(self, problem) -> EnvConfiguration:
        env_config = ParkingEnvConfiguration().generate_configuration()

        if problem.avf_test_policy == "random":
            problem.current_env_config = env_config
            problem.logger.info(
                "Env configuration: {}".format(self.current_env_config.get_str())
            )

        return env_config
    
if __name__ == "__main__":
    sampling = InitialImageSampling()

    sampling.generate_population(20, )