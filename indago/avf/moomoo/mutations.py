# Contains all the different mutations for our program.
import numpy as np
from pymoo.core.mutation import Mutation

import torch
from captum.attr import Saliency
from indago.utils.torch_utils import to_numpy

class CustomMutation(Mutation):
    def _do(self, problem, mutations, **kwargs):
        new_mutants = np.full((len(mutations), 1), None, dtype=object)

        for i in range(len(mutations)):
            chromosome = mutations[i][0]

            if chromosome is None:
                print(chromosome)
            
            env_config_transformed = problem.preprocessed_dataset.transform_env_configuration(
                        env_configuration=chromosome.env_config, policy=problem.avf_train_policy,
                    )
            saliency = Saliency(forward_func=problem.trained_avf_policy.get_model().forward)
            env_config_tensor = torch.tensor(env_config_transformed, dtype=torch.float32, requires_grad=True)
            env_config_tensor = env_config_tensor.view(1, -1)
            if not problem.regression:
                attributions = saliency.attribute(
                    env_config_tensor, abs=False, target=1
                )
            else:
                attributions = saliency.attribute(env_config_tensor, abs=False)

            mapping = problem.preprocessed_dataset.get_mapping_transformed(
                env_configuration=chromosome.env_config
            )

            attributions = to_numpy(attributions).squeeze()

            mutation = chromosome.mutate_hot(
                    attributions=attributions,
                    mapping=mapping
                )

            if mutation:
                new_mutants[i, 0] = mutation
            else:
                new_mutants[i, 0] = chromosome

        return new_mutants
