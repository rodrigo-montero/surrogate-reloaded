from typing import Callable, Union

from indago.avf.avf_mlp import AvfMlpPolicy
from indago.avf.avf_cnn import AvfCnnPolicy
from indago.avf.avf_bnn import AvfBnnPolicy
from indago.avf.avf_policy import AvfPolicy
from indago.avf.env_configuration import EnvConfiguration
from indago.avf.ga.chromsome import Chromosome
from indago.avf.ga.stopping_criterion import StoppingCriterion
from indago.avf.ga.stopping_criterion_impl import StoppingCriterionImpl


def get_avf_policy(
    env_name: str, policy: str, input_size: int, regression: bool = False, layers: int = 4, learning_rate: float = 3e-4, hidden_layer_size: int = 64, model: str = ""           # NEW (bnn)
) -> AvfPolicy:
    if policy == "mlp":
        return AvfMlpPolicy(
            env_name=env_name, input_size=input_size, regression=regression, layers=layers, learning_rate=learning_rate
        )
    elif policy == "cnn":
        return AvfCnnPolicy(
            env_name=env_name, input_size=input_size, regression=regression, layers=layers, learning_rate=learning_rate
        )
    elif policy == "bnn":                                                                                                       # NEW (bnn)
        return AvfBnnPolicy(
            env_name=env_name, input_size=input_size, regression=regression, layers=layers, learning_rate=learning_rate, hidden_layer_size=hidden_layer_size, bnn_type=model
        )
    raise NotImplementedError("Unknown policy: {}".format(policy))


def get_chromosome_factory(generate_env_config_fn: Callable[[bool], EnvConfiguration]) -> Callable[[], Chromosome]:
    def func() -> Chromosome:
        env_config = generate_env_config_fn()
        return Chromosome(env_config=env_config)

    return func


def get_stopping_criterion_factory(target_fitness: float = 0.0) -> Callable[[], StoppingCriterion]:
    def func() -> StoppingCriterion:
        return StoppingCriterionImpl(target_fitness=target_fitness)

    return func
