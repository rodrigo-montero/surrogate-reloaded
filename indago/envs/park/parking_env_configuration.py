import copy
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from indago.avf.env_configuration import EnvConfiguration, EnvMutations
from indago.config import PARAM_SEPARATOR
from indago.envs.park.parking_env import ParkingEnv
from indago.utils.randomness import get_randint_sample, get_random_float, get_random_int

MAX_NUM_LANES = 10


class ParkingEnvConfiguration(EnvConfiguration):
    def __init__(
        self,
        num_lanes: int = MAX_NUM_LANES,
        goal_lane_idx: int = -1,
        heading_ego: float = 0.0,
        parked_vehicles_lane_indices: List[int] = None,
        position_ego: Tuple[float, float] = (20.0, 0.0),
    ):
        super().__init__()
        self.num_lanes = num_lanes
        self.goal_lane_idx = goal_lane_idx
        self.heading_ego = heading_ego
        self.parked_vehicles_lane_indices = parked_vehicles_lane_indices if parked_vehicles_lane_indices is not None else []
        self.position_ego = position_ego

        self.key_names = ["num_lanes", "goal_lane_idx", "heading_ego", "parked_vehicles_lane_indices", "position_ego"]

        self.update_implementation(
            num_lanes=self.num_lanes,
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            parked_vehicles_lane_indices=list(set(sorted(map(lambda num: int(num), self.parked_vehicles_lane_indices)))),
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

    def generate_configuration(self) -> "EnvConfiguration":

        while not self._is_valid():
            # self.num_lanes = int(np.random.randint(low=1, high=20, dtype=np.int32))
            # self.goal_lane_idx = int(np.random.randint(low=0, high=2 * self.num_lanes - 1, dtype=np.int32))
            self.goal_lane_idx = get_random_int(low=0, high=2 * self.num_lanes - 1)
            self.heading_ego = round(get_random_float(), 2)
            # self.parked_vehicles_lane_indices = []
            self.position_ego = (
                round(float(get_random_float(low=-10, high=10)), 2),
                round(float(get_random_float(low=-5, high=5)), 2),
            )
            if self.num_lanes > 1:
                self.parked_vehicles_lane_indices = get_randint_sample(
                    low=0, high=2 * self.num_lanes, count=get_random_int(low=0, high=2 * self.num_lanes)
                )
            else:
                self.parked_vehicles_lane_indices = []

        self.update_implementation(
            num_lanes=self.num_lanes,
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            parked_vehicles_lane_indices=list(set(sorted(map(lambda num: int(num), self.parked_vehicles_lane_indices)))),
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

        return self

    def _is_valid(self) -> bool:

        if len(self.parked_vehicles_lane_indices) >= 2 * MAX_NUM_LANES:
            # print('Invalid len parked vehicles: {}'.format(len(self.parked_vehicles_lane_indices)))
            return False

        for idx in range(len(self.parked_vehicles_lane_indices)):
            if self.parked_vehicles_lane_indices[idx] == self.goal_lane_idx:
                # print('Invalid parked vehicle idx: {} - {}'.format(self.parked_vehicles_lane_indices[idx], self.goal_lane_idx))
                return False

        for parked_vehicles_lane_index in self.parked_vehicles_lane_indices:
            if parked_vehicles_lane_index < 0 or parked_vehicles_lane_index > 2 * MAX_NUM_LANES - 1:
                # print('Invalid parked vehicle idx: {}'.format(parked_vehicles_lane_index))
                return False

        if self.num_lanes < 1 or self.num_lanes > MAX_NUM_LANES:
            # print('Invalid lanes: {}'.format(self.num_lanes))
            return False

        if self.goal_lane_idx < 0 or self.goal_lane_idx > 2 * MAX_NUM_LANES - 1:
            return False

        if round(self.heading_ego, 2) < 0.00 or round(self.heading_ego, 2) > 1.00:
            # print('Invalid heading ego: {}'.format(self.heading_ego))
            return False

        if round(self.position_ego[0], 2) < -10.00 or round(self.position_ego[0], 2) > 10.00:
            # print('Invalid position ego 0: {}'.format(self.position_ego[0]))
            return False

        if round(self.position_ego[1], 2) < -5.00 or round(self.position_ego[1], 2) > 5.00:
            # print('Invalid position ego 1: {}'.format(self.position_ego[1]))
            return False

        if len(set(self.parked_vehicles_lane_indices)) < len(self.parked_vehicles_lane_indices):
            # duplicates
            return False

        return True

    def get_image(self) -> np.ndarray:
        # FIXME: maybe there is a better way of doing it, instead of instantiating the environment every time
        env = ParkingEnv()
        env.num_lanes = self.num_lanes
        env.goal_lane_idx = self.goal_lane_idx
        env.heading_ego = self.heading_ego
        env.parked_vehicles_lane_indices = self.parked_vehicles_lane_indices
        env.position_ego = self.position_ego

        _ = env.reset()
        image = env.render("rgb_array")
        buffered = BytesIO()
        pil_image = Image.fromarray(image)
        pil_image.save(buffered, optimize=True, format="PNG", quality=95)

        env.close()
        return np.asarray(pil_image)

    def get_str(self) -> str:
        return "{}{}{}{}{}{}{}{}{}".format(
            self.num_lanes,
            PARAM_SEPARATOR,
            self.goal_lane_idx,
            PARAM_SEPARATOR,
            self.heading_ego,
            PARAM_SEPARATOR,
            sorted(self.parked_vehicles_lane_indices),
            PARAM_SEPARATOR,
            (self.position_ego[0], self.position_ego[1]),
        )

    def str_to_config(self, s: str) -> "EnvConfiguration":
        split = s.split(PARAM_SEPARATOR)
        # split = self.split_dash(s=s)
        self.num_lanes = int(split[0])
        self.goal_lane_idx = int(split[1])
        self.heading_ego = float(split[2])
        self.parked_vehicles_lane_indices = []
        split_3 = split[3].replace("[", "").replace("]", "")
        if split_3 != "":
            self.parked_vehicles_lane_indices = [int(num) for num in split_3.split(",")]
        split_4 = split[4].replace("(", "").replace(")", "").replace("[", "").replace("]", "")
        self.position_ego = (float(split_4.split(",")[0]), float(split_4.split(",")[1]))

        self.update_implementation(
            num_lanes=self.num_lanes,
            goal_lane_idx=self.goal_lane_idx,
            heading_ego=self.heading_ego,
            parked_vehicles_lane_indices=list(set(sorted(map(lambda num: int(num), self.parked_vehicles_lane_indices)))),
            position_ego=(self.position_ego[0], self.position_ego[1]),
        )

        return self

    def mutate_num_lanes(
        self, env_config: "ParkingEnvConfiguration", env_mutation: EnvMutations = None, sign: str = "rnd"
    ) -> None:
        if env_mutation is not None or sign == "pos" or sign == "neg":
            if env_mutation == EnvMutations.LEFT or sign == "neg":
                env_config.num_lanes -= get_random_int(low=1, high=2 * self.num_lanes)
            elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                env_config.num_lanes += get_random_int(low=1, high=2 * self.num_lanes)
        else:
            if np.random.random() <= 0.5:
                env_config.num_lanes += get_random_int(low=1, high=2 * self.num_lanes)
            else:
                env_config.num_lanes -= get_random_int(low=1, high=2 * self.num_lanes)

    def mutate_goal_lane_idx(
        self, env_config: "ParkingEnvConfiguration", env_mutation: EnvMutations = None, sign: str = "rnd"
    ) -> None:
        if env_mutation is not None or sign == "pos" or sign == "neg":
            v = env_config.goal_lane_idx
            v_copy = copy.deepcopy(v)
            if (env_mutation == EnvMutations.LEFT or sign == "neg") and v > 1:
                v -= get_random_int(low=1, high=2 * self.num_lanes)
                while v < 0:
                    v = v_copy
                    v -= get_random_int(low=1, high=2 * self.num_lanes)
            elif (env_mutation == EnvMutations.RIGHT or sign == "pos") and v < (2 * self.num_lanes) - 1:
                v += get_random_int(low=1, high=2 * self.num_lanes)
                while v > 2 * self.num_lanes - 1:
                    v = v_copy
                    v += get_random_int(low=1, high=2 * self.num_lanes)
            env_config.goal_lane_idx = v
        else:
            v = env_config.goal_lane_idx
            v_copy = copy.deepcopy(v)
            if get_random_float() <= 0.5 and v < (2 * self.num_lanes) - 1:
                v += get_random_int(low=1, high=2 * self.num_lanes)
                while v > 2 * self.num_lanes - 1:
                    v = v_copy
                    v += get_random_int(low=1, high=2 * self.num_lanes)
            elif v > 1:
                v -= get_random_int(low=1, high=2 * self.num_lanes)
                while v < 0:
                    v = v_copy
                    v -= get_random_int(low=1, high=2 * self.num_lanes)
            env_config.goal_lane_idx = v

    @staticmethod
    def mutate_heading_ego(
        env_config: "ParkingEnvConfiguration", env_mutation: EnvMutations = None, sign: str = "rnd"
    ) -> None:
        if env_mutation is not None or sign == "pos" or sign == "neg":
            if env_mutation == EnvMutations.LEFT or sign == "neg":
                env_config.heading_ego -= float(get_random_float(low=0, high=1))
            elif env_mutation == EnvMutations.RIGHT or sign == "pos":
                env_config.heading_ego += float(get_random_float(low=0, high=1))
        else:
            if get_random_float() <= 0.5:
                env_config.heading_ego += float(get_random_float(low=0, high=1))
            else:
                env_config.heading_ego -= float(get_random_float(low=0, high=1))
        env_config.heading_ego = round(env_config.heading_ego, 2)

    def mutate_parked_vehicles_lane_indices(
        self,
        env_config: "ParkingEnvConfiguration",
        idx_to_mutate: int = None,
        env_mutation: EnvMutations = None,
        sign: str = "str",
    ) -> None:
        if idx_to_mutate is not None:
            if idx_to_mutate in env_config.parked_vehicles_lane_indices:
                if env_mutation is not None or sign == "pos" or sign == "neg":
                    idx = env_config.parked_vehicles_lane_indices.index(idx_to_mutate)
                    v = env_config.parked_vehicles_lane_indices[idx]
                    # the remove option is not possible when considering the sign of the attribution
                    if env_mutation == EnvMutations.REMOVE:
                        env_config.parked_vehicles_lane_indices.remove(idx_to_mutate)
                    elif (env_mutation == EnvMutations.LEFT or sign == "neg") and v > 1:
                        v_copy = copy.deepcopy(v)
                        v -= get_random_int(low=1, high=2 * self.num_lanes)
                        while v < 0:
                            v = v_copy
                            v -= get_random_int(low=1, high=2 * self.num_lanes)
                        env_config.parked_vehicles_lane_indices[idx] = v
                    elif (env_mutation == EnvMutations.RIGHT or sign == "pos") and v < (2 * self.num_lanes) - 1:
                        v_copy = copy.deepcopy(v)
                        v += get_random_int(low=1, high=2 * self.num_lanes)
                        while v > 2 * self.num_lanes - 1:
                            v = v_copy
                            v += get_random_int(low=1, high=2 * self.num_lanes)
                        env_config.parked_vehicles_lane_indices[idx] = v
                else:
                    if get_random_float() <= 0.5:
                        env_config.parked_vehicles_lane_indices.remove(idx_to_mutate)
                    else:
                        idx = env_config.parked_vehicles_lane_indices.index(idx_to_mutate)
                        v = env_config.parked_vehicles_lane_indices[idx]
                        if get_random_float() <= 0.5 and v > 1:
                            v_copy = copy.deepcopy(v)
                            v -= get_random_int(low=1, high=2 * self.num_lanes)
                            while v < 0:
                                v = v_copy
                                v -= get_random_int(low=1, high=2 * self.num_lanes)
                        elif v < (2 * self.num_lanes) - 1:
                            v_copy = copy.deepcopy(v)
                            v += get_random_int(low=1, high=2 * self.num_lanes)
                            while v > 2 * self.num_lanes - 1:
                                v = v_copy
                                v += get_random_int(low=1, high=2 * self.num_lanes)
                        env_config.parked_vehicles_lane_indices[idx] = v
            else:
                env_config.parked_vehicles_lane_indices.append(idx_to_mutate)
        else:
            if get_random_float() <= 0.5:
                if get_random_float() <= 0.5 and len(env_config.parked_vehicles_lane_indices) < 2 * self.num_lanes:
                    # add random indices
                    if 2 * self.num_lanes - len(env_config.parked_vehicles_lane_indices) == 1:
                        num_indices = 1
                    else:
                        num_indices = get_random_int(
                            low=1, high=2 * self.num_lanes - len(env_config.parked_vehicles_lane_indices)
                        )
                    old_num_indices = len(env_config.parked_vehicles_lane_indices)
                    while len(env_config.parked_vehicles_lane_indices) < num_indices + old_num_indices:
                        new_index = get_random_int(low=0, high=2 * self.num_lanes)
                        if new_index not in env_config.parked_vehicles_lane_indices:
                            env_config.parked_vehicles_lane_indices.append(new_index)
                elif len(env_config.parked_vehicles_lane_indices) > 0:
                    # select random indices to remove
                    if len(env_config.parked_vehicles_lane_indices) == 1:
                        indices = [0]
                    else:
                        indices = get_randint_sample(
                            low=1,
                            high=len(env_config.parked_vehicles_lane_indices),
                            count=get_random_int(low=1, high=len(env_config.parked_vehicles_lane_indices)),
                        )
                    for idx in indices:
                        if idx < len(env_config.parked_vehicles_lane_indices):
                            env_config.parked_vehicles_lane_indices.pop(idx)
            elif len(env_config.parked_vehicles_lane_indices) > 0:
                # select random indices to mutate
                if len(env_config.parked_vehicles_lane_indices) == 1:
                    indices = [0]
                else:
                    indices = get_randint_sample(
                        low=1,
                        high=len(env_config.parked_vehicles_lane_indices),
                        count=get_random_int(low=1, high=len(env_config.parked_vehicles_lane_indices)),
                    )
                for idx in indices:
                    v = env_config.parked_vehicles_lane_indices[idx]
                    if get_random_float() <= 0.5 and v > 1:
                        v_copy = copy.deepcopy(v)
                        while v < 0:
                            v = v_copy
                            v -= get_random_int(low=1, high=2 * self.num_lanes)
                    elif v < 2 * self.num_lanes - 1:
                        v_copy = copy.deepcopy(v)
                        while v > 2 * self.num_lanes - 1:
                            v = v_copy
                            v += get_random_int(low=1, high=2 * self.num_lanes)
                    env_config.parked_vehicles_lane_indices[idx] = v

    def mutate_position_ego(self, env_config, idx_to_mutate=None):
        """
        Mutates position_ego using Gaussian mutation for smoother exploration.
        """
        x, y = env_config.position_ego

        if idx_to_mutate == 0:  # Mutate x
            x = self.polynomial_mutation(x, -10, 10)
        elif idx_to_mutate == 1:  # Mutate y
            y = self.polynomial_mutation(y, -5, 5)
        else:  # Mutate both
            x = self.polynomial_mutation(x, -10, 10)
            y = self.polynomial_mutation(y, -5, 5)

        env_config.position_ego = (round(x, 2), round(y, 2))


    def mutate(self) -> Optional["EnvConfiguration"]:
        new_env_config = copy.deepcopy(self)

        # FIXME: change only one parameter with equal probability

        # self.mutate_goal_lane_idx()
        self.mutate_position_ego(env_config=new_env_config)
        self.mutate_parked_vehicles_lane_indices(env_config=new_env_config)
        self.mutate_heading_ego(env_config=new_env_config)
        self.mutate_goal_lane_idx(env_config=new_env_config)

        if new_env_config._is_valid():
            return new_env_config

        return None

    def mutate_hot(self, attributions: np.ndarray, mapping: Dict) -> Optional["EnvConfiguration"]:
        # Select an index to mutate based on attributions (probabilistic sampling).
        idx_to_mutate = random.choices(
            population=list(range(len(attributions))),
            weights=np.abs(attributions),  # Use absolute values for weighting.
            k=1
        )[0]

        # Determine the key to mutate.
        key_to_mutate = self.get_key_to_mutate(idx_to_mutate=idx_to_mutate, mapping=mapping)

        # Determine the direction of mutation (positive, negative, or random).
        sign = "rnd"  # Default to random.
        if attributions[idx_to_mutate] > 0:
            sign = "pos"
        elif attributions[idx_to_mutate] < 0:
            sign = "neg"

        # Create a deep copy of the configuration to mutate.
        new_env_config = copy.deepcopy(self)

        # Apply the mutation based on the selected key.
        if key_to_mutate == "num_lanes":
            self.mutate_num_lanes(env_config=new_env_config, sign=sign)
        elif key_to_mutate == "goal_lane_idx":
            self.mutate_goal_lane_idx(env_config=new_env_config, sign=sign)
        elif key_to_mutate == "heading_ego":
            self.mutate_heading_ego(env_config=new_env_config, sign=sign)
        elif key_to_mutate == "position_ego":
            idx_in_position = mapping[key_to_mutate].index(idx_to_mutate)
            self.mutate_position_ego(env_config=new_env_config, idx_to_mutate=idx_in_position)
        elif key_to_mutate == "parked_vehicles_lane_indices":
            idx_in_parked = mapping[key_to_mutate].index(idx_to_mutate)
            self.mutate_parked_vehicles_lane_indices(env_config=new_env_config, idx_to_mutate=idx_in_parked, sign=sign)
        else:
            raise KeyError(f"Unknown key to mutate: {key_to_mutate}")

        # Validate the new configuration.
        if new_env_config._is_valid():
            return new_env_config

        return None

    def crossover(self, other_env_config: "EnvConfiguration", pos1: int, pos2: int) -> Optional["EnvConfiguration"]:
        # FIXME similar to test suite crossover: implement also test case crossover
        #  (e.g. env.position_ego[0] can be exchanged with other_env.position_ego[0])
        new_env_config_impl = copy.deepcopy(self.impl)
        for i in range(pos1):
            new_env_config_impl[self.key_names[i]] = self.impl[self.key_names[i]]
        for i in range(pos2 + 1, self.get_length()):
            new_env_config_impl[self.key_names[i]] = other_env_config.impl[self.key_names[i]]

        # Extract ego positions from both parents
        position_ego_self = self.impl["position_ego"]
        position_ego_other = other_env_config.impl["position_ego"]

        # Apply SBX to x and y coordinates
        alpha = get_random_float(0, 1)
        new_x = position_ego_self[0] * alpha + (1.0 - alpha) * position_ego_other[0]
        new_y = position_ego_self[1] * alpha + (1.0 - alpha) * position_ego_other[1]

        # Update `position_ego` in the offspring
        new_env_config_impl["position_ego"] = (round(new_x, 2), round(new_y, 2))

        new_env_config = ParkingEnvConfiguration(**new_env_config_impl)

        if new_env_config._is_valid():
            return new_env_config
        return None

    import random

    @staticmethod
    def polynomial_mutation(value, lower_bound, upper_bound, eta=20):
        """
        Perform polynomial mutation on a float value.

        Parameters:
            value (float): The current value to mutate.
            lower_bound (float): The minimum allowable value.
            upper_bound (float): The maximum allowable value.
            eta (int): The distribution index controlling the mutation's magnitude.

        Returns:
            float: The mutated value.
        """
        if lower_bound == upper_bound:  # No mutation possible
            return value

        u = random.random()
        if u < 0.5:
            delta = (2 * u + (1 - 2 * u) * (1 - (value - lower_bound) / (upper_bound - lower_bound)) ** (eta + 1)) ** (
                        1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * ((value - lower_bound) / (upper_bound - lower_bound)) ** (
                        eta + 1)) ** (1 / (eta + 1))

        mutated_value = value + delta * (upper_bound - lower_bound)
        mutated_value = max(lower_bound, min(upper_bound, mutated_value))  # Keep within bounds

        return mutated_value
