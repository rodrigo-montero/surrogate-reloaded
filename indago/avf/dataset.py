from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch as th
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset as ThDataset

from indago.avf.env_configuration import EnvConfiguration
from indago.avf.training_logs import TrainingLogs
from indago.type_aliases import Scaler
from indago.utils.torch_utils import DEVICE
from log import Log


class Data:
    def __init__(
        self, filename: str, training_logs: TrainingLogs,
    ):
        self.filename = filename
        self.training_logs = training_logs
        self.training_progress = self.training_logs.get_training_progress()
        self.exploration_coef = self.training_logs.get_exploration_coefficient()
        self.label = self.training_logs.get_label()
        if self.training_logs.is_regression_value_set():
            self.regression_value = self.training_logs.get_regression_value()

    def __lt__(self, other: "Data"):
        # return np.max(self.reconstruction_losses) < np.max(other.reconstruction_losses)
        return self.training_progress < other.training_progress


# defining the Dataset class
class TorchDataset(ThDataset):
    def __init__(
        self, data: np.ndarray, labels: np.ndarray, regression: bool = False, weight_loss: bool = False,
    ):
        self.data = data
        self.labels = labels
        self.regression = regression
        self.weight_loss = weight_loss
        self.weights = None
        if len(self.labels) > 0:
            if not self.regression:
                if self.weight_loss:
                    hist = dict(Counter(self.labels))
                    n_classes = len(hist)
                    self.weights = list(
                        compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=self.labels)
                    )
                else:
                    hist = dict(Counter(self.labels))
                    self.weights = [np.float32(1.0) for _ in range(len(hist.keys()))]
            else:
                self.weights = [np.float32(1.0) for _ in range(len(self.labels))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not self.regression:
            return (
                th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
                th.tensor(self.labels[index], dtype=th.long).to(DEVICE),
                th.tensor(self.weights, dtype=th.float32).to(DEVICE),
            )

        weights = np.asarray([self.weights[index]]).astype("float32")
        return (
            th.tensor(self.data[index], dtype=th.float32).to(DEVICE),
            th.tensor(self.labels[index], dtype=th.float32).to(DEVICE),
            th.tensor(weights, dtype=th.float32).to(DEVICE),
        )


class Dataset(ABC):
    def __init__(self, policy: str = None):
        self.dataset: List[Data] = []
        self.input_scaler: Scaler = None
        self.output_scaler: Scaler = None
        self.policy = policy

    def add(self, data: Data) -> None:
        self.dataset.append(data)

    def get(self) -> List[Data]:
        return self.dataset

    def get_num_failures(self) -> int:
        assert len(self.dataset) != 0, "Cannot compute num failures since the dataset is empty"
        return sum([1 if data_item.label == 1 else 0 for data_item in self.get()])

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        raise NotImplementedError("Not implemented")

    def get_num_features(self) -> int:
        assert len(self.dataset) > 0, "Not possible to infer num features since there is no data point"
        assert self.policy is not None, "Policy not instantiated"
        if self.policy == "mlp" or self.policy == "cnn" or self.policy == "bnn":                                                # NEW (bnn)
            data_item = self.dataset[0]
            return len(self.transform_mlp(env_configuration=data_item.training_logs.get_config()))
        # TODO: add cnn, i.e. number of channels
        raise NotImplementedError("Unknown policy: {}".format(self.policy))

    def transform_data_item(self, data_item: Data) -> np.ndarray:
        return self.transform_env_configuration(env_configuration=data_item.training_logs.get_config(), policy=self.policy)

    def transform_env_configuration(self, env_configuration: EnvConfiguration, policy: str,) -> np.ndarray:
        assert self.policy is not None, "Policy not instantiated"
        if policy == "mlp" or self.policy == "cnn" or self.policy == "bnn":                                                     # NEW (bnn)
            transformed = self.transform_mlp(env_configuration=env_configuration)
            if self.input_scaler is not None:
                transformed = self.input_scaler.transform(X=transformed.reshape(1, -1)).squeeze()
            return transformed
        raise NotImplementedError("Unknown policy: {}".format(policy))

    @staticmethod
    def transform_mlp(env_configuration: EnvConfiguration) -> np.ndarray:
        raise NotImplementedError("Transform mlp not implemented")

    @abstractmethod
    def get_mapping_transformed(self, env_configuration: EnvConfiguration) -> Dict:
        raise NotImplementedError("Get mapping transformed not implemented")

    @abstractmethod
    def get_original_env_configuration(self, env_config_transformed: np.ndarray) -> EnvConfiguration:
        raise NotImplementedError("Get original env configuration not implemented")

    @staticmethod
    def get_scalers_for_data(
        data: np.ndarray, labels: np.ndarray, regression: bool
    ) -> Tuple[Optional[Scaler], Optional[Scaler]]:
        raise NotImplementedError("Get scalers for data not implemented")

    @abstractmethod
    def compute_distance(self, env_config_1: EnvConfiguration, env_config_2: EnvConfiguration) -> float:
        raise NotImplementedError("Compute distance not implemented")

    @staticmethod
    def sampling(
        data: np.ndarray, labels: np.ndarray, seed: int, under: bool = False, sampling_percentage: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:

        logger = Log("sampling")

        if sampling_percentage > 0.0:
            if under:
                sampler = RandomUnderSampler(sampling_strategy=sampling_percentage, random_state=seed)
            else:
                sampler = RandomOverSampler(sampling_strategy=sampling_percentage, random_state=seed)

            logger.info("Label proportions before sampling: {}".format(labels.mean()))
            sampled_data, sampled_labels = sampler.fit_resample(X=data, y=labels)
            logger.info("Label proportions after sampling: {}".format(sampled_labels.mean()))

            return sampled_data, sampled_labels

        return data, labels

    @staticmethod
    def split_train_test(
        test_split: float,
        data: np.ndarray,
        labels: np.ndarray,
        seed: int,
        oversample_minority_class_percentage: float = 0.0,
        regression: bool = False,
        shuffle: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        logger = Log("split_train_test")

        if test_split > 0.0:
            if not regression:
                train_data, test_data, train_labels, test_labels = train_test_split(
                    data, labels, test_size=test_split, shuffle=shuffle, stratify=labels, random_state=seed
                )
            else:
                train_data, test_data, train_labels, test_labels = train_test_split(
                    data, labels, test_size=test_split, shuffle=shuffle, random_state=seed
                )
        else:
            train_data, train_labels, test_data, test_labels = data, labels, np.asarray([]), np.asarray([])
            if shuffle:
                np.random.shuffle(train_data)
                np.random.shuffle(train_labels)

        if not regression and oversample_minority_class_percentage > 0.0:
            undersampler = RandomUnderSampler(sampling_strategy=oversample_minority_class_percentage)
            logger.info("Label proportions before undersampling: {}".format(labels.mean()))
            previous_shape = None
            # FIXME
            if len(data.shape) > 2:
                previous_shape = data.shape[1:]
                data = data.reshape(data.shape[0], -1)
            oversampled_data, oversampled_labels = undersampler.fit_resample(X=train_data, y=train_labels)
            logger.info("Label proportions after undersampling: {}".format(oversampled_labels.mean()))

            train_data, train_labels = oversampled_data, oversampled_labels

            if previous_shape is not None and test_split > 0.0:
                train_data = train_data.reshape(-1, *previous_shape)
                test_data = test_data.reshape(-1, *previous_shape)
                logger.info("Train data shape: {}, Test data shape: {}".format(train_data.shape, test_data.shape))
            elif previous_shape is not None:
                train_data = train_data.reshape(-1, *previous_shape)
                logger.info("Train data shape: {}".format(train_data.shape))

        return train_data, train_labels.reshape(-1), test_data, test_labels.reshape(-1)

    def preprocess_test_data(self, test_data: np.ndarray, test_labels: np.ndarray,) -> Tuple[np.ndarray, np.ndarray]:
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(X=test_labels.reshape(len(test_labels), 1)).reshape(
                    len(test_labels)
                )
        return test_data, test_labels

    # also assigns input and output scalers
    def preprocess_train_and_test_data(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        regression: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger = Log("preprocess_train_and_test_data")
        # The statistics required for the transformation (e.g., the mean) are estimated
        # from the training set and are applied to all data sets (e.g., the test set or new samples)
        self.input_scaler, self.output_scaler = self.get_scalers_for_data(
            data=train_data, labels=train_labels.reshape(len(train_labels), 1), regression=regression
        )
        if self.input_scaler is not None:
            logger.info("Preprocessing input data")
            train_data = self.input_scaler.transform(X=train_data)
        if self.output_scaler is not None:
            logger.info("Preprocessing output data")
            train_labels = self.output_scaler.transform(X=train_labels.reshape(len(train_labels), 1)).reshape(
                len(train_labels)
            )
        if len(test_data) > 0:
            if self.input_scaler is not None:
                test_data = self.input_scaler.transform(X=test_data)
            if self.output_scaler is not None:
                test_labels = self.output_scaler.transform(X=test_labels.reshape(len(test_labels), 1)).reshape(
                    len(test_labels)
                )
        return train_data, train_labels, test_data, test_labels

    def transform_data(
        self,
        seed: int,
        test_split: float = 0.2,
        oversample_minority_class_percentage: float = 0.0,
        regression: bool = False,
        weight_loss: bool = False,
        preprocess: bool = False,
        training_progress: bool = False,
        shuffle: bool = True,
        dnn: bool = True,
    ) -> Union[
        Tuple[TorchDataset, TorchDataset], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:

        num_features = self.get_num_features()

        if training_progress:
            # + 1 is the index of the data in the original dataset
            num_features += 1

        data = np.zeros(shape=(len(self.dataset), num_features))
        labels = np.zeros(shape=(len(self.dataset), 1))

        for idx in range(len(self.dataset)):
            data_item = self.dataset[idx]
            if not regression:
                labels[idx] = data_item.label
            else:
                assert data_item.regression_value is not None
                labels[idx] = data_item.regression_value
            a = self.transform_data_item(data_item=data_item)

            if training_progress:
                data[idx] = np.append(a, idx)

        train_data, train_labels, test_data, test_labels = self.split_train_test(
            test_split=test_split,
            data=data,
            labels=labels,
            oversample_minority_class_percentage=oversample_minority_class_percentage,
            regression=regression,
            seed=seed,
            shuffle=shuffle,
        )
        if training_progress:
            if regression and preprocess:
                train_data, train_labels, test_data, test_labels = self.preprocess_train_and_test_data(
                    train_data=train_data,
                    train_labels=train_labels,
                    test_data=test_data,
                    test_labels=test_labels,
                    regression=regression,
                )
                assert (
                    self.input_scaler is None
                ), "Not possible to scale input features when distinguishing train and test dataset"
                assert self.output_scaler is not None, "Output scaler must be assigned in regression problems"
            return train_data, train_labels, test_data, test_labels

        if preprocess:
            train_data, train_labels, test_data, test_labels = self.preprocess_train_and_test_data(
                train_data=train_data,
                train_labels=train_labels,
                test_data=test_data,
                test_labels=test_labels,
                regression=regression,
            )
        else:
            return train_data, train_labels, test_data, test_labels

        if dnn:
            return (
                TorchDataset(data=train_data, labels=train_labels, regression=regression, weight_loss=weight_loss),
                TorchDataset(data=test_data, labels=test_labels, regression=regression, weight_loss=weight_loss),
            )
        return train_data, train_labels, test_data, test_labels


    @staticmethod
    def augment_minority_class(                                                                                         # NEW (bnn)
        data: np.ndarray,
        labels: np.ndarray,
        seed: int = 0,
        num_new_samples_per_original: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates synthetic data for the minority class by adding slight perturbations.

        :param data: The dataset (2D np.ndarray)
        :param labels: The labels (1D np.ndarray)
        :param seed: Random seed
        :param num_new_samples_per_original: How many synthetic samples to create per minority sample
        :return: (augmented_data, augmented_labels)
        """
        features_to_augment = {
                            2: 0.01,  # heading_ego
                            3: 0.05,  # position_ego[0]
                            4: 0.05,  # position_ego[1]
                        }
        integer_features = {
                            0: (1, 10),  # num_lanes
                            1: (0, 10),  # goal_lane_idx
                        }

        logger = Log("augmentation")
        np.random.seed(seed)

        if features_to_augment is None:
            features_to_augment = {}
        if integer_features is None:
            integer_features = {}

        logger.info("Starting synthetic augmentation on minority class...")
        minority_idx = np.where(labels == 1)[0]
        synthetic_samples = []

        for idx in minority_idx:
            original = data[idx]
            for _ in range(num_new_samples_per_original):
                new_sample = original.copy()

                # Add noise to selected features
                for feat_idx, noise_std in features_to_augment.items():
                    new_sample[feat_idx] += np.random.normal(0, noise_std)

                # Handle integer features (round + clip)
                for int_idx, (min_val, max_val) in integer_features.items():
                    new_sample[int_idx] = np.clip(
                        int(round(new_sample[int_idx])), min_val, max_val
                    )

                synthetic_samples.append(new_sample)

        if synthetic_samples:
            synthetic_data = np.array(synthetic_samples)
            synthetic_labels = np.ones(len(synthetic_samples))

            # Stack to original
            augmented_data = np.vstack([data, synthetic_data])
            augmented_labels = np.hstack([labels, synthetic_labels])

            logger.info("Added {} synthetic samples.".format(len(synthetic_samples)))
            return augmented_data, augmented_labels
        else:
            logger.info("No minority samples found for augmentation.")
            return data, labels


