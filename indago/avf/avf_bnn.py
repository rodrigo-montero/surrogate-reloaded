import torch as th
import torch.nn as nn
from torch import Tensor

from typing import Tuple, Optional
from indago.avf.avf_policy import AvfPolicy
from indago.config import DONKEY_ENV_NAME, HUMANOID_ENV_NAME, PARK_ENV_NAME
from indago.utils.torch_utils import DEVICE


try:
    from blitz.modules import BayesianLinear
    from blitz.utils import variational_estimator
except ImportError:
    BayesianLinear = None
    variational_estimator = lambda x: x



class MC_Dropout(nn.Dropout):
    """
    Custom Dropout layer that remains active at test time (for MC Dropout)
    
    - Dropout is traditionally used during training as a regularization technique (randomly dropping activations).
    - In MC Dropout, we also apply dropout at test time, which causes each forward pass to randomly drop out parts of the network.
    - This results in stochastic predictions — running multiple forward passes yields a distribution over outputs.

    """
    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.dropout(input, self.p, training=True, inplace=self.inplace)


def make_mcdropout_mlp(input_size: int, layers: int = 1, regression: bool = False, hidden_layer_size: int = 64) -> nn.Module:
    """
    MLP with MC Dropout used at both train and test time.
    
    Hyperparameters:
        - hidden_layer_size: Number of units per hidden layer
        - layers: Number of hidden layers
        - dropout rate: Currently fixed at p=0.5, can be parameterized
        - activation: Currently ReLU, can be swapped for others if needed

    The output layer has 1 unit if regression=True, otherwise 2 units for binary classification.
    """
    modules = []

    modules.append(nn.Linear(input_size, hidden_layer_size))                    # Input layer
    modules.append(nn.ReLU())
    modules.append(MC_Dropout(p=0.5))

    for _ in range(layers - 1):
        modules.append(nn.Linear(hidden_layer_size, hidden_layer_size))         # Hidden layers (layers - 1 because the first layer is already added)
        modules.append(nn.ReLU())
        modules.append(MC_Dropout(p=0.5))

    modules.append(nn.Linear(hidden_layer_size, 1 if regression else 2))        # Output layer

    return nn.Sequential(*modules)



@variational_estimator
class BlitzBNN(nn.Module):
    """
    Model which uses BayesianLinear layers from blitz-bnn which model weights as probability distributions.
    Each bayesian layer learns mean and standard deviation.
    The @variational_estimator decorator enables sampling and loss computation for variational inference.

    Difference between MLP and BNN:
        - We sample weights from these distributions during training.
        - We don’t use standard loss functions directly — we optimize an Evidence Lower Bound (ELBO)
        - ELBO = Expected Log Likelihood - KL Divergence

    Hyperparameters that can be tuned: 
        - hidden_size: size of hidden layer
        - number of layers: add more BayesianLayers
        - Activation: change ReLu for tanh
    
    Current Results:
        - Accuracy test set: 0.9111880046136102
        - Precision: 0.22
        - Recall: 0.34
        - F-measure: 0.27
        - AUROC: 0.72

    """
    def __init__(self, input_size: int, layers: int = 1, regression: bool = False, hidden_layer_size: int = 128):
        super().__init__()
        self.regression = regression
        self.fcs = nn.ModuleList()
        #hidden_layer_size = 128

        # Fully connected Bayesian layers
        self.fcs.append(BayesianLinear(input_size, hidden_layer_size))
        for _ in range(layers - 1):
            self.fcs.append(BayesianLinear(hidden_layer_size, hidden_layer_size))

        # Output layer
        self.out = BayesianLinear(hidden_layer_size, 1 if regression else 2)

        # Classification: use LogSoftmax → NLLLoss
        if not regression:
            self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        for layer in self.fcs:
            x = nn.ReLU()(layer(x))
        x = self.out(x)
        if not self.regression:
            x = self.log_softmax(x)
        return x
    
    def predict_mc(self, x: Tensor, num_samples: int = 30) -> Tuple[Tensor, Tensor]:
        """
        Performs multiple forward passes through the BNN and returns:
        - mean prediction
        - standard deviation (uncertainty estimate)
        """
        print("Predicting with model in training mode:", self.training)
        self.train()  # forces stochastic behavior during inference
        outputs = [self.forward(x) for _ in range(num_samples)]
        stacked = th.stack(outputs)
        return stacked.mean(0), stacked.std(0)





class AvfBnnPolicy(AvfPolicy):
    """
    BNN wrapper policy that can be instantiated with different backends: MC Dropout or Blitz.
    This is controlled via the `bnn_type` argument.
    """
    def __init__(
        self, env_name: str, input_size: int, regression: bool = False, layers: int = 4,
        learning_rate: float = 3e-4, hidden_layer_size: int = 128, bnn_type: str = "blitz"
    ) -> None:
        super().__init__(
            loss_type="classification",
            regression=regression,
            learning_rate=learning_rate,
            input_size=input_size,
            layers=layers,
            avf_policy="bnn",
        )

        if env_name not in {PARK_ENV_NAME, HUMANOID_ENV_NAME, DONKEY_ENV_NAME}:
            raise NotImplementedError(f"Unknown env name: {env_name}")

        if bnn_type is None or bnn_type == "":
            bnn_type = "blitz"

        if bnn_type == "mc-dropout":
            self.model = make_mcdropout_mlp(input_size=input_size, layers=layers, regression=regression)
        elif bnn_type == "blitz":
            assert BayesianLinear is not None, "blitz-bnn is not installed. Run `pip install blitz-bnn`."
            self.model = BlitzBNN(input_size=input_size, layers=layers, regression=regression, hidden_layer_size=hidden_layer_size)
        else:
            raise ValueError(f"Unsupported BNN type: {bnn_type}")

        self.model.to(device=DEVICE)

    def get_model(self) -> nn.Module:
        return self.model
    

    def forward_and_loss(
        self, data: Tensor, target: Tensor, training: bool = True, weights: Tensor = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        output = self.forward(data)  # log-probabilities if classification

        if not self.regression:
            if self.loss_type == "classification":
                # During training: return NLLLoss and predictions
                if training:
                    # Output is log-softmax → use NLLLoss
                    loss = self.loss_function(input=output, target=target, weights=weights)
                    predictions = output.detach().argmax(dim=1)
                    return loss, predictions

                # During evaluation: return log-probs and predictions
                predictions = output.detach().argmax(dim=1)
                return output.detach(), predictions
            else:
                raise NotImplementedError("Loss type {} is not implemented".format(self.loss_type))

        # === Regression ===
        predictions = th.squeeze(self.forward(data))
        if training:
            return self.loss_function(input=predictions, target=target, weights=weights, regression=True), predictions
        else:
            return predictions






