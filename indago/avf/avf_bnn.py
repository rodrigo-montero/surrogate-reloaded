import torch as th
import torch.nn as nn
from torch import Tensor

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
    Hardcoded MLP with MC Dropout used at both train and test time.
    
    Hyperparameters that can be tuned:
        - hidden_size: Number of units per hidden layer
        - drop_out rate: How much randomness injected (p), higher dropout increases uncertainty and regularisation, lower dropout is more confident but might overfit
        - number of hidden layers: Repeat 3-line pattern
        - activation function: e.g. replace nn.ReLU() with nn.Tanh() or nn.LeakyReLU()

        - possible extra: add nn.BatchNorm1d(hidden_size) between Linear and ReLU to stabilize training
    """
    hidden_size = hidden_layer_size
    modules = [
        nn.Linear(input_size, hidden_size),                 # Standard fully connected layer
        nn.ReLU(),                                          # Activation Layer, applies non-linearity 
        MC_Dropout(p=0.5),                                  # During training and testing, randomly zeros out p of the hidden activations -> introduces stochasticity -> turning the model bayesian (in some way)
        nn.Linear(hidden_size, hidden_size),                # Start second layer
        nn.ReLU(),                                          #
        MC_Dropout(p=0.5),                                  # 
        nn.Linear(hidden_size, 1 if regression else 2),     # Output layer, we have binary classification so 2
    ]
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
    def __init__(self, input_size: int, layers: int = 1, regression: bool = False, hidden_layer_size: int = 64):
        super().__init__()
        print("Hidden layer size is:", hidden_layer_size)
        hidden = hidden_layer_size
        self.fcs = nn.ModuleList()

        
        self.fcs.append(BayesianLinear(input_size, hidden))             # First layer: input_size -> hidden
        for _ in range(layers - 1):                                     # Additional hidden layers: hidden -> hidden (layers - 1 times)
            self.fcs.append(BayesianLinear(hidden, hidden))
        self.out = BayesianLinear(hidden, 1 if regression else 2)       # Output layer

    def forward(self, x):
        for layer in self.fcs:
            x = nn.ReLU()(layer(x))
        return self.out(x)





class AvfBnnPolicy(AvfPolicy):
    """
    BNN wrapper policy that can be instantiated with different backends: MC Dropout or Blitz.
    This is controlled via the `bnn_type` argument.
    """
    def __init__(
        self, env_name: str, input_size: int, regression: bool = False, layers: int = 4,
        learning_rate: float = 3e-4, hidden_layer_size: int = 64, bnn_type: str = "blitz"
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

        if bnn_type is None:
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





