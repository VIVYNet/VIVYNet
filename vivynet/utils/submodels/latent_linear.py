# Torch Imports
import torch.nn as nn

# Debug Imports
from vivynet.utils.debug import Debug

# Datastructure import
from collections import OrderedDict


class LatentLinear(nn.Module):
    """Intermediary section to help translate the two modalities"""

    debug = Debug("LatentLinear", 7)

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_layers,
        dropout_rate,
    ):
        """Constructor for intermediary section"""

        LatentLinear.debug.ldf("<< START >>")

        # Super call
        super(LatentLinear, self).__init__()
        LatentLinear.debug.ldf("super()")

        # Cast the parameters
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        hidden_layers = int(hidden_layers)
        dropout_rate = float(dropout_rate)

        # Define the input and output layer
        self.input_layer = [
            (
                "latent_input_linear",
                nn.Linear(input_dim, hidden_dim),
            ),
            (
                "latent_input_ReLU",
                nn.ReLU(),
            ),
            (
                "latent_input_dropout",
                nn.Dropout(p=dropout_rate),
            ),
            (
                "latent_input_layer_norm",
                nn.LayerNorm(hidden_dim),
            ),
        ]
        self.output_layer = [
            (
                "latent_output_linear",
                nn.Linear(hidden_dim, output_dim),
            ),
            (
                "latent_output_ReLU",
                nn.ReLU(),
            ),
            (
                "latent_output_dropout",
                nn.Dropout(p=dropout_rate),
            ),
            (
                "latent_output_layer_norm",
                nn.LayerNorm(output_dim),
            ),
        ]

        # Define the hidden layers
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(
                (
                    f"latent_hidden_{i+1}_linear",
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.hidden_layers.append(
                (
                    f"latent_hidden_{i+1}_ReLU",
                    nn.ReLU(),
                )
            )
            self.hidden_layers.append(
                (
                    f"latent_hidden_{i+1}_dropout",
                    nn.Dropout(p=dropout_rate),
                )
            )
            self.hidden_layers.append(
                (
                    f"latent_hidden_{i+1}_layer_norm",
                    nn.LayerNorm(hidden_dim),
                )
            )

        # Build the model
        self.final_specifications = (
            self.input_layer + self.hidden_layers + self.output_layer
        )
        self.model = nn.Sequential(OrderedDict(self.final_specifications))

    @staticmethod
    def add_args(parser, check):
        """Method to add arguments for this specific model"""

        # Latent input dimension
        if not check(parser, "latent_input_dim"):
            parser.add_argument(
                "--latent_input_dim",
                type=int,
                metavar="N",
                help="Specify the latent model's input dimensions",
            )
            LatentLinear.debug.ldf("latent_input_dim")

        # Latent hidden dimension
        if not check(parser, "latent_hidden_dim"):
            parser.add_argument(
                "--latent_hidden_dim",
                type=int,
                metavar="N",
                help="Specify the latent model's hidden dimensions",
            )
            LatentLinear.debug.ldf("latent_hidden_dim")

        # Latent output dimension
        if not check(parser, "latent_output_dim"):
            parser.add_argument(
                "--latent_output_dim",
                type=int,
                metavar="N",
                help="Specify the latent model's output dimensions",
            )
            LatentLinear.debug.ldf("latent_output_dim")

        # Latent hidden layers
        if not check(parser, "latent_hidden_layers"):
            parser.add_argument(
                "--latent_hidden_layers",
                type=int,
                metavar="N",
                help="Specify the latent model's hidden layers",
            )
            LatentLinear.debug.ldf("latent_hidden_layers")

        # Latent hidden layers
        if not check(parser, "latent_dropout_rate"):
            parser.add_argument(
                "--latent_dropout_rate",
                type=float,
                metavar="N",
                help="Specify the latent model's dropout rate",
            )
            LatentLinear.debug.ldf("latent_dropout_rate")

    def forward(self, x):
        """Forward function to process input"""
        LatentLinear.debug.ldf("<< forward >>")
        return self.model(x)
