# Fairseq Imports
from fairseq.models import FairseqEncoder

# Torch Imports
import torch.nn as nn

# Debug Imports
from vivynet.utils.debug import Debug

# Datastructure import
from collections import OrderedDict


class IntermediarySection(FairseqEncoder):
    """Intermediary section to help translate the two modalities"""

    debug = Debug("IntermediarySection", 7)

    def __init__(
        self,
        args,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_layers,
        dropout_rate=0.25,
    ):
        """Constructor for intermediary section"""

        IntermediarySection.debug.ldf("<< START >>")

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

    def forward(self, x):
        """Forward function to process input"""
        IntermediarySection.debug.ldf("<< forward >>")
        return self.model(x)
