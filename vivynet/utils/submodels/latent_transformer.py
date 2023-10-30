# Fairseq Imports
from fairseq.models import FairseqEncoder

# FastTransformer Imports
from fast_transformers.builders import (
    TransformerEncoderBuilder,
    RecurrentEncoderBuilder,
)
from fast_transformers.masking import LengthMask, TriangularCausalMask

# Torch Imports
import torch
import torch.nn as nn

# Debug Imports
from vivynet.utils.debug import Debug


class LatentTransformerEncoder(FairseqEncoder):
    """SymphonyNet Model Specification"""

    debug = Debug("LatentTransformerEncoder", 2)

    def __init__(self, args, task, inference=False):
        """Transformer Latent Definition"""

        LatentTransformerEncoder.debug.ldf("<< START >>")

        # Super call for a FairseqEncoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        LatentTransformerEncoder.debug.ldf("super()")

        # Instance variable definition
        self.pad_idx = task.target_dictionary.pad()
        self.max_pos = args.tokens_per_sample
        self.inference = inference

        # Build the encoder model
        model_builder = (
            RecurrentEncoderBuilder
            if self.inference
            else TransformerEncoderBuilder
        )
        self.encoder_model = model_builder.from_kwargs(
            n_layers=args.latent_num_layers,
            n_heads=args.latent_num_attention_heads,
            query_dimensions=args.latent_embed_dim
            // args.latent_num_attention_heads,
            value_dimensions=args.latent_embed_dim
            // args.latent_num_attention_heads,
            feed_forward_dimensions=4 * args.latent_embed_dim,
            activation="gelu",
            dropout=args.latent_dropout,
            attention_type="linear",  # https://fast-transformers.github.io/attention/
        ).get()
        LatentTransformerEncoder.debug.ldf("Latent Encoder Model")

        # Generate attention mask
        # self.attn_mask = TriangularCausalMask(self.max_pos)
        LatentTransformerEncoder.debug.ldf("Attention Mask")

        # Define translation layers
        self.input_section = nn.Sequential(
            nn.Linear(args.latent_input_dim, args.latent_embed_dim),
            nn.ReLU(),
            nn.Dropout(p=args.latent_dropout),
        )
        self.output_section = nn.Sequential(
            nn.Linear(args.latent_embed_dim, args.latent_output_dim),
            nn.ReLU(),
            nn.Dropout(p=args.latent_dropout),
        )

    @staticmethod
    def add_args(parser, check):
        """Method to add arguments for this specific model"""

        # Latent encoder number layers
        if not check(parser, "latent_num_layers"):
            parser.add_argument(
                "--latent_num_layers",
                type=int,
                metavar="N",
                help="Latent encoder number of layers",
            )
            LatentTransformerEncoder.debug.ldf("latent_num_layers")

        # Latent encoder number of attention heads
        if not check(parser, "latent_num_attention_heads"):
            parser.add_argument(
                "--latent_num_attention_heads",
                type=int,
                metavar="N",
                help="Latent encoder number of attention heads",
            )
            LatentTransformerEncoder.debug.ldf("latent_num_attention_heads")

        # Latent encoder embedding dimensions
        if not check(parser, "latent_embed_dim"):
            parser.add_argument(
                "--latent_embed_dim",
                type=int,
                metavar="N",
                help="Latent encoder embedded dimensions",
            )
            LatentTransformerEncoder.debug.ldf("latent_embed_dim")

        # Latent encoder dropout
        if not check(parser, "latent_dropout"):
            parser.add_argument(
                "--latent_dropout",
                type=float,
                metavar="N",
                help="Latent encoder dropout",
            )
            LatentTransformerEncoder.debug.ldf("latent_dropout")

        # Latent encoder input dimension
        if not check(parser, "latent_input_dim"):
            parser.add_argument(
                "--latent_input_dim",
                type=float,
                metavar="N",
                help="Latent encoder input dimension",
            )
            LatentTransformerEncoder.debug.ldf("latent_input_dim")

        # Latent encoder output dimension
        if not check(parser, "latent_output_dim"):
            parser.add_argument(
                "--latent_output_dim",
                type=float,
                metavar="N",
                help="Latent encoder output dimension",
            )
            LatentTransformerEncoder.debug.ldf("latent_output_dim")

    def forward(
        self,
        x,          # [N, L, E] <=> [1, 54, 768]
        src_lengths=None,
        state=None,
    ):
        """LatentTransformerEncoder's Forward Function"""

        LatentTransformerEncoder.debug.ldf("<< START >>")

        # Permutate the tensor
        # x = x.permute(1, 0, 2)
        LatentTransformerEncoder.debug.ldf("Input Permute")

        # Breaking down the dimensions of the input seq
        bsz, seq_len, dim = x.size()
        LatentTransformerEncoder.debug.ldf("Dimension Breakdown")

        # Create pad masking
        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        LatentTransformerEncoder.debug.ldf("Create Pad Masking")

        # Fill masking with length in mind
        if src_lengths is not None:
            len_mask = LengthMask(src_lengths, max_len=seq_len, device=x.device)
            LatentTransformerEncoder.debug.ldf("SRC LENGTH Filled Mask")
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1), max_len=seq_len, device=x.device
            )
            LatentTransformerEncoder.debug.ldf("PAD_MASK Length Filled Mask")

        # Calculate information from the data into the model
        if self.inference:
            outputs = self.input_section(x.squeeze(0))
            outputs = outputs.permute(1, 0, 2)
            outputs = outputs.squeeze(0)
            outputs, state = self.encoder_model(x=outputs)
        else:
            outputs = self.input_section(x)
            # print("latent size: ", outputs.size())
            # input()
            outputs = self.encoder_model(x = outputs, attn_mask = None, length_mask = len_mask)
            # outputs = outputs.permute(1, 0, 2)
        outputs = self.output_section(outputs)
        # print(outputs.shape)
        # input()
        LatentTransformerEncoder.debug.ldf("Model Computation")

        # Return output
        LatentTransformerEncoder.debug.ldf("<< END >>")
        if self.inference:
            return outputs, state
        else:
            return outputs
