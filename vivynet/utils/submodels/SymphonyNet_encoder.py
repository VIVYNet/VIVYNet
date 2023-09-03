# Fairseq Imports
from fairseq.models import FairseqDecoder
from fairseq import utils

# FastTransformer Imports
from fast_transformers.builders import (
    TransformerEncoderBuilder,
    RecurrentEncoderBuilder,
)
from fast_transformers.masking import (
    TriangularCausalMask,
    LengthMask,
)

# Torch Imports
import torch
import torch.nn as nn
from torch import Tensor

# Debug Imports
from vivynet.utils.debug import Debug

# Miscellaneous Import
from typing import Dict, List, Optional, Tuple


class SymphonyNetEncoder(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task, inference=False):
        """SymphonyNet Structure Definition"""
        SymphonyNetEncoder.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        SymphonyNetEncoder.debug.ldf("super()")

        # Instance variable definition
        self.inference = inference

        # Get the embedding dimensions for the SymphonyNet model
        self.dec_embed_dim = args.dec_embed_dim
        SymphonyNetEncoder.debug.ldf("Decoder Dimension")

        # Set the EVENT, TRACK, and DURATION embedding layers
        self.wEvte = nn.Embedding(args.evt_voc_size, args.dec_embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.dec_embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.dec_embed_dim)
        SymphonyNetEncoder.debug.ldf("Embedding Layers")

        # Get the maximum number of tokens per sample
        self.max_pos = args.tokens_per_sample
        SymphonyNetEncoder.debug.ldf("Maximum Tokens Per Sample")

        # Set permutation invariance configurations
        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos + 1, args.dec_embed_dim)
            self.wMpe = nn.Embedding(args.max_mea_pos + 1, args.dec_embed_dim)
            SymphonyNetEncoder.debug.ldf("perm_inv > 1")
        else:
            self.wpe = nn.Embedding(self.max_pos + 1, args.dec_embed_dim)
            SymphonyNetEncoder.debug.ldf("perm_inv == 0")

        # Setup dropout and layer normalization layers for reuse
        self.drop = nn.Dropout(args.dec_dropout)
        self.ln_f = nn.LayerNorm(args.dec_embed_dim, eps=1e-6)
        SymphonyNetEncoder.debug.ldf("Dropout & LayerNorm")

        # Build the decoder model
        # Build the decoder model
        model_builder = (
            RecurrentEncoderBuilder
            if self.inference
            else TransformerEncoderBuilder
        )
        self.model = model_builder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation="gelu",
            dropout=args.dec_dropout,
            attention_type="causal-linear",
        ).get()
        SymphonyNetEncoder.debug.ldf("Decoder Model")

        # Generate attention mask
        self.attn_mask = TriangularCausalMask(self.max_pos)
        SymphonyNetEncoder.debug.ldf("Attention Mask")

        # Define output layers for EVENT, DURATION, TRACK, and INSTRUMENT
        self.proj_evt = nn.Linear(
            args.dec_embed_dim, args.evt_voc_size, bias=False
        )
        self.proj_dur = nn.Linear(
            args.dec_embed_dim, args.dur_voc_size, bias=False
        )
        self.proj_trk = nn.Linear(
            args.dec_embed_dim, args.trk_voc_size, bias=False
        )
        self.proj_ins = nn.Linear(
            args.dec_embed_dim, args.ins_voc_size, bias=False
        )
        SymphonyNetEncoder.debug.ldf("Output Layers")

        # Initialize the weights for the model
        self.apply(self._init_weights)
        SymphonyNetEncoder.debug.ldf("Init Weights")

        # Set zero embeddings for EVENT, DURATION, and TRACK for padding symbol
        # TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        SymphonyNetEncoder.debug.ldf("Zero Input Embedding Layers")

        # Set Zero embeddings for permutation invariance
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
            SymphonyNetEncoder.debug.ldf("perm_inv (zero) > 1")
        else:
            self.wpe.weight.data[0].zero_()
            SymphonyNetEncoder.debug.ldf("perm_inv (zero) == 1")

        SymphonyNetEncoder.debug.ldf("<< END >>")

    def _init_weights(self, module):
        """Initialization Step"""

        # If the the given model is a linear or an embedding layer,
        # initialize weights with a mean of zero and a set std dev
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.dec_embed_dim**-0.5)
            # If the module is a linear layer with bias, set bias to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        # If the module is a LayerNorm, set bias to zero
        # and weight initialized to 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def add_args(parser, check):
        """Method to add arguments for this specific model"""

        # Decoder embedding dimension
        # Latent input dimension
        if not check(parser, "dec_embed_dim"):
            parser.add_argument(
                "--dec_embed_dim",
                type=int,
                metavar="N",
                help="Decoder embedding dimension",
            )
            SymphonyNetEncoder.debug.ldf("dec_embed_dim")

        # Decoder number of attention heads
        if not check(parser, "dec_num_attention_heads"):
            parser.add_argument(
                "--dec_num_attention_heads",
                type=int,
                metavar="N",
                help="Decoder number of attention heads",
            )
            SymphonyNetEncoder.debug.ldf("dec_num_attention_heads")

        # Decoder number of transformer layers
        if not check(parser, "dec_num_layers"):
            parser.add_argument(
                "--dec_num_layers",
                type=int,
                metavar="N",
                help="Decoder number of transformer layers",
            )
            SymphonyNetEncoder.debug.ldf("dec_num_layers")

        # Decoder dropout
        if not check(parser, "dec_dropout"):
            parser.add_argument(
                "--dec_dropout",
                type=float,
                metavar="N",
                help="Decoder dropout",
            )
            SymphonyNetEncoder.debug.ldf("dec_dropout")

    def forward(
        self,
        decoder_in,
        src_lengths=None,
        state=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNetEncoder.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            x=decoder_in, src_lengths=src_lengths, state=state
        )
        SymphonyNetEncoder.debug.ldf("Feature Extract")

        # Project the given features into the output layers
        # to get the logit projections of EVENT, DURATION
        # TRACK, and PREDICTION
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)
        SymphonyNetEncoder.debug.ldf("Final Projection")
        SymphonyNetEncoder.debug.ldf("<< END >>")

        # Return the logits for the EVENT, DURATION, TRACK, and INSTRUMENT
        return (evt_logits, dur_logits, trk_logits, ins_logits)

    def extract_features(self, x, src_lengths=None, state=None):
        """Extract feature method"""

        SymphonyNetEncoder.debug.ldf("<< START >>")

        # Permutate the tensor
        x = x.permute(1, 0, 2)
        SymphonyNetEncoder.debug.ldf("Input Permute")

        # Breaking down the dimensions of the input seq
        bsz, seq_len, dim = x.size()
        SymphonyNetEncoder.debug.ldf("Dimension Breakdown")

        # Create pad masking
        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        SymphonyNetEncoder.debug.ldf("Create Pad Masking")

        # Fill masking with length in mind
        if src_lengths is not None:
            len_mask = LengthMask(src_lengths, max_len=seq_len, device=x.device)
            SymphonyNetEncoder.debug.ldf("SRC LENGTH Filled Mask")
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1), max_len=seq_len, device=x.device
            )
            SymphonyNetEncoder.debug.ldf("PAD_MASK Length Filled Mask")

        # Calculate information from the data into the model
        if self.inference:
            outputs = self.model(
                x=x.squeeze(0),
                memory_length_mask=None,
                state=state,
            )
        else:
            outputs = self.model(x, self.attn_mask, len_mask)
        outputs = self.ln_f(outputs)
        SymphonyNetEncoder.debug.ldf("Model Computation")

        # Return output
        SymphonyNetEncoder.debug.ldf("<< END >>")
        if self.inference:
            return outputs, state
        else:
            return outputs

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            return tuple(
                utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
                for logits in net_output
            )
        else:
            return tuple(
                utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
                for logits in net_output
            )

    def max_positions(self):
        """Return nothing for max positions"""
        SymphonyNetEncoder.debug.ldf("<< max_positions >>")
        return 4096
