# flake8: noqa

# Fairseq Imports
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data import (
    LanguagePairDataset,
    MonolingualDataset,
    TokenBlockDataset,
    Dictionary,
    plasma_utils,
    data_utils,
)
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqLanguageModel,
    BaseFairseqModel,
    FairseqEncoder,
    FairseqDecoder,
    register_model_architecture,
    register_model,
)
from fairseq import utils

# HuggingFace Imports
from transformers import BertModel

# FastTransformer Imports
from fast_transformers.builders import (
    TransformerEncoderBuilder,
    TransformerDecoderBuilder,
)
from fast_transformers.masking import (
    TriangularCausalMask,
    LengthMask,
    FullMask,
)

# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Miscellaneous Import
from colorama import Fore, Style
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import inspect
import math
import os

#
#   DEBUGGING
#


class Debug:
    """Debug Class"""

    # Color dictionary
    colors = {
        0: Fore.WHITE,
        1: Fore.BLACK,
        2: Fore.RED,
        3: Fore.GREEN,
        4: Fore.YELLOW,
        5: Fore.BLUE,
        6: Fore.MAGENTA,
        7: Fore.CYAN,
    }

    def __init__(self, name, color):
        """Constructor Method"""

        # Get the color
        self.color = Debug.colors[color]

        # Get the class name
        self.name = name

    def ldf(self, iter):
        """Litmus Debug Method"""

        # Get function name
        frame = inspect.currentframe().f_back
        func_name = inspect.getframeinfo(frame).function

        # Litmus print
        print(
            f"{self.color}>>>>>>{Fore.RESET} {Style.BRIGHT}{self.name}{Style.RESET_ALL} - {func_name} {self.color}***{Fore.RESET} {iter}"
        )


#
#   MODEL SPECIFICATION
#


class BERT(FairseqEncoder):
    """BERT Model Declaration"""

    debug = Debug("BERT", 6)

    def __init__(self, args, dictionary):
        """Constructor for BERT specifications"""

        BERT.debug.ldf("<< START >>")

        # Super module call
        super().__init__(dictionary)
        BERT.debug.ldf("super()")

        # Instance variables
        self.args = args
        BERT.debug.ldf("var dev")

        # Initialize model
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        BERT.debug.ldf("pretrained model")

        # Run model of CUDA
        self.model.cuda()
        BERT.debug.ldf("model CUDA")
        BERT.debug.ldf("<< END >>")

    def forward(self, src_token):
        """Forward function to specify forward propagation"""

        BERT.debug.ldf("<< START >>")

        # Send data to device
        src_token = src_token.to(src_token.device).long()
        BERT.debug.ldf("src_token")

        # Return logits from BERT << BROKEN >>
        output = self.model(src_token)
        BERT.debug.ldf("output")

        # Return result
        BERT.debug.ldf("<< END >>")
        return output


class SymphonyNet(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task):
        """SymphonyNet Structure Definition"""
        SymphonyNet.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        SymphonyNet.debug.ldf("super()")

        # Get the embedding dimensions for the SymphonyNet model
        self.dec_embed_dim = args.dec_embed_dim
        SymphonyNet.debug.ldf("Decoder Dimension")

        # Set the EVENT, TRACK, and DURATION embedding layers
        self.wEvte = nn.Embedding(args.evt_voc_size, args.dec_embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.dec_embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.dec_embed_dim)
        SymphonyNet.debug.ldf("Embedding Layers")

        # Get the maximum number of tokens per sample
        self.max_pos = args.tokens_per_sample
        SymphonyNet.debug.ldf("Maximum Tokens Per Sample")

        # Set permutation invariance configurations
        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos + 1, args.dec_embed_dim)
            self.wMpe = nn.Embedding(args.max_mea_pos + 1, args.dec_embed_dim)
            SymphonyNet.debug.ldf("perm_inv > 1")
        else:
            self.wpe = nn.Embedding(self.max_pos + 1, args.dec_embed_dim)
            SymphonyNet.debug.ldf("perm_inv == 0")

        # Setup dropout and layer normalization layers for reuse
        self.drop = nn.Dropout(args.dec_dropout)
        self.ln_f = nn.LayerNorm(args.dec_embed_dim, eps=1e-6)
        SymphonyNet.debug.ldf("Dropout & LayerNorm")

        # Build the decoder model
        self.model = TransformerEncoderBuilder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation='gelu',
            dropout=args.dec_dropout,
            attention_type="causal-linear",
        ).get()
        SymphonyNet.debug.ldf("Decoder Model")

        # Generate attention mask
        self.attn_mask = TriangularCausalMask(self.max_pos)
        SymphonyNet.debug.ldf("Attention Mask")

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
        SymphonyNet.debug.ldf("Output Layers")

        # Initialize the weights for the model
        self.apply(self._init_weights)
        SymphonyNet.debug.ldf("Init Weights")

        # Set zero embeddings for EVENT, DURATION, and TRACK for padding symbol
        # TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        SymphonyNet.debug.ldf("Zero Input Embedding Layers")

        # Set Zero embeddings for permuation invariance
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
            SymphonyNet.debug.ldf("perm_inv (zero) > 1")
        else:
            self.wpe.weight.data[0].zero_()
            SymphonyNet.debug.ldf("perm_inv (zero) == 1")

        SymphonyNet.debug.ldf("<< END >>")

    def _init_weights(self, module):
        """Initialization Step"""

        SymphonyNet.debug.ldf(f"{type(module)} | << START >>")

        # If the the given model is a linear or an embedding layer,
        # initialize weights with a mean of zero and a set std dev
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.dec_embed_dim**-0.5
            )
            SymphonyNet.debug.ldf("  0 Mean and Std Dev WEIGHT Init")

            # If the module is a linear layer with bias, set bias to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                SymphonyNet.debug.ldf("  0 BIAS")

        # If the module is a LayerNorm, set bias to zero
        # and weight initialized to 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            SymphonyNet.debug.ldf("  0 BIAS and 1 WEIGHT Fill")

        SymphonyNet.debug.ldf("  << END >>")

    def forward(
        self,
        decoder_in,
        src_lengths=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNet.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            x=decoder_in,
            src_lengths=src_lengths,
        )
        SymphonyNet.debug.ldf("Feature Extract")

        # Project the given features into the output layers
        # to get the logit projections of EVENT, DURATION
        # TRACK, and PREDICTION
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)
        SymphonyNet.debug.ldf("Final Projection")
        SymphonyNet.debug.ldf("<< END >>")

        # Return the logits for the EVENT, DURATION, TRACK, and INSTRUMENT
        return (evt_logits, dur_logits, trk_logits, ins_logits)

    def extract_features(self, x, src_lengths=None):
        """Extract feature method"""

        SymphonyNet.debug.ldf("<< START >>")

        # Permutate the tensor
        x = x.permute(1, 0, 2)
        SymphonyNet.debug.ldf("Input Permute")

        # Breaking down the dimensions of the input seq
        bsz, seq_len, dim = x.size()
        SymphonyNet.debug.ldf("Dimension Breakdown")

        # Create pad masking
        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        SymphonyNet.debug.ldf("Create Pad Masking")

        # Fill masking with length in mind
        if src_lengths is not None:
            len_mask = LengthMask(
                src_lengths, max_len=seq_len, device=x.device
            )
            SymphonyNet.debug.ldf("SRC LENGTH Filled Mask")
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1), max_len=seq_len, device=x.device
            )
            SymphonyNet.debug.ldf("PAD_MASK Length Filled Mask")

        # Pass to model
        outputs = self.model(x, self.attn_mask, len_mask)
        SymphonyNet.debug.ldf("Model Transformer Processing")

        # Pass to linear layer
        outputs = self.ln_f(outputs)
        SymphonyNet.debug.ldf("Linear Processing")

        # Return output
        SymphonyNet.debug.ldf("<< END >>")
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
        SymphonyNet.debug.ldf("<< max_positions >>")
        return None


#
#   FULL MODEL DEFINITION
#


@register_model("vivy_ve")
class VIVYNet_VE(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

    # DEBUG
    debug = Debug("VIVYNet", 3)

    @staticmethod
    def add_args(parser):
        """Argument Definition class"""
        VIVYNet_VE.debug.ldf("<< START >>")

        # Shorten Method
        parser.add_argument("--shorten_method", type=str, metavar="N")
        VIVYNet_VE.debug.ldf("shorten_method")

        # Shorten Data Split List
        parser.add_argument("--shorten_data_split_list", type=str, metavar="N")
        VIVYNet_VE.debug.ldf("shorten_data_split_list")

        # Token Per Sample
        parser.add_argument("--tokens_per_sample", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("tokens_per_sample")

        # Sample Break Mode
        parser.add_argument("--sample_break_mode", type=str, metavar="N")
        VIVYNet_VE.debug.ldf("sample_break_mode")

        # Ratio
        parser.add_argument("--ratio", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("ratio")

        # Sample Overlap Rate
        parser.add_argument("--sample_overlap_rate", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("sample_overlap_rate")

        # Permutation invariance
        parser.add_argument("--perm_inv", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("perm_inv")

        # Event Token Size
        parser.add_argument("--evt_voc_size", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("evt_voc_size")

        # Track Token Size
        parser.add_argument("--trk_voc_size", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("trk_voc_size")

        # Duration Vocab Size
        parser.add_argument("--dur_voc_size", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("dur_voc_size")

        # Instrument Vocab Size
        parser.add_argument("--ins_voc_size", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("ins_voc_size")

        # Maximum Relative Position
        parser.add_argument("--max_rel_pos", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("max_rel_pos")

        # Maximum Measure Count within a Sample
        parser.add_argument("--max_mea_pos", type=int, metavar="N")
        VIVYNet_VE.debug.ldf("max_mea_pos")

        # Decoder Embedding Dimension
        parser.add_argument(
            "--dec-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension",
        )
        VIVYNet_VE.debug.ldf("dec-embed-dim")

        # Decoder Attention Head Numbers
        parser.add_argument(
            "--dec-num-attention-heads",
            type=int,
            metavar="N",
            help="num attention heads",
        )
        VIVYNet_VE.debug.ldf("dec-num-attention-heads")

        # Number Decoder Layers
        parser.add_argument(
            "--dec-num-layers", type=int, metavar="N", help="num layers"
        )
        VIVYNet_VE.debug.ldf("dec-num-layers")

        # Decoder Dropout
        parser.add_argument(
            "--dec-dropout",
            type=float,
            metavar="D",
            help="dropout probability for all fully connected layers "
            "in the embeddings, encoder, and pooler",
        )
        VIVYNet_VE.debug.ldf("dec-dropout")

        parser.add_argument(
            "--freeze_enc",
            type=int,
            metavar="N",
            help="Freeze pretrained Encoder layers",
        )

        parser.add_argument(
            "--freeze_dec",
            type=int,
            metavar="N",
            help="Freeze pretrained Decoder layers",
        )

        VIVYNet_VE.debug.ldf("<< END >>")

    @classmethod
    def build_model(cls, args, task):
        """Build model function"""

        VIVYNet_VE.debug.ldf("<< START >>")

        # Create BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        VIVYNet_VE.debug.ldf("Model Creation: BERT")

        # Freezing the Encoder layers and load pretrained weights
        if args.freeze_enc == 1:
            # Freezing BERT
            for name, param in bert.named_parameters():
                param.requires_grad = False
        VIVYNet_VE.debug.ldf("Freezing pretrained Encoder layers")

        # Create SymphonyNet model
        symphony_net = SymphonyNet(args=args, task=task)
        VIVYNet_VE.debug.ldf("Model Creation: SymphonyNet")

        # Get the checkpoint
        checkpoint = torch.load(
            "../symphonynet/ckpt/checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
        )
        VIVYNet_VE.debug.ldf("Checkpoint loading")

        # WIP: Currently unable to transfer weights since the original checkpoint has different dimension due to
        #      being trained on a different dataset.

        # Freezing the Decoder layers and load pretrained weights
        if args.freeze_dec == 1:
            # Freezing self-attentions
            for name, param in symphony_net.named_parameters():
                if "self_attention" in name:
                    param.requires_grad = False
            VIVYNet_VE.debug.ldf("Freezing pretrained Decoder layers")

            # Zipping two models param dicts
            pretrained_params = []
            for param in symphony_net.state_dict():
                if not ("cross_attention" in param or "norm3" in param):
                    pretrained_params.append(param)
            VIVYNet_VE.debug.ldf("Weight targeting copy")

            # Weight copying
            VIVYNet_VE.debug.ldf("Proceed loading Decoder pretrained weights")
            with torch.no_grad():
                for param1, param2 in zip(
                    pretrained_params, checkpoint["model"]
                ):
                    symphony_net.state_dict()[param1].copy_(
                        checkpoint["model"][param2]
                    )
                    VIVYNet_VE.debug.ldf(f"Loading {param1}")
            VIVYNet_VE.debug.ldf("Loading Finished!")

        vivynet = VIVYNet_VE(bert, symphony_net)
        VIVYNet_VE.debug.ldf("COMPLETE MODEL COMPILATION: VIVYNet")

        # Return compiled vivynet
        VIVYNet_VE.debug.ldf("<< END >>")
        return vivynet

    def __init__(self, encoder, decoder):
        """Constructor for the VIVYNet model"""

        VIVYNet_VE.debug.ldf("<< START >>")

        # Retrieves attributes
        super().__init__(encoder, decoder)
        VIVYNet_VE.debug.ldf("super()")

        # Create instance variables based on parameters given
        self.encoder = encoder
        self.linear = torch.nn.Linear(768, 512)
        self.decoder = decoder
        VIVYNet_VE.debug.ldf("var dec")

        # Put models into train mode
        self.encoder.train()
        VIVYNet_VE.debug.ldf("encoder.train")
        VIVYNet_VE.debug.ldf("<< END >>")

    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        prev_output_tokens_lengths=None,
    ):
        """Forward propagation method"""

        VIVYNet_VE.debug.ldf("<< START >>")

        # Clear previously caluclated gradients
        self.encoder.zero_grad()
        VIVYNet_VE.debug.ldf("encoder.zero_grad()")

        # Get loss and the logits from the model
        enc_output = self.encoder(src_tokens.reshape(-1, 1))
        VIVYNet_VE.debug.ldf("res 1")

        # Process BERT out from 768 dimension to 512 dimension
        bert_out = self.linear(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNet_VE.debug.ldf(
            "res 2 : " + str(bert_out.shape) + " : " + str(src_lengths)
        )

        # Get overall features from decoder
        features = self.decoder(
            decoder_in=bert_out,
        )
        VIVYNet_VE.debug.ldf("res 3")

        # Return the logits
        VIVYNet_VE.debug.ldf("<< END >>")
        return features

    @property
    def supported_targets(self):
        """Supported Targets Property"""
        VIVYNet_VE.debug.ldf("<< supported_targets >>")
        return {"future"}


@register_model_architecture("vivy_ve", "vivy_train_ve")
def train_VE(args):
    """Train function"""

    # DEBUG
    debug = Debug("train", 4)
    debug.ldf("<< train >>")

    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 16)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)


#
#   DATASET SPECIFICATIONS
#


def copy_tensor(src, dst):
    """Tensor Copying Function"""

    # Check if the source and target tensors are equal in length
    assert dst.numel() == src.numel(), f"dst ({dst.numel()}) is not src ({src.numel()})"

    # Copy the target tokens to the source information
    dst.copy_(src)


def collate_tokens(
    values,
    max_sample_size,
    pad_idx,
    eos_idx=None,
    left_pad=False,
):
    """2D to 3D Tensor Function"""
    # Max batch size
    size = max(v.size(0) for v in values)

    # Generate the resulting values from the merge
    res = values[0].new(len(values), max_sample_size, values[0].size(-1)).fill_(pad_idx)

    # Iterate through the provided values for collation and copy the
    # tensor values to the resulting list
    for i, v in enumerate(values):
        copy_tensor(
            v, res[i][size - len(v) :] if left_pad else res[i][: len(v)]
        )

    # Return the result
    return res


def midi_collate(samples, max_sample_size, pad_idx, eos_idx):
    """Midi MultiHeadDataset Collater Function"""

    def merge(key, is_list=False):
        """Merge inner function"""

        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Apped the collated tokens to the resulting list
                res.append(
                    collate_tokens(
                        [s[key][i] for s in samples],
                        max_sample_size,
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )

            # Retun the result of the appending
            return res

        # If the given key is not a list, move here
        else:
            # Just return the collated tokens normally
            return collate_tokens(
                [s[key] for s in samples],
                max_sample_size,
                pad_idx,
                eos_idx,
                left_pad=False,
            )

    # Return nothing if samples provided is nothing
    if len(samples) == 0:
        return {}

    # Merge the source tokens
    src_tokens = merge("source")

    # If the sample's target is empty, merge the target tokens
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    # If not, set the target equal to the source dataset
    else:
        target = src_tokens

    # Return the resulting information
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(s["source"].size(0) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor(
                [s["source"].size(0) for s in samples]
            ),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples),
    }


def t2m_collate(samples, pad_idx, eos_idx):
    """Text2Music PairDataset Collate Function"""

    # TODO: add a merge func for text encoder_in
    def merge(key, is_list=False):
        """Merge inner function"""

        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Apped the collated tokens to the resulting list
                res.append(
                    collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )

            # Retun the result of the appending
            return res

        # If the given key is not a list, move here
        else:
            # Just return the collated tokens normally
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )

    # Return nothing if samples provided is nothing
    if len(samples) == 0:
        return {}

    # Merge the source tokens
    dec_in_tokens = merge("dec_input")

    # If the sample's target is empty, merge the target tokens
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    # If not, set the target equal to the source dataset
    else:
        target = dec_in_tokens

    # Return the resulting information
    # TODO: add info for text data
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(s["dec_input"].size(0) for s in samples),
        "net_input": {
            "enc_input": samples[0]["enc_input"],
            "dec_in_tokens": dec_in_tokens,
            "dec_in_lengths": torch.LongTensor(
                [s["dec_input"].size(0) for s in samples]
            ),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples),
    }


class TupleMultiHeadDataset(TokenBlockDataset):
    """Class Specification for Multiheaded Information"""

    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        ratio=4 + 1,
        sample_overlap_rate=4,
        permutation_invariant=3,
        trk_idx=2,
        spec_tok_cnt=4,
        evt_vocab_size=425,
        trk_vocab_size=44,
    ):
        """Constructor for class"""

        # Try to import modules from fairseq
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )

        # Raise errors if importingn fails
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        # Super call attributes and operations from parent class
        super(TokenBlockDataset, self).__init__()

        # Variable initialization
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets
        self.ratio = ratio
        self.perm_inv = permutation_invariant
        self.sample_len_max = block_size
        self.trk_idx = trk_idx
        self.cc_idx = evt_vocab_size - 1
        self.spec_tok_cnt = spec_tok_cnt
        self.max_trk_cnt = trk_vocab_size - spec_tok_cnt
        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        # Turn sizes list into a numpy array datatype
        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        # Set valuie of break_mode
        break_mode = break_mode if break_mode is not None else "complete_doc"
        assert break_mode == "complete_doc", break_mode

        # Transform and process sizes and other attributes
        sizes_cs = np.cumsum(sizes)
        piece_sep_ids = np.where(sizes == document_sep_len)[0].tolist()
        totpieces = len(piece_sep_ids)
        slice_indices = np.zeros((totpieces, 2), dtype=int)
        block_to_dataset_index = np.zeros((totpieces, 3), dtype=int)

        # Process slicde_indices and block_to_dataset_index arrays
        for i in range(len(piece_sep_ids)):
            s = piece_sep_ids[i - 1] if i > 0 else -1
            e = piece_sep_ids[i]
            slice_indices[i, :] = (
                sizes_cs[s] if s >= 0 else 0,
                sizes_cs[e - 1],
            )
            block_to_dataset_index[i, :] = (s + 1, 0, e - 1)

        # # Transform the slices, sizes, and block information
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]
        self._sizes[:] = self.sample_len_max
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(
            block_to_dataset_index
        )

    def __getitem__(self, index):
        """Item Retrieval Method"""

        # Create index pointers
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[
            index
        ]
        assert start_offset == 0, (start_ds_idx, start_offset, end_ds_idx)

        # Create temporary variables
        buffer = []
        cur_len = 0

        st = start_ds_idx

        # Process information
        for idx in range(st, end_ds_idx + 1):
            tmp = self.dataset[idx].view(-1, self.ratio)
            if self.perm_inv % 2 == 1:
                all_cc_pos = (
                    torch.nonzero(tmp[..., 0] == self.cc_idx).view(-1).tolist()
                )
                all_cc_pos.append(tmp.size(0))
                to_swap = []
                for pos, nexp in zip(all_cc_pos[:-1], all_cc_pos[1:]):
                    to_swap.append(tmp[pos:nexp, ...])
                to_swap_idx = torch.randperm(len(to_swap))
                tmp = torch.cat(
                    [tmp[: all_cc_pos[0], ...]]
                    + [to_swap[x] for x in to_swap_idx]
                )
            mea = (idx - st + 1) * 3
            mea_num = torch.zeros((tmp.size(0), 1), dtype=int)
            mea_num[2:, 0] = mea
            mea_num[1][0] = mea - 1
            mea_num[0][0] = mea - 2
            buffer.append(torch.cat((tmp, mea_num), dim=1))
            cur_len += tmp.size(0)
            if cur_len >= self.sample_len_max:
                break

        # Create buffer and calculate it
        buffer = torch.cat(buffer)
        if cur_len < self.sample_len_max:
            buffer = torch.cat(
                [buffer, buffer.new([[self.eos] * (self.ratio + 1)])]
            )

        # Get item
        item = buffer[: self.sample_len_max, ...]
        if self.perm_inv > 0:
            perm = torch.cat(
                [
                    torch.arange(self.spec_tok_cnt),
                    torch.randperm(self.max_trk_cnt) + self.spec_tok_cnt,
                ]
            )
            item[..., self.trk_idx].apply_(lambda x: perm[x])

        assert self.include_targets

        # Process item
        source = torch.cat(
            [
                item.new([[self.eos] * (self.ratio - 1) + [0, 0]]),
                item[:-1, ...],
            ]
        )
        on = torch.sum(item[:, 1].ne(self.pad)).item()

        # Return item
        return source, item, on


class MultiheadDataset(MonolingualDataset):
    """Final Preprocessing of the Multiheaded Datapoints"""

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab,
        add_eos_for_other_targets,
        shuffle,
        max_sample_size,
        targets=None,
        add_bos_token=False,
    ):
        """Contstructor for the class"""

        # Variable declaration and initialization
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.max_sample_size = max_sample_size
        self.add_bos_token = add_bos_token

        # Check if the a token in the given dataset
        # is taken the intended <bos> token
        assert not self.add_bos_token, "<bos> is occupied"

        # Format the target data into correct format where
        # its geared for future format
        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        assert len(targets) == 1 and targets[0] == "future"

        # Set target data
        self.targets = targets

    def collater(self, samples):
        """Token collater method"""

        # Return the collated information of the given sample
        return midi_collate(samples, self.max_sample_size, self.vocab.pad(), self.vocab.eos())

    def __getitem__(self, index):
        """Get item of an iterable based on its index"""

        # Make sure that the target data is not empty
        assert self.targets is not None

        # Get the source, target, and on of the passed in dataset
        source, target, on = self.dataset[index]

        # Generate the source and target information from the parsed info
        source, target = self._make_source_target(source, target, None)

        # Add the BOS token
        source, target = self._maybe_add_bos(source, target)

        # Return the processed information
        return {"id": index, "source": source, "target": target, "on": on}


class PairDataset(LanguagePairDataset):
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        midi_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
    ):
        """Text2Music Dataset classification"""

        # Super call
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict)

        # Variable definitions and initialization
        self.src = src
        self.src_dict = src_dict
        self.midi_dict = midi_dict
        self.tgt = tgt
        self.tgt_dict = tgt_dict

    def __getitem__(self, index):
        """Get item method"""

        # Extract information at given index
        enc_input = self.src[index]
        tgt_input = self.tgt[index]
        dec_input = tgt_input["source"]
        target = tgt_input["target"]
        on = tgt_input["on"]

        # Return the information
        return {
            "id": index,
            "enc_input": enc_input,
            "dec_input": dec_input,
            "target": target,
            "on": on,
        }

    def collater(self, samples):
        """Token collater method"""
        # Return the collated information of the given sample
        return t2m_collate(samples, self.midi_dict.pad(), self.midi_dict.eos())


@register_task("text2music_ve")
class VIVYData_VE(LanguageModelingTask):
    """Dataset Class Specification"""

    debug = Debug("VIVYData", 7)

    @staticmethod
    def add_args(parser):
        """Argument parsing"""

        VIVYData_VE.debug.ldf("<< START >>")

        # Get the data
        parser.add_argument("data", metavar="FILE", help="data")
        VIVYData_VE.debug.ldf("data")
        VIVYData_VE.debug.ldf("<< END >>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Task setup method"""

        VIVYData_VE.debug.ldf("<< START >>")

        # Load dictionaries from the data
        src_vocab = Dictionary.load(
            os.path.join(args.data + "/features", "dict.txt")
        )
        VIVYData_VE.debug.ldf("src_vocab")
        tgt_vocab = Dictionary.load(
            os.path.join(args.data + "/labels/bin", "dict.txt")
        )
        VIVYData_VE.debug.ldf("tgt_vocab")
        print("| [input] dictionary: {} types".format(len(src_vocab)))
        print("| [label] dictionary: {} types".format(len(tgt_vocab)))

        # Return the instance of the training class
        VIVYData_VE.debug.ldf("<< END >>")
        return VIVYData_VE(args, tgt_vocab, src_vocab)

    def __init__(self, args, label_vocab, input_vocab):
        """Constructor for VIVYTrain class"""

        VIVYData_VE.debug.ldf("<< START >>")

        # Set instance variables
        super().__init__(args, input_vocab, output_dictionary=label_vocab)
        # self.args = args
        self.src_vocab = input_vocab
        self.tgt_vocab = label_vocab
        VIVYData_VE.debug.ldf("var dec")
        VIVYData_VE.debug.ldf("<< END >>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split"""

        """
        TARGET DATA HANDLING
        """

        VIVYData_VE.debug.ldf(f"<< START (split: {split}) >>")

        # Split the paths to the data
        paths = utils.split_paths(self.args.data + "/labels/bin")
        assert len(paths) > 0
        VIVYData_VE.debug.ldf("TGT - paths")

        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        VIVYData_VE.debug.ldf("TGT - path split")

        # Read and get the information from the .bin and .idx files
        tgt_datasets = data_utils.load_indexed_dataset(
            split_path, self.tgt_vocab, self.args.dataset_impl, combine=combine
        )
        VIVYData_VE.debug.ldf("TGT - tgt_datasets")

        # If no dataset instance is created, raise an error
        if tgt_datasets is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # Shorten dataset if need be
        tgt_datasets = maybe_shorten_dataset(
            tgt_datasets,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        VIVYData_VE.debug.ldf("TGT - maybe_shorten_dataset")

        tgt_datasets = TupleMultiHeadDataset(
            tgt_datasets,
            tgt_datasets.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            ratio=self.args.ratio + 1,
            sample_overlap_rate=self.args.sample_overlap_rate,
            permutation_invariant=self.args.perm_inv,
            # trk_idx=self.args.trk_idx,
            # spec_tok_cnt=self.args.spec_tok_cnt,
            evt_vocab_size=self.args.evt_voc_size,
            trk_vocab_size=self.args.trk_voc_size,
        )
        VIVYData_VE.debug.ldf("TGT - TupleMultiHeadDataset Init")

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        VIVYData_VE.debug.ldf("TGT - Add EOS for other targets")

        final_target = MultiheadDataset(
            dataset=tgt_datasets,
            sizes=tgt_datasets.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            max_sample_size=self.args.tokens_per_sample,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=False,  # Note: it should be from args,
        )
        VIVYData_VE.debug.ldf("TGT - MultiheadDataset Init")
        VIVYData_VE.debug.ldf(
            f"TGT - *FINALIZED* (size: {len(final_target.sizes)})"
        )

        """
        SOURCE DATA HANDLING
        """

        # Split the paths to the data
        paths = utils.split_paths(self.args.data + "/features")
        assert len(paths) > 0
        VIVYData_VE.debug.ldf("SRC - paths")

        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        VIVYData_VE.debug.ldf("SRC - path split")

        # Create dataset instance
        src_dataset = data_utils.load_indexed_dataset(
            split_path, self.src_vocab, self.args.dataset_impl, combine=combine
        )
        VIVYData_VE.debug.ldf(
            f"SRC - *FINALIZED* (size: {len(src_dataset.sizes)})"
        )

        """
        DATASET COMPILATION
        """

        self.datasets[split] = PairDataset(
            src=src_dataset,
            src_sizes=src_dataset.sizes,
            src_dict=self.src_vocab,
            midi_dict=self.dictionary,
            tgt=final_target,
            tgt_sizes=final_target.sizes,
            tgt_dict=self.tgt_vocab,
        )
        VIVYData_VE.debug.ldf("COMPILATION")
        VIVYData_VE.debug.ldf(f"<< END (split: {split}) >>")

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        VIVYData_VE.debug.ldf("<< src_vocab >>")
        return self.src_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        VIVYData_VE.debug.ldf("<< tgt_vocab >>")
        return self.tgt_vocab

    def _initialize_pair_dataset(self, **kwargs):
        """Method to Initialize the Pair Dataset (Text, Midi)"""
        return PairDataset(**kwargs)


#
#   CRITERION SPECIFICATION
#


@register_criterion("nll_loss_ve")
class ModelCriterion_VE(CrossEntropyCriterion):
    """Model criterion class"""

    debug = Debug("ModelCriterion_VE", 5)

    def forward(self, model, sample, reduce=True):
        """Forward function for the criterion"""

        ModelCriterion_VE.debug.ldf("<< START >>")

        # Get output of the model
        net_output = model(
            sample["net_input"]["enc_input"],
            sample["net_input"]["dec_in_tokens"],
        )
        ModelCriterion_VE.debug.ldf("VIVYNet Output")

        # Compute the losses of the output
        losses = self.compute_loss(model, net_output, sample, reduce=reduce)
        ModelCriterion_VE.debug.ldf("Process Losses")

        # Aggregate losses
        loss = torch.mean(torch.stack(losses))
        ModelCriterion_VE.debug.ldf("Aggregate Losses")

        # Create logging output
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample["ntokens"],
            "on_sample_size": sample["ntokens"],
        }
        ModelCriterion_VE.debug.ldf("Generate Logging")

        # Return information
        ModelCriterion_VE.debug.ldf("<< END >>")
        return loss, sample["ntokens"], logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """Loss computation"""

        ModelCriterion_VE.debug.ldf("<< START >>")

        # Get normalized probability from the net_ouput
        lprobs_tuple = model.get_normalized_probs(net_output, log_probs=True)
        losses = []
        ModelCriterion_VE.debug.ldf("Normalized Probability")

        # Iterate through all normalized probability
        for idx, lprobs in enumerate(lprobs_tuple):
            # Change the probability dimension
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output)[..., idx].view(-1)

            # Calculate loss
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )

            # Append the loss to the loss list
            losses.append(loss)
        ModelCriterion_VE.debug.ldf("Losses Calculations")

        # Return the list of losses
        ModelCriterion_VE.debug.ldf("<< END >>")
        return losses
