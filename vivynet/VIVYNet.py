# flake8: noqa

# Fairseq Imports
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.tasks import register_task
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

# Debug imports
from vivynet.debug import Debug

# Miscellaneous Import
from colorama import Fore, Style
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import inspect
import math
import os

#
#   CONSTANT DEFINITIONS
#

DISABLE_DEBUG = True

#
#   MODEL SPECIFICATION
#


class BERT(FairseqEncoder):
    """BERT Model Declaration"""

    debug = Debug("BERT", 6, disable=DISABLE_DEBUG)

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
        """Forward function to specify forward propogation"""

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

    debug = Debug("SymphonyNet", 2, disable=DISABLE_DEBUG)

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
        self.decoder_model = TransformerDecoderBuilder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim
            // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim
            // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation="gelu",
            dropout=args.dec_dropout,
            self_attention_type="causal-linear",
            cross_attention_type="full",  # Fully masked so that each domain can be merged
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
        encoder_out,
        decoder_in,
        src_lengths=None,
        encoder_out_lengths=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNet.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            decoder_in=decoder_in,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            encoder_out_lengths=encoder_out_lengths,
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

    def extract_features(
        self,
        decoder_in,
        encoder_out=None,
        src_lengths=None,
        encoder_out_lengths=None,
    ):
        """Extract feature method"""

        SymphonyNet.debug.ldf("<< START >>")

        SymphonyNet.debug.ldf("process decoder_in")
        bsz, seq_len, ratio = decoder_in.size()

        SymphonyNet.debug.ldf("process encoder_out")
        enc_len, enc_bsz, embed_dim = encoder_out.size()

        SymphonyNet.debug.ldf("event embedding")
        evt_emb = self.wEvte(decoder_in[..., 0])

        SymphonyNet.debug.ldf("event mask")
        # if not mapping to pad, padding idx will only occer at last
        evton_mask = (
            decoder_in[..., 1]
            .ne(self.pad_idx)
            .float()[..., None]
            .to(decoder_in.device)
        )  # TODO: elaborate, why the mask is on the 2nd

        SymphonyNet.debug.ldf("duration embedding")
        tmp = self.wDure(decoder_in[..., 1])
        dur_emb = tmp * evton_mask

        SymphonyNet.debug.ldf("track embedding")
        tmp = self.wTrke(decoder_in[..., 2])
        trk_emb = tmp * evton_mask

        SymphonyNet.debug.ldf("Calculating LengthMask for tgt")
        # Note: Calc LengthMask for src_lengths
        pad_mask = (
            decoder_in[..., 0].ne(self.pad_idx).long().to(decoder_in.device)
        )
        if src_lengths is not None:
            len_mask = LengthMask(
                src_lengths, max_len=seq_len, device=decoder_in.device
            )
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1),
                max_len=seq_len,
                device=decoder_in.device,
            )

        SymphonyNet.debug.ldf("Calculating LengthMask for src")
        # Note: Calc LengthMask for endoer_out_lengths
        if encoder_out_lengths is not None:
            enc_len_mask = LengthMask(
                torch.tensor(encoder_out_lengths, dtype=torch.int),
                max_len=enc_len,
                device=encoder_out.device,
            )
        else:
            # WIP: Calc LengthMask when enc_out_len is none
            # enc_pad_mask = x[1].ne(self.enc_pad_idx).long().to(x.device)
            enc_len_mask = LengthMask(
                torch.tensor(enc_len, dtype=torch.int),
                max_len=enc_len,
                device=encoder_out.device,
            )

        SymphonyNet.debug.ldf("full mask for cross attention layer")
        # WIP: Implement FullMask for Cross Attention layer
        full_mask = FullMask(N=seq_len, M=enc_len, device=decoder_in.device)

        SymphonyNet.debug.ldf("permutation invariant")
        # Note: Perform Permutation Invariant
        if self.perm_inv > 1:
            rel_pos = pad_mask * decoder_in[..., 4]
            rel_pos_mask = (
                rel_pos.ne(0).float()[..., None].to(decoder_in.device)
            )  # ignore bom, chord, eos

            measure_ids = pad_mask * decoder_in[..., 5]
            mea_mask = (
                measure_ids.ne(0).float()[..., None].to(decoder_in.device)
            )  # ignore eos

            pos_emb = rel_pos_mask * self.wRpe(rel_pos) + mea_mask * self.wMpe(
                measure_ids
            )

        else:
            # set position ids to exclude padding symbols
            position_ids = pad_mask * (
                torch.arange(1, 1 + seq_len)
                .to(decoder_in.device)
                .repeat(bsz, 1)
            )
            pos_emb = self.wpe(position_ids)

        SymphonyNet.debug.ldf("combine all midi features")
        x = (
            evt_emb + dur_emb + trk_emb + pos_emb
        )  # [bsz, seq_len, embedding_dim]

        SymphonyNet.debug.ldf("apply dropout")
        x = self.drop(x)

        SymphonyNet.debug.ldf("Model Computation")
        doutputs = self.decoder_model(
            x=x,  # decoder_in shape: [batch_size, dec_length, embed_dim]
            memory=encoder_out,  # encoder_out shape: [batch_size, enc_length, embed_dim]
            x_mask=self.attn_mask,
            x_length_mask=len_mask,
            memory_mask=full_mask,  # WIP
            memory_length_mask=enc_len_mask,  # WIP
        )
        SymphonyNet.debug.ldf("apply layer norm")
        doutputs = self.ln_f(doutputs)

        SymphonyNet.debug.ldf("<< END >>")
        return doutputs

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
        return 4096


#
#   FULL MODEL DEFINITION
#


@register_model("vivy")
class VIVYNet(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

    # DEBUG
    debug = Debug("VIVYNet", 3, disable=DISABLE_DEBUG)

    @staticmethod
    def add_args(parser):
        """Argument Definition class"""
        VIVYNet.debug.ldf("<< START >>")

        # Shorten Method
        parser.add_argument("--shorten_method", type=str, metavar="N")
        VIVYNet.debug.ldf("shorten_method")

        # Shorten Data Split List
        parser.add_argument("--shorten_data_split_list", type=str, metavar="N")
        VIVYNet.debug.ldf("shorten_data_split_list")

        # Token Per Sample
        parser.add_argument("--tokens_per_sample", type=int, metavar="N")
        VIVYNet.debug.ldf("tokens_per_sample")

        # Sample Break Mode
        parser.add_argument("--sample_break_mode", type=str, metavar="N")
        VIVYNet.debug.ldf("sample_break_mode")

        # Ratio
        parser.add_argument("--ratio", type=int, metavar="N")
        VIVYNet.debug.ldf("ratio")

        # Sample Overlap Rate
        parser.add_argument("--sample_overlap_rate", type=int, metavar="N")
        VIVYNet.debug.ldf("sample_overlap_rate")

        # Permutation invariance
        parser.add_argument("--perm_inv", type=int, metavar="N")
        VIVYNet.debug.ldf("perm_inv")

        # Event Token Size
        parser.add_argument("--evt_voc_size", type=int, metavar="N")
        VIVYNet.debug.ldf("evt_voc_size")

        # Track Token Size
        parser.add_argument("--trk_voc_size", type=int, metavar="N")
        VIVYNet.debug.ldf("trk_voc_size")

        # Duration Vocab Size
        parser.add_argument("--dur_voc_size", type=int, metavar="N")
        VIVYNet.debug.ldf("dur_voc_size")

        # Instrument Vocab Size
        parser.add_argument("--ins_voc_size", type=int, metavar="N")
        VIVYNet.debug.ldf("ins_voc_size")

        # Maximum Relative Position
        parser.add_argument("--max_rel_pos", type=int, metavar="N")
        VIVYNet.debug.ldf("max_rel_pos")

        # Maximum Measure Count within a Sample
        parser.add_argument("--max_mea_pos", type=int, metavar="N")
        VIVYNet.debug.ldf("max_mea_pos")

        # Decoder Embedding Dimension
        parser.add_argument(
            "--dec-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension",
        )
        VIVYNet.debug.ldf("dec-embed-dim")

        # Decoder Attention Head Numbers
        parser.add_argument(
            "--dec-num-attention-heads",
            type=int,
            metavar="N",
            help="num attention heads",
        )
        VIVYNet.debug.ldf("dec-num-attention-heads")

        # Number Decoder Layers
        parser.add_argument(
            "--dec-num-layers", type=int, metavar="N", help="num layers"
        )
        VIVYNet.debug.ldf("dec-num-layers")

        # Decoder Dropout
        parser.add_argument(
            "--dec-dropout",
            type=float,
            metavar="D",
            help="dropout probability for all fully connected layers "
            "in the embeddings, encoder, and pooler",
        )
        VIVYNet.debug.ldf("dec-dropout")

        # Freeze encoder
        parser.add_argument(
            "--freeze_enc",
            type=int,
            metavar="N",
            help="Freeze pretrained Encoder layers",
        )
        VIVYNet.debug.ldf("freeze_enc")

        # Freeze decoder
        parser.add_argument(
            "--freeze_dec",
            type=int,
            metavar="N",
            help="Freeze pretrained Decoder layers",
        )
        VIVYNet.debug.ldf("freeze_dec")

        VIVYNet.debug.ldf("<< END >>")

    @classmethod
    def build_model(cls, args, task):
        """Build model function"""

        VIVYNet.debug.ldf("<< START >>")

        # Create BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        VIVYNet.debug.ldf("Model Creation: BERT")

        # Freezing the Encoder layers and load pretrained weights
        if args.freeze_enc == 1:
            # Freezing BERT
            VIVYNet.debug.ldf("Freezing pretrained Encoder layers")
            for name, param in bert.named_parameters():
                param.requires_grad = False

        # Create SymphonyNet model
        symphony_net = SymphonyNet(args=args, task=task)
        VIVYNet.debug.ldf("Model Creation: SymphonyNet")

        # Get the checkpoint
        checkpoint = torch.load(
            "../symphonynet/ckpt/checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
        )
        VIVYNet.debug.ldf("Checkpoint loading")

        # WIP: Currently unable to transfer weights since the original checkpoint has different dimension due to
        #      being trained on a different dataset.

        # Freezing the Decoder layers and load pretrained weights
        if args.freeze_dec == 1:
            # Freezing self-attentions
            VIVYNet.debug.ldf("Freezing pretrained Decoder layers")
            for name, param in symphony_net.named_parameters():
                if "self_attention" or "wEvte.weight" or "wTrke.weight" or "wDure.weight" or "wRpe.weight" or "wMpe.weight" in name:
                    param.requires_grad = False

            # for name, param in symphony_net.named_parameters():
            #     print(name, " ", param)

            # Zipping two models param dicts
            pretrained_params = []
            for param in symphony_net.state_dict():
                if not ("cross_attention" in param or "norm3" in param):
                    pretrained_params.append(param)
            VIVYNet.debug.ldf("Weight targeting copy")

            # Weight copying
            VIVYNet.debug.ldf("Proceed loading Decoder pretrained weights")
            with torch.no_grad():
                for param1, param2 in zip(
                    pretrained_params, checkpoint["model"]
                ):
                    symphony_net.state_dict()[param1].copy_(
                        checkpoint["model"][param2]
                    )
                    VIVYNet.debug.ldf(f"Loading {param1}")
            VIVYNet.debug.ldf("Loading Finished!")

        vivynet = VIVYNet(bert, symphony_net)
        VIVYNet.debug.ldf("COMPLETE MODEL COMPILATION: VIVYNet")

        # Return
        VIVYNet.debug.ldf("<< END >>")
        return vivynet

    def __init__(self, encoder, decoder):
        """Constructor for the VIVYNet model"""

        VIVYNet.debug.ldf("<< START >>")

        # Retrieves attributes
        super().__init__(encoder, decoder)
        VIVYNet.debug.ldf("super()")

        # Create instance variables based on parameters given
        self.encoder = encoder
        self.linear = torch.nn.Linear(768, 512)
        self.decoder = decoder
        VIVYNet.debug.ldf("var dec")

        # Put models into train mode
        self.encoder.train()
        VIVYNet.debug.ldf("encoder.train")
        VIVYNet.debug.ldf("<< END >>")

    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        prev_output_tokens_lengths=None,
    ):
        """Forward propagation method"""

        VIVYNet.debug.ldf("<< START >>")

        # Clear previously caluclated gradients
        self.encoder.zero_grad()
        VIVYNet.debug.ldf("encoder.zero_grad()")

        # Get loss and the logits from the model
        enc_output = self.encoder(src_tokens)
        VIVYNet.debug.ldf("res 1")

        bert_out = self.linear(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNet.debug.ldf(
            "res 2 : " + str(bert_out.shape) + " : " + str(src_lengths)
        )

        # Get overall features from decoder
        features = self.decoder(
            encoder_out=bert_out,
            decoder_in=prev_output_tokens,
            src_lengths=prev_output_tokens_lengths,
            encoder_out_lengths=src_lengths,  # TODO: Pass in the Encoder Output length
        )
        VIVYNet.debug.ldf("res 3")

        # Return the logits
        VIVYNet.debug.ldf("<< END >>")
        return features

    @property
    def supported_targets(self):
        """Supported Targets Property"""
        VIVYNet.debug.ldf("<< supported_targets >>")
        return {"future"}


@register_model_architecture("vivy", "vivy_train")
def train(args):
    """Train function"""

    # DEBUG
    debug = Debug("train", 4, disable=DISABLE_DEBUG)
    debug.ldf("<< train >>")

    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 32)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)
