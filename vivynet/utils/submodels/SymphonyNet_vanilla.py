# Fairseq Imports
from fairseq.models import FairseqDecoder
from fairseq import utils

# FastTransformer Imports
from fast_transformers.builders import (
    TransformerDecoderBuilder,
    RecurrentDecoderBuilder,
)
from fast_transformers.masking import (
    TriangularCausalMask,
    LengthMask,
    FullMask,
)

# Torch Imports
import torch
import torch.nn as nn
from torch import Tensor

# Debug Imports
from vivynet.utils.debug import Debug

# Miscellaneous Import
from typing import Dict, List, Optional, Tuple


class SymphonyNetVanilla(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task, inference=False):
        """SymphonyNet Structure Definition"""
        SymphonyNetVanilla.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        super().__init__(task.target_dictionary)
        SymphonyNetVanilla.debug.ldf("super()")

        # Instance variable definition
        self.inference = inference

        # Get the embedding dimensions for the SymphonyNet model
        self.dec_embed_dim = args.dec_embed_dim
        SymphonyNetVanilla.debug.ldf("Decoder Dimension")

        # Set the EVENT, TRACK, and DURATION embedding layers
        self.wEvte = nn.Embedding(args.evt_voc_size, args.dec_embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.dec_embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.dec_embed_dim)
        SymphonyNetVanilla.debug.ldf("Embedding Layers")

        # Get the maximum number of tokens per sample
        self.max_pos = args.tokens_per_sample
        SymphonyNetVanilla.debug.ldf("Maximum Tokens Per Sample")

        # Set permutation invariance configurations
        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos + 1, args.dec_embed_dim)
            self.wMpe = nn.Embedding(args.max_mea_pos + 1, args.dec_embed_dim)
            SymphonyNetVanilla.debug.ldf("perm_inv > 1")
        else:
            self.wpe = nn.Embedding(self.max_pos + 1, args.dec_embed_dim)
            SymphonyNetVanilla.debug.ldf("perm_inv == 0")

        # Setup dropout and layer normalization layers for reuse
        self.drop = nn.Dropout(args.dec_dropout)
        self.ln_f = nn.LayerNorm(args.dec_embed_dim, eps=1e-6)
        SymphonyNetVanilla.debug.ldf("Dropout & LayerNorm")

        # Build the decoder model
        model_builder = (
            RecurrentDecoderBuilder
            if self.inference
            else TransformerDecoderBuilder
        )
        self.decoder_model = model_builder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation="gelu",
            dropout=args.dec_dropout,
            self_attention_type="causal-linear",
            cross_attention_type="full",
        ).get()
        SymphonyNetVanilla.debug.ldf("Decoder Model")

        # Generate attention mask
        self.attn_mask = TriangularCausalMask(self.max_pos)
        SymphonyNetVanilla.debug.ldf("Attention Mask")

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
        SymphonyNetVanilla.debug.ldf("Output Layers")

        # Initialize the weights for the model
        self.apply(self._init_weights)
        SymphonyNetVanilla.debug.ldf("Init Weights")

        # Set zero embeddings for EVENT, DURATION, and TRACK for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        SymphonyNetVanilla.debug.ldf("Zero Input Embedding Layers")

        # Set Zero embeddings for permutation invariance
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
            SymphonyNetVanilla.debug.ldf("perm_inv (zero) > 1")
        else:
            self.wpe.weight.data[0].zero_()
            SymphonyNetVanilla.debug.ldf("perm_inv (zero) == 1")

        SymphonyNetVanilla.debug.ldf("<< END >>")

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
            SymphonyNetVanilla.debug.ldf("dec_embed_dim")

        # Decoder number of attention heads
        if not check(parser, "dec_num_attention_heads"):
            parser.add_argument(
                "--dec_num_attention_heads",
                type=int,
                metavar="N",
                help="Decoder number of attention heads",
            )
            SymphonyNetVanilla.debug.ldf("dec_num_attention_heads")

        # Decoder number of transformer layers
        if not check(parser, "dec_num_layers"):
            parser.add_argument(
                "--dec_num_layers",
                type=int,
                metavar="N",
                help="Decoder number of transformer layers",
            )
            SymphonyNetVanilla.debug.ldf("dec_num_layers")

        # Decoder dropout
        if not check(parser, "dec_dropout"):
            parser.add_argument(
                "--dec_dropout",
                type=float,
                metavar="N",
                help="Decoder dropout",
            )
            SymphonyNetVanilla.debug.ldf("dec_dropout")

    def forward(
        self,
        encoder_out,
        decoder_in,
        src_lengths=None,
        encoder_out_lengths=None,
        state=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNetVanilla.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            decoder_in=decoder_in,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            encoder_out_lengths=encoder_out_lengths,
            state=state,
        )
        if self.inference:
            features, state = features
        SymphonyNetVanilla.debug.ldf("Feature Extract")

        # Project the given features into the output layers
        # to get the logit projections of EVENT, DURATION
        # TRACK, and PREDICTION
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)
        SymphonyNetVanilla.debug.ldf("Final Projection")
        SymphonyNetVanilla.debug.ldf("<< END >>")

        # Return the logits for the EVENT, DURATION, TRACK, and INSTRUMENT
        if self.inference:
            return (evt_logits, dur_logits, trk_logits, ins_logits), state
        return (evt_logits, dur_logits, trk_logits, ins_logits)

    def extract_features(
        self,
        decoder_in,
        encoder_out=None,
        src_lengths=None,
        encoder_out_lengths=None,
        state=None,
    ):
        """Extract feature method"""
        # TODO: Understand how SymphonyNet masks work, including LengthMask and
        # TODO:   TriangularMask
        # TODO: Understand Permutation Invariant in code

        SymphonyNetVanilla.debug.ldf("<< START >>")

        # Process the decoder input
        bsz, seq_len, _ = decoder_in.size()
        SymphonyNetVanilla.debug.ldf("process decoder_in")

        # Process the output of the decoder
        enc_len, enc_bsz, embed_dim = encoder_out.size()
        SymphonyNetVanilla.debug.ldf("process encoder_out")

        # Get the event embedding
        evt_emb = self.wEvte(decoder_in[..., 0])
        SymphonyNetVanilla.debug.ldf("event embedding")

        # Get the event mask
        SymphonyNetVanilla.debug.ldf("event mask")
        evton_mask = (
            decoder_in[..., 1]
            .ne(self.pad_idx)
            .float()[..., None]
            .to(decoder_in.device)
        )
        SymphonyNetVanilla.debug.ldf("event mask")

        # Get the duration embedding
        tmp = self.wDure(decoder_in[..., 1])
        dur_emb = tmp * evton_mask
        SymphonyNetVanilla.debug.ldf("duration embedding")

        # Get the track embedding
        tmp = self.wTrke(decoder_in[..., 2])
        trk_emb = tmp * evton_mask
        SymphonyNetVanilla.debug.ldf("track embedding")

        # Calculate length mask for the decoder input information
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
        SymphonyNetVanilla.debug.ldf("Calculating LengthMask for tgt")

        # Calculate length mask for the encoder output information
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
        SymphonyNetVanilla.debug.ldf("Calculating LengthMask for src")

        # Apply the full mask to the cross attention layers
        full_mask = FullMask(N=seq_len, M=enc_len, device=decoder_in.device)
        SymphonyNetVanilla.debug.ldf("Full mask for cross attention layer")

        # Apply permutation variance techniques
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
        SymphonyNetVanilla.debug.ldf("permutation invariant")

        # Combine the embeddings together
        x = evt_emb + dur_emb + trk_emb + pos_emb
        SymphonyNetVanilla.debug.ldf("combine all midi features")

        SymphonyNetVanilla.debug.ldf("apply dropout")
        x = self.drop(x)

        # Calculate information from the data into the model
        if self.inference:
            outputs, state = self.decoder_model(
                x=x.squeeze(0),
                memory=encoder_out,
                memory_length_mask=enc_len_mask
            )
        else:
            outputs = self.decoder_model(
                x=x,
                memory=encoder_out,
                x_mask=self.attn_mask,
                x_length_mask=len_mask,
                memory_mask=full_mask,
                memory_length_mask=enc_len_mask,
            )
        outputs = self.ln_f(outputs)
        SymphonyNetVanilla.debug.ldf("Model Computation")

        # Return outputs
        SymphonyNetVanilla.debug.ldf("<< END >>")
        if self.inference:
            print('>>>>>>>>>>>>>>>>>>>> INFERENCE')
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
        SymphonyNetVanilla.debug.ldf("<< max_positions >>")
        return 4096
