# Fairseq Imports
from fairseq.models import FairseqEncoder, FairseqDecoder
from fairseq import utils

# HuggingFace Imports
from transformers import BertModel

# FastTransformer Imports
from fast_transformers.builders import (
    TransformerEncoderBuilder,
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

# Note: a flag that checks if its inference or not inside the bert

class BERTBaseMulti(FairseqEncoder):
    """BERTBaseMulti Model Declaration"""

    debug = Debug("BERTBaseMulti", 6)

    def __init__(self, args, dictionary):
        """Constructor for BERTBaseMulti specifications"""

        BERTBaseMulti.debug.ldf("<< START >>")

        # Super module call
        super().__init__(dictionary)
        BERTBaseMulti.debug.ldf("super()")

        # Instance variables
        self.args = args
        BERTBaseMulti.debug.ldf("var dev")

        # Initialize model
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        BERTBaseMulti.debug.ldf("pretrained model")

        # Run model of CUDA
        self.model.cuda()
        BERTBaseMulti.debug.ldf("model CUDA")
        BERTBaseMulti.debug.ldf("<< END >>")

    def forward(self, src_token):
        """Forward function to specify forward propogation"""

        BERTBaseMulti.debug.ldf("<< START >>")

        # Send data to device
        src_token = src_token.to(src_token.device).long()
        BERTBaseMulti.debug.ldf("src_token")

        # Return logits from BERTBaseMulti << BROKEN >>
        output = self.model(src_token)
        BERTBaseMulti.debug.ldf("output")

        # Return result
        BERTBaseMulti.debug.ldf("<< END >>")
        return output


class BERTBaseEN(FairseqEncoder):
    """BERTBaseEN Model Declaration"""

    debug = Debug("BERTBaseEN", 6)

    def __init__(self, args, dictionary):
        """Constructor for BERTBaseEN specifications"""

        BERTBaseEN.debug.ldf("<< START >>")

        # Super module call
        super().__init__(dictionary)
        BERTBaseEN.debug.ldf("super()")

        # Instance variables
        self.args = args
        BERTBaseEN.debug.ldf("var dev")

        # Initialize model
        self.model = BertModel.from_pretrained("bert-base-cased")
        BERTBaseEN.debug.ldf("pretrained model")

        # Run model of CUDA
        self.model.cuda()
        BERTBaseEN.debug.ldf("model CUDA")
        BERTBaseEN.debug.ldf("<< END >>")

    def forward(self, src_token):
        """Forward function to specify forward propogation"""

        BERTBaseEN.debug.ldf("<< START >>")

        # Send data to device
        src_token = src_token.to(src_token.device).long()
        BERTBaseEN.debug.ldf("src_token")

        # Return logits from BERTBaseEN << BROKEN >>
        output = self.model(src_token)
        BERTBaseEN.debug.ldf("output")

        # Return result
        BERTBaseEN.debug.ldf("<< END >>")
        return output



class BERTInference(FairseqEncoder):
    """BERTBaseEN Model Declaration"""

    debug = Debug("BERTBaseEN", 6)

    def __init__(self, args, dictionary):
        """Constructor for BERTBaseEN specifications"""

        BERTBaseEN.debug.ldf("<< START >>")

        # Super module call
        super().__init__(dictionary)
        BERTBaseEN.debug.ldf("super()")

        # Instance variables
        self.args = args
        BERTBaseEN.debug.ldf("var dev")

        # Initialize model
        self.model = BertModel.from_pretrained("bert-base-cased")
        BERTBaseEN.debug.ldf("pretrained model")

        # Run model of CUDA
        self.model.cuda()
        BERTBaseEN.debug.ldf("model CUDA")
        BERTBaseEN.debug.ldf("<< END >>")

    def forward(self, src_token):
        """Forward function to specify forward propogation"""

        BERTBaseEN.debug.ldf("<< START >>")

        # Send data to device
        src_token = src_token.to(src_token.device).long().unsqueeze(0)
        BERTBaseEN.debug.ldf("src_token")

        # Return logits from BERTBaseEN << BROKEN >>
        output = self.model(src_token)
        BERTBaseEN.debug.ldf("output")

        # Return result
        BERTBaseEN.debug.ldf("<< END >>")
        return output


class SymphonyNetVanilla(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task):
        """SymphonyNet Structure Definition"""
        SymphonyNetVanilla.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        SymphonyNetVanilla.debug.ldf("super()")

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
        self.decoder_model = TransformerDecoderBuilder.from_kwargs(
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
        # TODO: check will the pad id be trained? (as TZ RZ YZ)
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

        # SymphonyNetVanilla.debug.ldf(
        #     f"{type(module)} | << START >>"
        # )

        # If the the given model is a linear or an embedding layer,
        # initialize weights with a mean of zero and a set std dev
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.dec_embed_dim**-0.5)
            # SymphonyNetVanilla.debug.ldf(
            #     "  0 Mean and Std Dev WEIGHT Init"
            # )

            # If the module is a linear layer with bias, set bias to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                # SymphonyNetVanilla.debug.ldf("  0 BIAS")

        # If the module is a LayerNorm, set bias to zero
        # and weight initialized to 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # SymphonyNetVanilla.debug.ldf(
            #     "  0 BIAS and 1 WEIGHT Fill"
            # )

        # SymphonyNetVanilla.debug.ldf("  << END >>")

    def forward(
        self,
        encoder_out,
        decoder_in,
        src_lengths=None,
        encoder_out_lengths=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNetVanilla.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            decoder_in=decoder_in,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            encoder_out_lengths=encoder_out_lengths,
        )
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
        return (evt_logits, dur_logits, trk_logits, ins_logits)

    # TODO: Understand how SymphonyNet masks work, including LengthMask and
    # TODO:   TriangularMask
    # TODO: Understand Permutation Invariant in code
    def extract_features(
        self,
        decoder_in,
        encoder_out=None,
        src_lengths=None,
        encoder_out_lengths=None,
    ):
        """Extract feature method"""

        SymphonyNetVanilla.debug.ldf("<< START >>")

        SymphonyNetVanilla.debug.ldf("process decoder_in")
        bsz, seq_len, ratio = decoder_in.size()

        SymphonyNetVanilla.debug.ldf("process encoder_out")
        enc_len, enc_bsz, embed_dim = encoder_out.size()

        SymphonyNetVanilla.debug.ldf("event embedding")
        evt_emb = self.wEvte(decoder_in[..., 0])

        SymphonyNetVanilla.debug.ldf("event mask")
        # if not mapping to pad, padding idx will only occur at last
        evton_mask = (
            decoder_in[..., 1]
            .ne(self.pad_idx)
            .float()[..., None]
            .to(decoder_in.device)
        )  # TODO: elaborate, why the mask is on the 2nd

        SymphonyNetVanilla.debug.ldf("duration embedding")
        tmp = self.wDure(decoder_in[..., 1])
        dur_emb = tmp * evton_mask

        SymphonyNetVanilla.debug.ldf("track embedding")
        tmp = self.wTrke(decoder_in[..., 2])
        trk_emb = tmp * evton_mask

        SymphonyNetVanilla.debug.ldf(
            "Calculating LengthMask for tgt"
        )
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

        SymphonyNetVanilla.debug.ldf(
            "Calculating LengthMask for src"
        )
        # Note: Calc LengthMask for encoder_out_lengths
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

        SymphonyNetVanilla.debug.ldf(
            "full mask for cross attention layer"
        )
        # WIP: Implement FullMask for Cross Attention layer
        full_mask = FullMask(N=seq_len, M=enc_len, device=decoder_in.device)

        SymphonyNetVanilla.debug.ldf("permutation invariant")
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

        SymphonyNetVanilla.debug.ldf("combine all midi features")
        x = (
            evt_emb + dur_emb + trk_emb + pos_emb
        )  # [bsz, seq_len, embedding_dim]

        SymphonyNetVanilla.debug.ldf("apply dropout")
        x = self.drop(x)

        SymphonyNetVanilla.debug.ldf("Model Computation")
        outputs = self.decoder_model(
            x=x,  # decoder_in shape: [batch_size, dec_length, embed_dim]
            memory=encoder_out,  # encoder_out shape: [batch_size, enc_length, embed_dim]
            x_mask=self.attn_mask,
            x_length_mask=len_mask,
            memory_mask=full_mask,  # WIP
            memory_length_mask=enc_len_mask,  # WIP
        )
        SymphonyNetVanilla.debug.ldf("apply layer norm")
        outputs = self.ln_f(outputs)

        SymphonyNetVanilla.debug.ldf("<< END >>")
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


class SymphonyNetVanillaNoTokenCalc(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task):
        """SymphonyNet Structure Definition"""
        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("super()")

        # Get the embedding dimensions for the SymphonyNet model
        self.dec_embed_dim = args.dec_embed_dim
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Decoder Dimension")

        # Set the EVENT, TRACK, and DURATION embedding layers
        self.wEvte = nn.Embedding(args.evt_voc_size, args.dec_embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.dec_embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.dec_embed_dim)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Embedding Layers")

        # Get the maximum number of tokens per sample
        self.max_pos = args.tokens_per_sample
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Maximum Tokens Per Sample")

        # Set permutation invariance configurations
        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos + 1, args.dec_embed_dim)
            self.wMpe = nn.Embedding(args.max_mea_pos + 1, args.dec_embed_dim)
            SymphonyNetVanillaNoTokenCalc.debug.ldf("perm_inv > 1")
        else:
            self.wpe = nn.Embedding(self.max_pos + 1, args.dec_embed_dim)
            SymphonyNetVanillaNoTokenCalc.debug.ldf("perm_inv == 0")

        # Setup dropout and layer normalization layers for reuse
        self.drop = nn.Dropout(args.dec_dropout)
        self.ln_f = nn.LayerNorm(args.dec_embed_dim, eps=1e-6)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Dropout & LayerNorm")

        # Build the decoder model
        self.model = TransformerEncoderBuilder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation="gelu",
            dropout=args.dec_dropout,
            attention_type="causal-linear",
        ).get()
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Decoder Model")

        # Generate attention mask
        self.attn_mask = TriangularCausalMask(self.max_pos)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Attention Mask")

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
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Output Layers")

        # Initialize the weights for the model
        self.apply(self._init_weights)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Init Weights")

        # Set zero embeddings for EVENT, DURATION, and TRACK for padding symbol
        # TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Zero Input Embedding Layers")

        # Set Zero embeddings for permutation invariance
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
            SymphonyNetVanillaNoTokenCalc.debug.ldf("perm_inv (zero) > 1")
        else:
            self.wpe.weight.data[0].zero_()
            SymphonyNetVanillaNoTokenCalc.debug.ldf("perm_inv (zero) == 1")

        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< END >>")

    def _init_weights(self, module):
        """Initialization Step"""

        # SymphonyNetVanillaNoTokenCalc.debug.ldf(
        #     f"{type(module)} | << START >>"
        # )

        # If the the given model is a linear or an embedding layer,
        # initialize weights with a mean of zero and a set std dev
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.dec_embed_dim**-0.5)
            # SymphonyNetVanillaNoTokenCalc.debug.ldf(
            #     "  0 Mean and Std Dev WEIGHT Init"
            # )

            # If the module is a linear layer with bias, set bias to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                # SymphonyNetVanillaNoTokenCalc.debug.ldf("  0 BIAS")

        # If the module is a LayerNorm, set bias to zero
        # and weight initialized to 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # SymphonyNetVanillaNoTokenCalc.debug.ldf(
            #     "  0 BIAS and 1 WEIGHT Fill"
            # )

        # SymphonyNetVanillaNoTokenCalc.debug.ldf("  << END >>")

    def forward(
        self,
        decoder_in,
        src_lengths=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features = self.extract_features(
            x=decoder_in,
            src_lengths=src_lengths,
        )
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Feature Extract")

        # Project the given features into the output layers
        # to get the logit projections of EVENT, DURATION
        # TRACK, and PREDICTION
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Final Projection")
        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< END >>")

        # Return the logits for the EVENT, DURATION, TRACK, and INSTRUMENT
        return (evt_logits, dur_logits, trk_logits, ins_logits)

    def extract_features(self, x, src_lengths=None):
        """Extract feature method"""

        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< START >>")

        # Permutate the tensor
        x = x.permute(1, 0, 2)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Input Permute")

        # Breaking down the dimensions of the input seq
        bsz, seq_len, dim = x.size()
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Dimension Breakdown")

        # Create pad masking
        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Create Pad Masking")

        # Fill masking with length in mind
        if src_lengths is not None:
            len_mask = LengthMask(src_lengths, max_len=seq_len, device=x.device)
            SymphonyNetVanillaNoTokenCalc.debug.ldf("SRC LENGTH Filled Mask")
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1), max_len=seq_len, device=x.device
            )
            SymphonyNetVanillaNoTokenCalc.debug.ldf(
                "PAD_MASK Length Filled Mask"
            )

        # Pass to model
        outputs = self.model(x, self.attn_mask, len_mask)
        SymphonyNetVanillaNoTokenCalc.debug.ldf(
            "Model Transformer Processing"
        )

        # Pass to linear layer
        outputs = self.ln_f(outputs)
        SymphonyNetVanillaNoTokenCalc.debug.ldf("Linear Processing")

        # Return output
        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< END >>")
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
        SymphonyNetVanillaNoTokenCalc.debug.ldf("<< max_positions >>")
        return 4096


class SymphonyNetInference(FairseqDecoder):
    """SymphonyNet Model Specification"""

    debug = Debug("SymphonyNet", 2)

    def __init__(self, args, task):
        """SymphonyNet Structure Definition"""
        SymphonyNetInference.debug.ldf("<< START >>")

        # Super call for a FairseqDecoder
        # TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        SymphonyNetInference.debug.ldf("super()")

        # Get the embedding dimensions for the SymphonyNet model
        self.dec_embed_dim = args.dec_embed_dim
        SymphonyNetInference.debug.ldf("Decoder Dimension")

        # Set the EVENT, TRACK, and DURATION embedding layers
        self.wEvte = nn.Embedding(args.evt_voc_size, args.dec_embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.dec_embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.dec_embed_dim)
        SymphonyNetInference.debug.ldf("Embedding Layers")

        # Get the maximum number of tokens per sample
        self.max_pos = args.tokens_per_sample
        SymphonyNetInference.debug.ldf("Maximum Tokens Per Sample")

        # Set permutation invariance configurations
        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos + 1, args.dec_embed_dim)
            self.wMpe = nn.Embedding(args.max_mea_pos + 1, args.dec_embed_dim)
            SymphonyNetInference.debug.ldf("perm_inv > 1")
        else:
            self.wpe = nn.Embedding(self.max_pos + 1, args.dec_embed_dim)
            SymphonyNetInference.debug.ldf("perm_inv == 0")

        # Setup dropout and layer normalization layers for reuse
        self.drop = nn.Dropout(args.dec_dropout)
        self.ln_f = nn.LayerNorm(args.dec_embed_dim, eps=1e-6)
        SymphonyNetInference.debug.ldf("Dropout & LayerNorm")

        # Build the recurrent decoder model
        # Note: RecurrentDecoder is able to utilize the previous context to predict the next token in order
        self.decoder_model = RecurrentDecoderBuilder.from_kwargs(
            n_layers=args.dec_num_layers,
            n_heads=args.dec_num_attention_heads,
            query_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            value_dimensions=args.dec_embed_dim // args.dec_num_attention_heads,
            feed_forward_dimensions=4 * args.dec_embed_dim,
            activation="gelu",
            dropout=args.dec_dropout,
            self_attention_type="causal-linear",
            cross_attention_type="full",  # Fully masked so that each domain can be merged
        ).get()
        SymphonyNetInference.debug.ldf("Decoder Model")

        # Generate attention mask
        self.attn_mask = TriangularCausalMask(self.max_pos)
        SymphonyNetInference.debug.ldf("Attention Mask")

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
        SymphonyNetInference.debug.ldf("Output Layers")

        # Initialize the weights for the model
        self.apply(self._init_weights)
        SymphonyNetInference.debug.ldf("Init Weights")

        # Set zero embeddings for EVENT, DURATION, and TRACK for padding symbol
        # TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        SymphonyNetInference.debug.ldf("Zero Input Embedding Layers")

        # Set Zero embeddings for permutation invariance
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
            SymphonyNetInference.debug.ldf("perm_inv (zero) > 1")
        else:
            self.wpe.weight.data[0].zero_()
            SymphonyNetInference.debug.ldf("perm_inv (zero) == 1")

        SymphonyNetInference.debug.ldf("<< END >>")

    def _init_weights(self, module):
        """Initialization Step"""

        SymphonyNetInference.debug.ldf(f"{type(module)} | << START >>")

        # If the the given model is a linear or an embedding layer,
        # initialize weights with a mean of zero and a set std dev
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.dec_embed_dim**-0.5)
            SymphonyNetInference.debug.ldf("  0 Mean and Std Dev WEIGHT Init")

            # If the module is a linear layer with bias, set bias to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                SymphonyNetInference.debug.ldf("  0 BIAS")

        # If the module is a LayerNorm, set bias to zero
        # and weight initialized to 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            SymphonyNetInference.debug.ldf("  0 BIAS and 1 WEIGHT Fill")

        SymphonyNetInference.debug.ldf("  << END >>")

    def forward(
        self,
        encoder_out,
        decoder_in,
        src_lengths=None,
        encoder_out_lengths=None,
        state=None,
    ):
        """SymphonyNet's Forward Function"""

        SymphonyNetInference.debug.ldf("<< START >>")

        # Extract features from the given encoder's output, and decoder_input
        features, memory = self.extract_features(
            decoder_in=decoder_in,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            encoder_out_lengths=encoder_out_lengths,
            state=state,
        )
        SymphonyNetInference.debug.ldf("Feature Extract")

        # Project the given features into the output layers
        # to get the logit projections of EVENT, DURATION
        # TRACK, and PREDICTION
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)
        SymphonyNetInference.debug.ldf("Final Projection")
        SymphonyNetInference.debug.ldf("<< END >>")

        # Return the logits for the EVENT, DURATION, TRACK, and INSTRUMENT
        return (evt_logits, dur_logits, trk_logits, ins_logits), memory

    # TODO: Understand how SymphonyNet masks work, including LengthMask
    # TODO:   and TriangularMask
    # TODO: Understand Permutation Invariant in code
    def extract_features(
        self,
        decoder_in,
        encoder_out=None,
        src_lengths=None,
        encoder_out_lengths=None,
        state=None,
    ):
        """Extract feature method"""

        SymphonyNetInference.debug.ldf("<< START >>")

        SymphonyNetInference.debug.ldf("process decoder_in")
        bsz, seq_len, ratio = decoder_in.size()

        SymphonyNetInference.debug.ldf("process encoder_out")
        enc_len, enc_bsz, embed_dim = encoder_out.size()

        SymphonyNetInference.debug.ldf("event embedding")
        evt_emb = self.wEvte(decoder_in[..., 0])

        SymphonyNetInference.debug.ldf("event mask")
        # if not mapping to pad, padding idx will only occur at last
        evton_mask = (
            decoder_in[..., 1]
            .ne(self.pad_idx)
            .float()[..., None]
            .to(decoder_in.device)
        )  # TODO: elaborate, why the mask is on the 2nd

        SymphonyNetInference.debug.ldf("duration embedding")
        tmp = self.wDure(decoder_in[..., 1])
        dur_emb = tmp * evton_mask

        SymphonyNetInference.debug.ldf("track embedding")
        tmp = self.wTrke(decoder_in[..., 2])
        trk_emb = tmp * evton_mask

        SymphonyNetInference.debug.ldf("Calculating LengthMask for tgt")
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

        SymphonyNetInference.debug.ldf("Calculating LengthMask for src")
        # Note: Calc LengthMask for encoder_out_lengths
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

        SymphonyNetInference.debug.ldf("full mask for cross attention layer")
        # WIP: Implement FullMask for Cross Attention layer
        full_mask = FullMask(N=seq_len, M=enc_len, device=decoder_in.device)

        SymphonyNetInference.debug.ldf("permutation invariant")
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

        SymphonyNetInference.debug.ldf("combine all midi features")
        x = (
            evt_emb + dur_emb + trk_emb + pos_emb
        )  # [bsz, seq_len, embedding_dim]

        SymphonyNetInference.debug.ldf("apply dropout")
        x = self.drop(x)

        SymphonyNetInference.debug.ldf("Model Computation")
        outputs, state = self.decoder_model(
            x=x.squeeze(0),
            memory=encoder_out,
            memory_length_mask=None,
            state=state,
        )
        SymphonyNetInference.debug.ldf("apply layer norm")
        outputs = self.ln_f(outputs)
        SymphonyNetInference.debug.ldf("<< END >>")
        return outputs, state

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
        SymphonyNetInference.debug.ldf("<< max_positions >>")
        return 10000  # WIP: Should change later


class IntermediarySection(FairseqEncoder):
    """Intermediary section to help translate the two modalities"""

    debug = Debug("IntermediarySection", 7)

    def __init__(self, args):
        """Constructor for intermediary section"""
        IntermediarySection.debug.ldf("<< START >>")

        # Super call
        super().__init__(dictionary=None)
        IntermediarySection.debug.ldf("super()")

        # Linear layer section specification
        self.linear_layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
        )
        IntermediarySection.debug.ldf("linear_layers")
        IntermediarySection.debug.ldf("<< END >>")

    def forward(self, x):
        """Forward function to process input"""
        IntermediarySection.debug.ldf("<< forward >>")
        return self.linear_layers(x)
