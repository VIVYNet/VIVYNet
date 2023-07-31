# Fairseq Imports
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model_architecture,
    register_model,
)

# Submodule imports
from vivynet.utils.VIVYNetSubModels import (
    BERT,
    SymphonyNetVanillaAE,
    IntermediarySection,
)

# Torch Imports
import torch

# Debug imports
from vivynet.utils.debug import Debug


@register_model("vivy")
class VIVYNet(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

    # DEBUG
    debug = Debug("VIVYNet", 3)

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

        #
        #   BERT BUILDING
        #   region
        #

        # Create BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        VIVYNet.debug.ldf("Model Creation: BERT")

        # Freezing the Encoder layers and load pretrained weights
        if args.freeze_enc == 1:
            # Freezing BERT
            VIVYNet.debug.ldf("Freezing pretrained Encoder layers")
            for name, param in bert.named_parameters():
                param.requires_grad = False

        #   endregion

        #
        #   SymphonyNet BUILDING
        #   region
        #

        # Create SymphonyNet model
        symphony_net = SymphonyNetVanillaAE(args=args, task=task)
        VIVYNet.debug.ldf("Model Creation: SymphonyNet")

        # Get the checkpoint
        checkpoint = torch.load(
            "../symphonynet/ckpt/"
            + "checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
        )
        VIVYNet.debug.ldf("Checkpoint loading")

        # Freezing the Decoder layers and load pretrained weights
        if args.freeze_dec == 1:
            # Freezing self-attentions
            VIVYNet.debug.ldf("Freezing pretrained Decoder layers")
            for name, param in symphony_net.named_parameters():
                if (
                    "self_attention"
                    or "wEvte.weight"
                    or "wTrke.weight"
                    or "wDure.weight"
                    or "wRpe.weight"
                    or "wMpe.weight" in name
                ):
                    param.requires_grad = False

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
                    # print(param1 + " " + param2)
                    symphony_net.state_dict()[param1].copy_(
                        checkpoint["model"][param2]
                    )
                    VIVYNet.debug.ldf(f"Loading {param1}")
            VIVYNet.debug.ldf("Loading Finished!")

        #   endregion

        #
        #   Intermediary BUILDING
        #   region
        #

        # Create intermediary layer
        intermediary = IntermediarySection(args)
        VIVYNet.debug.ldf("Model Creation: Intermediary Layer")

        #   endregion

        # Return
        VIVYNet.debug.ldf("<< END >>")
        return VIVYNet(bert, symphony_net, intermediary)

    def __init__(self, encoder, decoder, intermediary):
        """Constructor for the VIVYNet model"""

        VIVYNet.debug.ldf("<< START >>")

        # Retrieves attributes
        super().__init__(encoder, decoder)
        VIVYNet.debug.ldf("super()")

        # Create instance variables based on parameters given
        self.encoder = encoder
        self.decoder = decoder
        self.intermediary = intermediary
        VIVYNet.debug.ldf("models")

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

        # Clear previously calculated gradients
        self.encoder.zero_grad()
        VIVYNet.debug.ldf("encoder.zero_grad()")

        # Get loss and the logits from the model
        enc_output = self.encoder(src_tokens)
        VIVYNet.debug.ldf("res 1")

        # Intermediary layer pass
        intermediate = self.intermediary(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNet.debug.ldf(f"res 2 : {intermediate.shape} : {src_lengths}")

        # Get overall features from decoder
        features = self.decoder(
            encoder_out=intermediate,
            decoder_in=prev_output_tokens,
            src_lengths=prev_output_tokens_lengths,
            encoder_out_lengths=src_lengths,
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


@register_model_architecture("vivy", "vivy_transformer")
def train_transformer(args):
    """Train function"""

    # DEBUG
    debug = Debug("train", 4)
    debug.ldf("<< train >>")

    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 32)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)
