# Fairseq Imports
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model_architecture,
    register_model,
)

# Submodule imports
from vivynet.utils.VIVYNetSubModels import (
    BERTBaseMulti,
    SymphonyNetVanillaNoTokenCalc,
    IntermediarySection,
)

# Torch Imports
import torch

# Debug imports
from vivynet.utils.debug import Debug


@register_model("vivy_van_ae")
class VIVYNetVanAE(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

    # DEBUG
    debug = Debug("VIVYNet", 3)

    @staticmethod
    def add_args(parser):
        """Argument Definition class"""
        VIVYNetVanAE.debug.ldf("<< START >>")

        # Shorten Method
        parser.add_argument("--shorten_method", type=str, metavar="N")
        VIVYNetVanAE.debug.ldf("shorten_method")

        # Shorten Data Split List
        parser.add_argument("--shorten_data_split_list", type=str, metavar="N")
        VIVYNetVanAE.debug.ldf("shorten_data_split_list")

        # Token Per Sample
        parser.add_argument("--tokens_per_sample", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("tokens_per_sample")

        # Sample Break Mode
        parser.add_argument("--sample_break_mode", type=str, metavar="N")
        VIVYNetVanAE.debug.ldf("sample_break_mode")

        # Ratio
        parser.add_argument("--ratio", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("ratio")

        # Sample Overlap Rate
        parser.add_argument("--sample_overlap_rate", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("sample_overlap_rate")

        # Permutation invariance
        parser.add_argument("--perm_inv", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("perm_inv")

        # Event Token Size
        parser.add_argument("--evt_voc_size", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("evt_voc_size")

        # Track Token Size
        parser.add_argument("--trk_voc_size", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("trk_voc_size")

        # Duration Vocab Size
        parser.add_argument("--dur_voc_size", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("dur_voc_size")

        # Instrument Vocab Size
        parser.add_argument("--ins_voc_size", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("ins_voc_size")

        # Maximum Relative Position
        parser.add_argument("--max_rel_pos", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("max_rel_pos")

        # Maximum Measure Count within a Sample
        parser.add_argument("--max_mea_pos", type=int, metavar="N")
        VIVYNetVanAE.debug.ldf("max_mea_pos")

        # Decoder Embedding Dimension
        parser.add_argument(
            "--dec-embed-dim",
            type=int,
            metavar="N",
            help="embedding dimension",
        )
        VIVYNetVanAE.debug.ldf("dec-embed-dim")

        # Decoder Attention Head Numbers
        parser.add_argument(
            "--dec-num-attention-heads",
            type=int,
            metavar="N",
            help="num attention heads",
        )
        VIVYNetVanAE.debug.ldf("dec-num-attention-heads")

        # Number Decoder Layers
        parser.add_argument(
            "--dec-num-layers", type=int, metavar="N", help="num layers"
        )
        VIVYNetVanAE.debug.ldf("dec-num-layers")

        # Decoder Dropout
        parser.add_argument(
            "--dec-dropout",
            type=float,
            metavar="D",
            help="dropout probability for all fully connected layers "
            "in the embeddings, encoder, and pooler",
        )
        VIVYNetVanAE.debug.ldf("dec-dropout")

        # Freeze encoder
        parser.add_argument(
            "--freeze_enc",
            type=int,
            metavar="N",
            help="Freeze pretrained Encoder layers",
        )
        VIVYNetVanAE.debug.ldf("freeze_enc")

        # Freeze decoder
        parser.add_argument(
            "--freeze_dec",
            type=int,
            metavar="N",
            help="Freeze pretrained Decoder layers",
        )
        VIVYNetVanAE.debug.ldf("freeze_dec")

        VIVYNetVanAE.debug.ldf("<< END >>")

    @classmethod
    def build_model(cls, args, task):
        """Build model function"""

        VIVYNetVanAE.debug.ldf("<< START >>")

        #
        #   BERTBaseMulti BUILDING
        #   region
        #

        # Create BERTBaseMulti model
        bert = BERTBaseMulti(args=args, dictionary=task.source_dictionary)
        VIVYNetVanAE.debug.ldf("Model Creation: BERTBaseMulti")

        # Freezing the Encoder layers and load pretrained weights
        if args.freeze_enc == 1:
            # Freezing BERTBaseMulti
            for name, param in bert.named_parameters():
                param.requires_grad = False
        VIVYNetVanAE.debug.ldf("Freezing pretrained Encoder layers")

        #   endregion

        #
        #   SymphonyNet BUILDING
        #   region
        #

        # Create SymphonyNet model
        symphony_net = SymphonyNetVanillaNoTokenCalc(args=args, task=task)
        VIVYNetVanAE.debug.ldf("Model Creation: SymphonyNet")

        # Get the checkpoint
        checkpoint = torch.load(
            "../symphonynet/ckpt/"
            + "checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
        )
        VIVYNetVanAE.debug.ldf("Checkpoint loading")

        # WIP: Currently unable to transfer weights since the original
        #      checkpoint has different dimension due to
        #      being trained on a different dataset.

        # Freezing the Decoder layers and load pretrained weights
        if args.freeze_dec == 1:
            # Freezing self-attentions
            for _, param in symphony_net.named_parameters():
                param.requires_grad = False

            # Zipping two models param dicts
            pretrained_params = []
            for param in symphony_net.state_dict():
                if not ("cross_attention" in param or "norm3" in param):
                    pretrained_params.append(param)
            VIVYNetVanAE.debug.ldf("Weight targeting copy")

            # Weight copying
            VIVYNetVanAE.debug.ldf("Proceed loading Decoder pretrained weights")
            with torch.no_grad():
                for param1, param2 in zip(
                    pretrained_params, checkpoint["model"]
                ):
                    symphony_net.state_dict()[param1].copy_(
                        checkpoint["model"][param2]
                    )
                    VIVYNetVanAE.debug.ldf(f"Loading {param1}")
            VIVYNetVanAE.debug.ldf("Loading Finished!")

        #   endregion

        #
        #   Intermediary BUILDING
        #   region
        #

        # Create intermediary layer
        intermediary = IntermediarySection(args)
        VIVYNetVanAE.debug.ldf("Model Creation: Intermediary Layer")

        #   endregion

        # Return compiled vivynet
        VIVYNetVanAE.debug.ldf("<< END >>")
        return VIVYNetVanAE(bert, symphony_net, intermediary)

    def __init__(self, encoder, decoder, intermediary):
        """Constructor for the VIVYNet model"""

        VIVYNetVanAE.debug.ldf("<< START >>")

        # Retrieves attributes
        super().__init__(encoder, decoder)
        VIVYNetVanAE.debug.ldf("super()")

        # Create instance variables based on parameters given
        self.encoder = encoder
        self.decoder = decoder
        self.intermediary = intermediary
        VIVYNetVanAE.debug.ldf("models")

        # Put models into train mode
        self.encoder.train()
        VIVYNetVanAE.debug.ldf("encoder.train")
        VIVYNetVanAE.debug.ldf("<< END >>")

    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        prev_output_tokens_lengths=None,
    ):
        """Forward propagation method"""

        VIVYNetVanAE.debug.ldf("<< START >>")

        # Clear previously calculated gradients
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.intermediary.zero_grad()
        VIVYNetVanAE.debug.ldf("zero_grad()")

        # Get loss and the logits from the model
        enc_output = self.encoder(src_tokens.reshape(-1, 1))
        VIVYNetVanAE.debug.ldf("res 1")

        # Process BERTBaseMulti out from 768 dimension to 512 dimension
        intermediate = self.intermediary(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNetVanAE.debug.ldf(f"res 2 : {intermediate.shape} : {src_lengths}")

        # Get overall features from decoder
        features = self.decoder(decoder_in=intermediate)
        VIVYNetVanAE.debug.ldf("res 3")

        # Return the logits
        VIVYNetVanAE.debug.ldf("<< END >>")
        return features

    @property
    def supported_targets(self):
        """Supported Targets Property"""
        VIVYNetVanAE.debug.ldf("<< supported_targets >>")
        return {"future"}


@register_model_architecture("vivy_van_ae", "vivy_vanae")
def train_VanAE(args):
    """Train function"""

    # DEBUG
    debug = Debug("train", 4)
    debug.ldf("<< train >>")

    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 16)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)
