# Fairseq Imports
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model_architecture,
    register_model,
)

# Submodule imports
from vivynet.utils.submodels import *

# Torch Imports
import torch

# Debug imports
from vivynet.utils.debug import Debug


@register_model("vivy")
class VIVYNet(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

    # DEBUG
    debug = Debug("VIVYNet", 3)

    # Mappings
    ENCODER_MAPS = {
        "BERT_multilingual": BERT,
        "BERT_english": BERT,
    }
    DECODER_MAPS = {
        "SymphonyNet_Encoder": SymphonyNetEncoder,
        "SymphonyNet_Vanilla": SymphonyNetVanilla,
    }
    LATENT_MAPS = {
        "linear": LatentLinear,
        "transformer_encoder": LatentTransformerEncoder,
    }

    @staticmethod
    def add_args(parser):
        """Argument Definition class"""
        VIVYNet.debug.ldf("<< START >>")

        #
        #   *** Data Specification ***
        #   region
        #

        # Token Per Sample
        parser.add_argument("--tokens_per_sample", type=int, metavar="N")
        VIVYNet.debug.ldf("tokens_per_sample")

        # Permutation invariance
        parser.add_argument("--perm_inv", type=int, metavar="N")
        VIVYNet.debug.ldf("perm_inv")

        #   endregion

        #
        #   *** VIVYNet Specifications ***
        #   region
        #

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

        # endregion

        #
        #   *** Encoder, Decoder, and Latent Specifications ***
        #   region
        #

        # Specify which encoder to load
        parser.add_argument(
            "--enc",
            type=str,
            metavar="N",
            required=True,
            choices=list(VIVYNet.ENCODER_MAPS.keys()),
            help="Specify which latent encoder model to load",
        )
        VIVYNet.debug.ldf("enc")

        # Specify which decoder to load
        parser.add_argument(
            "--dec",
            type=str,
            metavar="N",
            required=True,
            choices=list(VIVYNet.DECODER_MAPS.keys()),
            help="Specify which latent decoder model to load",
        )
        VIVYNet.debug.ldf("dec")

        # Specify which latent model to load
        parser.add_argument(
            "--latent",
            type=str,
            metavar="N",
            required=True,
            choices=list(VIVYNet.LATENT_MAPS.keys()),
            help="Specify which latent space model to load",
        )
        VIVYNet.debug.ldf("latent")

        # Load pretrained weights encoder
        parser.add_argument(
            "--pt_enc",
            type=int,
            metavar="N",
            help="Load pretrained encoder layers",
        )
        VIVYNet.debug.ldf("pt_enc")

        # Freeze encoder
        parser.add_argument(
            "--freeze_enc",
            type=int,
            metavar="N",
            help="Freeze Encoder layers",
        )
        VIVYNet.debug.ldf("freeze_enc")

        # Load pretrained weights decoder
        parser.add_argument(
            "--pt_dec",
            type=int,
            metavar="N",
            help="Load pretrained decoder layers",
        )
        VIVYNet.debug.ldf("pt_dec")

        # Freeze decoder
        parser.add_argument(
            "--freeze_dec",
            type=int,
            metavar="N",
            help="Freeze pretrained Decoder layers",
        )
        VIVYNet.debug.ldf("freeze_dec")

        #   endregion

        # Define a helper function for checking if the argument is already
        # in place
        check = lambda parser, name: name in (a.dest for a in parser._actions)

        #   *** Encoder Specific Specifications ***
        for i in VIVYNet.ENCODER_MAPS:
            VIVYNet.ENCODER_MAPS[i].add_args(parser, check)

        #   *** Decoder Specific Specifications ***
        for i in VIVYNet.DECODER_MAPS:
            VIVYNet.DECODER_MAPS[i].add_args(parser, check)

        #   *** Latent Specific Specifications ***
        for i in VIVYNet.LATENT_MAPS:
            VIVYNet.LATENT_MAPS[i].add_args(parser, check)

        # End argument parsing
        VIVYNet.debug.ldf("<< END >>")

    @classmethod
    def build_model(cls, args, task):
        """Build model function"""

        VIVYNet.debug.ldf("<< START >>")

        #
        #   ENCODER BUILDING
        #   region
        #

        # Create BERTBaseEN model
        encoder_model = VIVYNet.ENCODER_MAPS[args.enc](args=args, task=task)
        VIVYNet.debug.ldf("Model Creation: BERTBaseEN")

        # Freezing the Encoder layers and load pretrained weights
        if args.freeze_enc == 1:
            # Freezing BERTBaseEN
            for name, param in encoder_model.named_parameters():
                param.requires_grad = False
            VIVYNet.debug.ldf("Froze pretrained Encoder layers")

        #   endregion

        #
        #   SymphonyNet BUILDING
        #   region
        #

        # Create SymphonyNet model
        decoder_model = VIVYNet.DECODER_MAPS[args.dec](args=args, task=task)
        VIVYNet.debug.ldf("Model Creation: SymphonyNet")

        # Conduct these actions based on certain decoder types
        if "SymphonyNet" in args.dec:
            # Load pretrained weights for decoder
            if args.pt_dec:
                # Get the checkpoint
                checkpoint = torch.load(
                    "../symphonynet/ckpt/"
                    + "checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
                )
                VIVYNet.debug.ldf("Checkpoint loading")

                # Zipping two models param dicts
                pretrained_params = []
                for param in decoder_model.state_dict():
                    if not ("cross_attention" in param or "norm3" in param):
                        pretrained_params.append(param)
                VIVYNet.debug.ldf("Weight Targeting Copy")

                # Weight copying
                VIVYNet.debug.ldf("Proceed Loading Decoder Pretrained Weights")
                with torch.no_grad():
                    for param1, param2 in zip(
                        pretrained_params, checkpoint["model"]
                    ):
                        decoder_model.state_dict()[param1].copy_(
                            checkpoint["model"][param2]
                        )
                        VIVYNet.debug.ldf(f"Loading {param1}")

            # Freezing the Decoder layers
            if args.freeze_dec == 1:
                # Freezing self-attentions
                VIVYNet.debug.ldf("Freezing pretrained Decoder layers")
                for name, param in decoder_model.named_parameters():
                    if (
                        "self_attention"
                        or "wEvte.weight"
                        or "wTrke.weight"
                        or "wDure.weight"
                        or "wRpe.weight"
                        or "wMpe.weight" in name
                    ):
                        param.requires_grad = False
                VIVYNet.debug.ldf("Froze Decoder")

        #   endregion

        #
        #   Intermediary BUILDING
        #   region
        #

        # Create intermediary layer if it is specified to LatentLinear
        if VIVYNet.LATENT_MAPS[args.latent] == LatentLinear:
            latent = LatentLinear(
                input_dim=args.latent_input_dim,
                hidden_dim=args.latent_hidden_dim,
                output_dim=args.latent_output_dim,
                hidden_layers=args.latent_hidden_layers,
                dropout_rate=args.latent_dropout_rate,
            )
            VIVYNet.debug.ldf("Model Creation: Latent Layer")

        # Create intermediary layer if it is specified to
        # LatentTransformerDecoder
        if VIVYNet.LATENT_MAPS[args.latent] == LatentTransformerEncoder:
            latent = LatentTransformerEncoder(args=args, task=task)
            VIVYNet.debug.ldf("Model Creation: Latent Transformer Decoder")

        #   endregion

        # Return
        VIVYNet.debug.ldf("<< END >>")
        return VIVYNet(encoder_model, decoder_model, latent)

    def __init__(self, encoder, decoder, latent):
        """Constructor for the VIVYNet model"""

        VIVYNet.debug.ldf("<< START >>")

        # Retrieves attributes
        super().__init__(encoder, decoder)
        VIVYNet.debug.ldf("super()")

        # Create instance variables based on parameters given
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent
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
        self.decoder.zero_grad()
        self.latent.zero_grad()
        VIVYNet.debug.ldf("encoder.zero_grad()")

        # Get loss and the logits from the model
        enc_output = self.encoder(src_tokens)
        VIVYNet.debug.ldf("res 1")
        
        # Intermediary layer pass
        latent_output = self.latent(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNet.debug.ldf(f"res 2 : {latent_output.shape} : {src_lengths}")

        # Get overall features from decoder
        decoder_output = self.decoder(
            encoder_out=latent_output,
            decoder_in=prev_output_tokens,
            src_lengths=prev_output_tokens_lengths,
            encoder_out_lengths=None,
        )
        VIVYNet.debug.ldf("res 3")

        # Return the logits
        VIVYNet.debug.ldf("<< END >>")
        return decoder_output

    @property
    def supported_targets(self):
        """Supported Targets Property"""
        VIVYNet.debug.ldf("<< supported_targets >>")
        return {"future"}


@register_model_architecture("vivy", "vivy_transformer")
def train_transformer(args):
    """Train function"""
    pass
