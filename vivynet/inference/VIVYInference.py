# Fairseq Imports
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model_architecture,
    register_model,
)

# Torch Imports
import torch

# Submodels imports
from vivynet.utils.submodels import *

# Debug imports
from vivynet.utils.debug import Debug


@register_model("vivy")
class VIVYNet(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""

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

    # DEBUG
    debug = Debug("VIVYNet", 3)

    @classmethod
    def build_model(cls, args, task):
        """Build model function"""

        VIVYNet.debug.ldf("<< START >>")

        #
        #   ENCODER BUILDING
        #   region
        #

        # Create BERTBaseEN model
        encoder_model = VIVYNet.ENCODER_MAPS[args.enc](args=args, task=task, inference=True)
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
        decoder_model = VIVYNet.DECODER_MAPS[args.dec](args=args, task=task, inference=True)
        VIVYNet.debug.ldf("Model Creation: SymphonyNet")

        # Conduct these actions based on certain decoder types
        if "SymphonyNet" in args.dec:
            # Load pretrained weights for decoder
            if args.pt_dec:
                # Get the checkpoint
                checkpoint = torch.load(
                    "symphonynet/ckpt/checkpoint_last_linear_4096_chord_bpe_hardloss1_PI2.pt"
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
            latent = LatentTransformerEncoder(args=args, task=task, inference=True)
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
        state=None,
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
        intermediate, _ = self.latent(enc_output[0])
        src_lengths = len(src_tokens)
        VIVYNet.debug.ldf(f"res 2 : {intermediate.shape} : {src_lengths}")

        # Get overall features from decoder
        features, state = self.decoder(
            encoder_out=intermediate.unsqueeze(1),
            decoder_in=prev_output_tokens,
            src_lengths=prev_output_tokens_lengths,
            encoder_out_lengths=src_lengths,
        )
        VIVYNet.debug.ldf("res 3")

        # Return the logits
        VIVYNet.debug.ldf("<< END >>")
        return features, state

    @property
    def supported_targets(self):
        """Supported Targets Property"""
        VIVYNet.debug.ldf("<< supported_targets >>")
        return {"future"}


@register_model_architecture("vivy", "vivy_transformer")
def train(args):
    """Train function"""

    # DEBUG
    debug = Debug("train", 4)
    debug.ldf("<< train >>")

    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 16)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)
