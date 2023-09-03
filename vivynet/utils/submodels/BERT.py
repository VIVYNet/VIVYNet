# Fairseq Imports
from fairseq.models import FairseqEncoder

# HuggingFace Imports
from transformers import BertModel, BertConfig

# Debug Imports
from vivynet.utils.debug import Debug


class BERT(FairseqEncoder):
    """BERTBaseEN Model Declaration"""

    debug = Debug("BERTBaseEN", 6)

    def __init__(self, args, task, multilingual=False, inference=False):
        """Constructor for BERTBaseEN specifications"""

        BERT.debug.ldf("<< START >>")

        # Super module call
        super().__init__(task.source_dictionary)
        BERT.debug.ldf("super()")

        # Instance variables
        self.args = args
        self.inference = inference
        BERT.debug.ldf("var dev")

        # Get huggingface model name based on multilingual value
        if multilingual:
            hf_model_name = "bert-base-multilingual-cased"
        else:
            hf_model_name = "bert-base-cased"

        # Initialize model
        if args.pt_enc:
            self.model = BertModel.from_pretrained(hf_model_name)
        else:
            config = BertConfig.from_pretrained(hf_model_name)
            self.model = BertModel(config)
        BERT.debug.ldf("pretrained model")

        # Run model of CUDA
        self.model.cuda()
        BERT.debug.ldf("model CUDA")
        BERT.debug.ldf("<< END >>")

    @staticmethod
    def add_args(parser, check):
        """Method to add arguments for this specific model"""
        return

    def forward(self, src_token):
        """Forward function to specify forward propogation"""

        BERT.debug.ldf("<< START >>")

        # Send data to device
        if self.inference:
            src_token = src_token.to(src_token.device).long().unsqueeze(0)
        else:
            src_token = src_token.to(src_token.device).long()
        BERT.debug.ldf("src_token")

        # Return logits from BERTBaseEN << BROKEN >>
        output = self.model(src_token)
        BERT.debug.ldf("output")

        # Return result
        BERT.debug.ldf("<< END >>")
        return output
