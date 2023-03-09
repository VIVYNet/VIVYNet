from fairseq.models import FairseqEncoder, FairseqEncoderModel, FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq.tasks import LegacyFairseqTask, register_task
from transformers import BertTokenizer, BertModel, BertConfig
from encoder import BertEncoder
from decoder.symphony_net.src.fairseq.linear_transformer_inference import linear_transformer_multi
import torch
import torch.nn

class Bert2SymphonyNet(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
    
    @staticmethod
    def add_args(parser):
        # WORK IN PROCESS
        pass

    @classmethod
    def build_model(cls, args, task):
        # WORK IN PROCESS
        encoder = BertEncoder.BertEncoder(args.dropout, args.fn_dim)
        decoder = linear_transformer_multi.LinearTransformerMultiHeadDecoder(args, task)
        return Bert2SymphonyNet(encoder, decoder)