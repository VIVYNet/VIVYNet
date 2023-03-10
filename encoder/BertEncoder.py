from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, register_model_architecture
from fairseq.tasks import LegacyFairseqTask, FairseqTask, register_task
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn

class BertEncoder(FairseqEncoder):
    def __init__(self, fn_dim, dropout):
        super().__init__()
        self.config = BertConfig()
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = torch.nn.Dropout(dropout)
        self.fn = torch.nn.Linear(self.config.hidden_size, fn_dim)

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        output = self.model(src_tokens)[0]
        output = self.dropout(output)
        output = self.fn(output)
        return output

### BertEncoderModel and ExtractFeature are only used for testing 
@register_model('BertEncoderModel')    
class BERTEncoderModel(FairseqEncoderModel):
    def __init__(self, bert, input_vocab):
        super(BERTEncoderModel).__init__()
        self.bert = bert
        self.input_vocab = input_vocab

    @staticmethod
    def add_args(parser):
        parser.add_argument('--dropout', type = float, metavar = 'D',
                            help = 'dropout rate between the model and the connected layer')
        parser.add_argument('--fn-dim', type = int, metavar = "N",
                            help = 'dimensionality for forward layer')
    
    @classmethod
    def build_model(cls, args, task):
        bert = BertEncoder(args.dropout, args.fn_dim)
        return BERTEncoderModel(bert, task.source_dictionary)

    def forward(self, src_tokens, src_lengths, **kwargs):
        # batch_size, max_src_len = src_tokens.size()
        output = self.bert(src_tokens)
        return output

@register_model_architecture("BertEncoderModel","Default")
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.fn_dim = getattr(args, 'fn_dim', 768)

### This task may not work since BertEncoder is mainly used for combining with Decoder and does not have
### certain target dataset for training
@register_task('extract_feature')
class ExtractFeature(LegacyFairseqTask):
    
    @staticmethod
    def add_args(parser):
        pass
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        pass

    def __init__(self, args, input_vocab, target_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
    
    def load_dataset(self, split, **kwargs):
        pass

    def source_dictionary(self):
        return self.input_vocab

    def target_dictionary(self):
        return self.target_vocab