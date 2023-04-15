# Import
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model, register_model_architecture
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import Dictionary, LanguagePairDataset
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import torch.nn
import torch
import os

class BERT(FairseqEncoder):
    
    def __init__(self, args, dictionary):
        
        # Super module call
        super().__init__(dictionary)
        
        # Instance variables
        self.device = torch.device("cuda")
        self.args = args
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Softmax
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
        # Run model on CUDA
        self.model.cuda()
        
    def forward(self, src_token, src_length):
        
        # Send data to device
        input_ids = src_token.to(self.device).long()
        
        # Return logits from BERT
        output = self.model(input_ids)
        logits = output["logits"]
        
        # Get the softmax
        sm = self.softmax(logits)
        sm = sm.detach().cpu().clone().numpy()
        
        # Get the predictions and change its dimensions
        pred = np.argmax(sm, axis=1)
        length = len(pred)
        pred = torch.Tensor(pred).cuda()
        pred.requires_grad_(True)
        pred = pred.view(length, 1)
        
        # Return result
        return {"encoder_out": pred}

@register_model('bert')
class BERTValidityClassifier(FairseqEncoderModel):
    
    @classmethod
    def build_model(cls, args, task):
        
        # Initiate BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        
        # Return the wrapped version of the module
        return BERTValidityClassifier(
            model=bert, 
            input_vocab=task.source_dictionary,
        )

    def __init__(self, model, input_vocab):
        
        # Retrieves attributes
        super(BERTValidityClassifier, self).__init__(model)
        
        # Save instance variables
        self.model = model
        self.input_vocab = input_vocab
        
        # Put model into train mode
        self.model.train()
        
    def forward(self, src_tokens, src_lengths):
        
        # Clear previously caluclated gradients
        self.model.zero_grad()
        
        # Get loss and the logits from the model
        logits = self.model(src_tokens, len(src_lengths))
        
        # Return the logits
        return logits
    
@register_model_architecture('bert', 'bert_train')
def train(args):
    pass

@register_task('bert_cola_train')
class BertCoLATrain(FairseqTask):
    
    @staticmethod
    def add_args(parser):
        
        # Get the data 
        parser.add_argument('data', metavar='FILE', help='data')
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Load dictionaries from the data
        input_vocab = Dictionary.load(os.path.join(args.data, 'dict.feat.txt'))
        label_vocab = Dictionary.load(os.path.join(args.data, 'dict.labl.txt'))
        print('| [input] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(label_vocab)))
        
        return BertCoLATrain(args, input_vocab, label_vocab)
    
    def __init__(self, args, input_vocab, label_vocab):
        
        # Set instance variables
        super().__init__(args)
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab
    
    def load_dataset(self, split, **kwargs):
        
        # Get prefix
        prefix = os.path.join(self.args.data, '{}.feat-labl'.format(split))
        
        # Prep source data and the length of the source data
        sources = []
        lengths = []
        
        # Read source sentences file
        with open(prefix + '.feat', encoding='utf-8') as file:
            
            # Iterate through each line
            for line in file:
                
                # Strip the source sentence
                sentence = line.strip()
                
                # Tokenize the sentence, splitting on spaces
                tokens = self.input_vocab.encode_line(
                    sentence, add_if_not_exist=False
                )
                
                # Append tokens to the sentences list
                # and its length to length list
                sources.append(tokens)
                lengths.append(len(tokens))
                
        # Prep label list
        labels = []
        
        # Read label file
        with open(prefix + '.labl', encoding='utf-8') as file:
            
            # Iterate through each line
            for line in file:
                
                # Strip the lable
                label = line.strip()
                
                # Process label
                process = torch.LongTensor([int(label)])
                
                # Add to labels
                labels.append(process)
        
        # Check the alignment of the data
        assert len(sources) == len(labels)
        print('| {} {} {} examples'.format(self.args.data, split, len(sources)))
        
        # We reuse LanguagePairDataset since classification can be modeled as a
        # sequence-to-sequence task where the target sequence has length 1.
        self.datasets[split] = LanguagePairDataset(
            src=sources,
            src_sizes=lengths,
            src_dict=self.input_vocab,
            tgt=labels,
            tgt_sizes=[1] * len(labels),  # targets have length 1
            tgt_dict=self.label_vocab,
            left_pad_source=False,
            input_feeding=False,
        )
    
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab