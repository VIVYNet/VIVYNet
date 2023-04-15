# Import
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.criterions import register_criterion
from fairseq.models import ( 
    FairseqEncoder, 
    BaseFairseqModel, 
    register_model,
    register_model_architecture
)
from fairseq.tasks import (
    FairseqTask, 
    register_task
)
from fairseq.data import (
    Dictionary, 
    LanguagePairDataset,
    TokenBlockDataset,
    data_utils, 
    shorten_dataset,
    MonolingualDataset,
    plasma_utils
)
from fairseq import utils

from transformers import BertForSequenceClassification

import torch.nn.functional as F
import torch

import numpy as np
import math
import os

#
#   MODEL SPECIFICATION
#

class BERT(FairseqEncoder):
    """BERT Model Declaration"""    
    
    def __init__(self, args, dictionary):
        """Constructor for BERT specifications"""
        
        # Super module call
        super().__init__(dictionary)
        
        # Instance variables
        self.device = torch.device("cuda")
        self.args = args
        
        # Initialize model
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased"
        )
        
        # Run model of CUDA
        self.model.cuda()
        
    def forward(self, src_token, src_length):
        """Forward function to specify forward propogation"""
        
        # Send data to device
        src_token = src_token.to(self.device).long()
        
        # Return logits from BERT
        output = self.model(src_token)
        
        # Return result
        return output
    
class SymphonyNet():
    """SymphonyNet Model Specification"""
    
    # Pass
    pass

#
#   FULL MODEL DEFINITION
#

@register_model('vivy')
class VIVYNet(BaseFairseqModel):
    """Encoder and Decoder Specification for Full Training"""
    
    @classmethod
    def build_model(cls, args, task):
        """Build model function"""
        
        # Create BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        
        # Create SymphonyNet model
        symphony_net = SymphonyNet()
        
        # Return 
        return VIVYNet()

    def __init__(self, encoder, decoder, input_vocab):
        """Constructor for the VIVYNet model"""
        
        # Retrieves attributes
        super(VIVYNet, self).__init__()
        
        # Create instance variables based on parameters given
        self.encoder = encoder
        self.decoder = decoder
        self.input_vocab = input_vocab
        
        # Put models into train mode
        self.encoder.train()
    
    def forward(self, src_tokens, src_lengths):
        """Forward propagation method"""
        
        # Clear previously caluclated gradients
        self.encoder.zero_grad()
        
        # Get loss and the logits from the model
        result = self.encoder(src_tokens, len(src_lengths))
        
        # Return the logits
        return result

@register_model_architecture('vivy', 'vivy_train')
def train(args):
    """Train function"""
    
    # Do nothing
    pass
 
#
#   DATASET SPECIFICATION
#

class TupleMultiHeadDataset(TokenBlockDataset):
    """Class Specification for Multiheaded Information"""
    
    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        ratio=4+1,
        sample_overlap_rate=4,
        permutation_invariant=3,
        trk_idx=2,
        spec_tok_cnt=4,
        evt_vocab_size=425,
        trk_vocab_size=44,
    ):
        """Constructor for class"""
        
        # Try to import modules from fairseq
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        
        # Raise errors if importingn fails
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )
        
        # Super call attributes and operations from parent class
        super(TokenBlockDataset, self).__init__()
        
        # Variable initialization
        self.dataset = dataset
        self.pad = pad
        self.eos = eos
        self.include_targets = include_targets
        self.ratio = ratio
        self.perm_inv = permutation_invariant
        self.sample_len_max = block_size
        self.trk_idx = trk_idx
        self.cc_idx = evt_vocab_size - 1
        self.spec_tok_cnt = spec_tok_cnt
        self.max_trk_cnt = trk_vocab_size - spec_tok_cnt
        assert len(dataset) == len(sizes)
        assert len(dataset) > 0
        
        # Turn sizes list into a numpy array datatype
        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        # Set valuie of break_mode
        break_mode = break_mode if break_mode is not None else "complete_doc"
        assert break_mode == 'complete_doc', break_mode
        
        # Transform and process sizes and other attributes
        sizes_cs = np.cumsum(sizes)
        piece_sep_ids = np.where(sizes == document_sep_len)[0].tolist()
        totpieces = len(piece_sep_ids)
        slice_indices = np.zeros((totpieces,2), dtype=int)
        block_to_dataset_index = np.zeros((totpieces,3), dtype=int)
        
        # Process slicde_indices and block_to_dataset_index arrays
        for i in range(len(piece_sep_ids)):
            s = piece_sep_ids[i-1] if i > 0 else -1
            e = piece_sep_ids[i]
            slice_indices[i, :] = (sizes_cs[s] if s >= 0 else 0, sizes_cs[e-1])
            block_to_dataset_index[i, :] = (s+1, 0, e-1)
        
        # Calculate the sample step
        sample_step = max(round(self.sample_len_max / sample_overlap_rate), 1) 
        
        # Variable declaration for slices and blocks
        new_slice_indices = []
        new_block_to_dataset_index = []
        
        # Add line information into slice and block indexes
        for line, line_piece in zip(slice_indices, block_to_dataset_index):
            l_piece_tot = line[1] - line[0]
            assert l_piece_tot % self.ratio == 0, (line[0], line[1])
            l_toks = l_piece_tot // self.ratio
            chosen_cnt = math.ceil((l_toks + np.random.randint(sample_step)) / sample_step)
            new_slice_indices.append(np.stack([line]*chosen_cnt))
            new_block_to_dataset_index.append(np.stack([line_piece]*chosen_cnt))

        # Concatentate new slice and block indexes together with their other counterparts
        slice_indices = np.concatenate(new_slice_indices)
        block_to_dataset_index = np.concatenate(new_block_to_dataset_index)
        
        # Transform the slices, sizes, and block information
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]
        self._sizes[:] = self.sample_len_max
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)
    
    def __getitem__(self, index):
        """Item Retrieval Method"""
        
        # Create index pointers
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        assert start_offset == 0, (start_ds_idx, start_offset, end_ds_idx)
        
        # Create a starting point randomly
        st = np.random.randint(start_ds_idx, end_ds_idx+1)
        
        # Create temporary variables
        buffer = []
        cur_len = 0
        
        # Process information
        for idx in range(st, end_ds_idx+1):
            tmp = self.dataset[idx].view(-1, self.ratio)
            if self.perm_inv % 2 == 1:
                all_cc_pos = torch.nonzero(tmp[..., 0] == self.cc_idx).view(-1).tolist()
                all_cc_pos.append(tmp.size(0))
                to_swap = []
                for pos, nexp in zip(all_cc_pos[:-1], all_cc_pos[1:]):
                    to_swap.append(tmp[pos:nexp, ...])
                to_swap_idx = torch.randperm(len(to_swap))
                tmp = torch.cat([tmp[:all_cc_pos[0], ...]] + [to_swap[x] for x in to_swap_idx])
            mea = (idx-st+1) * 3
            mea_num = torch.zeros((tmp.size(0),1), dtype=int)
            mea_num[2:, 0] = mea
            mea_num[1][0] = mea-1
            mea_num[0][0] = mea-2
            buffer.append(torch.cat((tmp, mea_num), dim=1))
            cur_len += tmp.size(0)
            if cur_len >= self.sample_len_max:
                break

        # Create buffer and calculate it
        buffer = torch.cat(buffer)
        if cur_len < self.sample_len_max:
            buffer = torch.cat([buffer, buffer.new([[self.eos]*(self.ratio+1)])])
        
        # Get item
        item = buffer[:self.sample_len_max, ...]
        if self.perm_inv > 0:
            perm = torch.cat([torch.arange(self.spec_tok_cnt), torch.randperm(self.max_trk_cnt) + self.spec_tok_cnt])
            item[..., self.trk_idx].apply_(lambda x: perm[x])
        assert self.include_targets
        
        # Process item
        source = torch.cat([item.new([[self.eos]*(self.ratio-1) + [0, 0]]), item[:-1, ...]])
        on = torch.sum(item[:, 1].ne(self.pad)).item()
        
        # Return item
        return source, item, on

# def collate_tokens(
#     values,
#     pad_idx,
#     eos_idx=None,
#     left_pad=False,
# ):
#     """Convert a list of 2d tensors into a padded 3d tensor."""
#     size = max(v.size(0) for v in values) # max batch size
 
#     res = values[0].new(len(values), size, values[0].size(-1)).fill_(pad_idx)

#     def copy_tensor(src, dst):
#         assert dst.numel() == src.numel()
#         dst.copy_(src)

#     for i, v in enumerate(values):
#         copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])

#     return res

# # pad = 1, eos = 2
# def collate(samples, pad_idx, eos_idx):
#     if len(samples) == 0:
#         return {}
#     # print('raw length', end = ' ')
#     # for s in samples:
#     #     print(len(s['source']), end = ' ')
#     # print()
#     def merge(key, is_list=False):
#         if is_list:
#             res = []
#             for i in range(len(samples[0][key])):
#                 res.append(
#                     collate_tokens(
#                         [s[key][i] for s in samples],
#                         pad_idx,
#                         eos_idx,
#                         left_pad=False,
#                     )
#                 )
#             return res
#         else:
#             return collate_tokens(
#                 [s[key] for s in samples],
#                 pad_idx,
#                 eos_idx,
#                 left_pad=False,
#             )

# class MultiheadDataset(MonolingualDataset):
#     def __init__(
#         self,
#         dataset,
#         sizes,
#         src_vocab,
#         tgt_vocab,
#         add_eos_for_other_targets,
#         shuffle,
#         targets=None,
#         add_bos_token=False,
#     ):
#         # print(len(sizes))
#         # print(type(dataset))
#         # print(len(dataset))
#         self.dataset = dataset
#         self.sizes = np.array(sizes)
#         self.vocab = src_vocab
#         self.tgt_vocab = tgt_vocab
#         self.add_eos_for_other_targets = add_eos_for_other_targets
#         self.shuffle = shuffle
#         self.add_bos_token = add_bos_token
#         assert not self.add_bos_token, "<bos> is occupied"

#         assert targets is None or all(
#             t in {"self", "future", "past"} for t in targets
#         ), "targets must be none or one of 'self', 'future', 'past'"
#         if targets is not None and len(targets) == 0:
#             targets = None
#         assert len(targets) == 1 and targets[0] == 'future'
#         self.targets = targets
#     def collater(self, samples):
#         return collate(samples, self.vocab.pad(), self.vocab.eos())

#     def __getitem__(self, index):
#         assert self.targets is not None
#         source, target, on = self.dataset[index]
#         source, target = self._make_source_target(
#                 source, target, None
#             )

#         source, target = self._maybe_add_bos(source, target)
#         return {"id": index, "source": source, "target": target, "on": on}

# @register_task("text2music")
# class VIVYData(LanguageModelingTask):
#     def load_dataset(self, split, epoch=1, combine=False, **kwargs):
#         """Load a given dataset split.

#         Args:
#             split (str): name of the split (e.g., train, valid, test)
#         """
#         paths = utils.split_paths(self.args.data)
#         assert len(paths) > 0

#         data_path = paths[(epoch - 1) % len(paths)]
#         split_path = os.path.join(data_path, split)

#         print(self.args)
#         input()
        
#         dataset = data_utils.load_indexed_dataset(
#             split_path, self.dictionary, self.args.dataset_impl, combine=combine
#         )
#         if dataset is None:
#             raise FileNotFoundError(
#                 "Dataset not found: {} ({})".format(split, split_path)
#             )
#         #print('load indexed dataset finished')
#         dataset = maybe_shorten_dataset(
#             dataset,
#             split,
#             self.args.shorten_data_split_list,
#             self.args.shorten_method,
#             self.args.tokens_per_sample,
#             self.args.seed,
#         )
#         #print('maybe_shorten_dataset finished')
#         dataset = TupleMultiHeadDataset(
#             dataset,
#             dataset.sizes,
#             self.args.tokens_per_sample,
#             pad=self.dictionary.pad(),
#             eos=self.dictionary.eos(),
#             break_mode=self.args.sample_break_mode,
#             include_targets=True,
#             ratio=self.args.ratio + 1,
#             sample_overlap_rate=self.args.sample_overlap_rate,
#             permutation_invariant=self.args.perm_inv,
#             #trk_idx=self.args.trk_idx,
#             #spec_tok_cnt=self.args.spec_tok_cnt,
#             evt_vocab_size=self.args.evt_voc_size,
#             trk_vocab_size=self.args.trk_voc_size,
#         )
#         #print('TupleMultiHeadDataset init finished')
#         add_eos_for_other_targets = (
#             self.args.sample_break_mode is not None
#             and self.args.sample_break_mode != "none"
#         )

#         self.datasets[split] = self._initialize_dataset(
#             dataset=dataset,
#             sizes=dataset.sizes,
#             src_vocab=self.dictionary,
#             tgt_vocab=self.output_dictionary,
#             add_eos_for_other_targets=add_eos_for_other_targets,
#             shuffle=True,
#             targets=self.targets,
#             add_bos_token=self.args.add_bos_token,
#         )
#         #print('_initialize_dataset finished')

#     # def _initialize_dataset(self, **kwargs):
#     #     return MultiheadDataset(**kwargs)
#     # def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
#     #     assert False, "inference not implemented"

@register_task('text2music')
class VIVYData(LanguageModelingTask):
    """Dataset Class Specification"""
    
    @staticmethod
    def add_args(parser):
        """Argument parsing"""
        
        # Get the data 
        parser.add_argument('data', metavar='FILE', help='data')
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Task setup method"""
        
        # Load dictionaries from the data
        src_vocab = Dictionary.load(os.path.join(args.data, 'dict.x.txt'))
        tgt_vocab = Dictionary.load(os.path.join(args.data, 'dict.y.txt'))
        print('| [input] dictionary: {} types'.format(len(src_vocab)))
        print('| [label] dictionary: {} types'.format(len(tgt_vocab)))
        
        # Return the instance of the training class
        return VIVYData(args, src_vocab, tgt_vocab)

    def __init__(self, args, input_vocab, label_vocab):
        """Constructor for VIVYTrain class"""
        
        # Set instance variables
        # super().__init__(args)
        self.src_vocab = input_vocab
        self.tgt_vocab = label_vocab
        
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        
        # # Get prefix
        # prefix = os.path.join(self.args.data, '{}.x-y'.format(split))        
        
        """
        SOURCE DATA HANDLING
        """
        
        # Prep source data and the length of the source data
        sources = []
        lengths = []
        
        # # Read source sentences file
        # with open(prefix + '.x', encoding='utf-8') as file:
            
        #     # Iterate through each line
        #     for line in file:
                
        #         # Strip the source sentence
        #         sentence = line.strip()
                
        #         # Tokenize the sentence, splitting on spaces
        #         tokens = self.input_vocab.encode_line(
        #             sentence, add_if_not_exist=False
        #         )
                
        #         # Append tokens to the sentences list
        #         # and its length to length list
        #         sources.append(tokens)
        #         lengths.append(len(tokens))

        """
        TARGET DATA HANDLING
        """
        
        # Split the paths to the data
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        
        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        
        # Create dataset instance
        tgt_dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        
        # If no dataset instance is created, raise an error
        if tgt_dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )
        
        # Shorten dataset if need be
        tgt_dataset = shorten_dataset.maybe_shorten_dataset(
            tgt_dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        
        # Create a tupled multihead dataset
        tgt_dataset = TupleMultiHeadDataset(
            tgt_dataset,
            tgt_dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            ratio=self.args.ratio + 1,
            sample_overlap_rate=self.args.sample_overlap_rate,
            permutation_invariant=self.args.perm_inv,
            evt_vocab_size=self.args.evt_voc_size,
            trk_vocab_size=self.args.trk_voc_size,
        )
        
        """
        DATASET COMPILATION
        """

        # Generate the dataset
        self.datasets[split] = LanguagePairDataset(
            src=sources,
            src_sizes=lengths,
            src_dict=self.src_vocab,
            tgt=tgt_dataset,
            tgt_sizes=tgt_dataset.sizes,
            tgt_dict=self.tgt_vocab
        )
        
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab

#
#   CRITERION SPECIFICATION
#

@register_criterion("nll_loss")
class ModelCriterion(CrossEntropyCriterion):
    
    #
    #   NOT NEEDED ANYMORE
    #
    
    def forward(self, model, sample, reduce=True):
        
        # Get output of the model
        net_output = model(**sample["net_input"])
        
        # Compute the losses of the output
        losses = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        # Aggregate losses
        loss = torch.mean(torch.stack(losses))
        
        # Create logging output
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample["ntokens"],
            "on_sample_size": sample["ntokens"],
        }
        
        # Return information
        return loss, sample["ntokens"], logging_output
    
    def compute_loss(self, model, net_output, sample, reduce=True):
        
        # Get normalized probability from the net_ouput
        lprobs_tuple = model.get_normalized_probs(net_output, log_probs=True)
        
        # Declare a list to store losess
        losses = []
        
        # Iterate through all normalized probability
        for idx, lprobs in enumerate(lprobs_tuple):
            
            # Change the probability dimension
            lprobs = lprobs.view(-1, lprobs.size(-1))
            
            # Get the target data
            target = model.get_targets(sample, net_output)[..., idx].view(-1)

            # Calculate loss
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            
            # Append the loss to the loss list
            losses.append(loss)
            
        # Return the list of losses
        return losses