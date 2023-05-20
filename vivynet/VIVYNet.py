# Fairseq Imports
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.tasks import register_task
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data import (
    LanguagePairDataset,
    MonolingualDataset,
    TokenBlockDataset,
    Dictionary, 
    plasma_utils,
    data_utils, 
)
from fairseq.models import ( 
    BaseFairseqModel,
    FairseqDecoder, 
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model_architecture, 
    register_model,
)
from fairseq import utils

# HuggingFace Imports
from transformers import BertForSequenceClassification

# Torch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# FastTransformer Imports
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask, FullMask

# Miscellaneous Imports
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
    
class SymphonyNet(FairseqDecoder):
    """SymphonyNet Model Specification"""
    def __init__(self, args, task):
        #TODO: Add dictionary for encoder
        super().__init__(task.target_dictionary)
        #print(task.target_dictionary)
        # for i in range(len(task.target_dictionary)):
        #     print(i, task.target_dictionary[i])
        self.embed_dim = args.embed_dim
        self.wEvte = nn.Embedding(args.evt_voc_size, args.embed_dim)
        self.wTrke = nn.Embedding(args.trk_voc_size, args.embed_dim)
        self.wDure = nn.Embedding(args.dur_voc_size, args.embed_dim)
        self.max_pos = args.tokens_per_sample

        self.perm_inv = args.perm_inv
        if self.perm_inv > 1:
            self.wRpe = nn.Embedding(args.max_rel_pos+1, args.embed_dim) 
            self.wMpe = nn.Embedding(args.max_mea_pos+1, args.embed_dim)
        else:
            self.wpe = nn.Embedding(self.max_pos+1, args.embed_dim) # max_pos_len = 4096
        self.drop = nn.Dropout(args.dropout)
        self.ln_f = nn.LayerNorm(args.embed_dim, eps=1e-6)
        
        self.decoder_model = TransformerDecoderBuilder.from_kwargs(
                n_layers = args.num_layers,
                n_heads=args.num_attention_heads,
                query_dimensions=args.embed_dim // args.num_attention_heads,
                value_dimensions=args.embed_dim // args.num_attention_heads,
                feed_forward_dimensions=4 * args.embed_dim,
                activation='gelu',
                #final_normalization=True,
                dropout=args.dropout,
                self_attention_type="causal-linear", 
                cross_attention_type="full", # Fully masked so that each domain can be merged
            ).get()

        self.attn_mask = TriangularCausalMask(self.max_pos)
        self.proj_evt = nn.Linear(args.embed_dim, args.evt_voc_size, bias=False)
        self.proj_dur = nn.Linear(args.embed_dim, args.dur_voc_size, bias=False)
        self.proj_trk = nn.Linear(args.embed_dim, args.trk_voc_size, bias=False)
        self.proj_ins = nn.Linear(args.embed_dim, args.ins_voc_size, bias=False)

        self.apply(self._init_weights)
        # set zero embedding for padding symbol
        #TODO: check will the pad id be trained? (as TZ RZ YZ)
        self.pad_idx = task.target_dictionary.pad()
        self.wEvte.weight.data[self.pad_idx].zero_()
        self.wDure.weight.data[self.pad_idx].zero_()
        self.wTrke.weight.data[self.pad_idx].zero_()
        if self.perm_inv > 1:
            self.wRpe.weight.data[0].zero_()
            self.wMpe.weight.data[0].zero_()
        else:
            self.wpe.weight.data[0].zero_()
            
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.embed_dim ** -0.5)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        encoder_out,
        x,
        src_lengths = None,
        encoder_out_lengths = None,
    ):
        features = self.extract_features(
            x = x, 
            encoder_out = encoder_out,
            src_lengths = src_lengths,
            encoder_out_lengths = encoder_out_lengths
            )
        
        evt_logits = self.proj_evt(features)
        dur_logits = self.proj_dur(features)
        trk_logits = self.proj_trk(features)
        ins_logits = self.proj_ins(features)

        return (evt_logits, dur_logits, trk_logits, ins_logits)

    # TODO: Understand how SymphonyNet masks work, including LengthMask and TriangularMask
    # TODO: Understand Permutiation Imvariant in code
    def extract_features(
        self,
        x,
        encoder_out = None,
        src_lengths = None,
        encoder_out_lengths = None
    ):
        bsz, seq_len, ratio = x.size()
        enc_bsz, enc_len = encoder_out.size()

        evt_emb = self.wEvte(x[..., 0])

        # if not mapping to pad, padding idx will only occer at last
        evton_mask = x[..., 1].ne(self.pad_idx).float()[..., None].to(x.device) 
        tmp = self.wDure(x[..., 1])
        dur_emb = tmp * evton_mask
        # assert ((tmp==dur_emb).all())
        tmp = self.wTrke(x[..., 2])
        trk_emb = tmp * evton_mask
        # assert ((tmp==trk_emb).all())

        # Note: Calc LengthMask for src_lengths
        pad_mask = x[..., 0].ne(self.pad_idx).long().to(x.device)
        if src_lengths is not None:
            len_mask = LengthMask(
                src_lengths, 
                max_len=seq_len, 
                device=x.device
                )
        else:
            len_mask = LengthMask(
                torch.sum(pad_mask, axis=1), 
                max_len=seq_len, 
                device= x.device)
        
        # Note: Calc LengthMask for endoer_out_lengths
        if encoder_out_lengths is not None:
            enc_len_mask = LengthMask(
                encoder_out_lengths, 
                max_len = enc_len,
                device= encoder_out.device)
        else:
            # WIP: Calc LengthMask when enc_out_len is none
            enc_pad_mask = x[1].ne(self.enc_pad_idx).long().to(x.device)
            enc_len_mask = LengthMask(
                torch.sum(enc_pad_mask, axis=1),
                max_len=enc_len,
                device= encoder_out.device)
            
        
        # WIP: Implement FullMask for Cross Attention layer
        full_mask = FullMask(
            N = seq_len,
            M = enc_len,
            device = x.device
        )
        
        # Note: Perform Permutation Invariant
        if self.perm_inv > 1:
            rel_pos = pad_mask * x[..., 4]
            rel_pos_mask = rel_pos.ne(0).float()[..., None].to(x.device) # ignore bom, chord, eos

            measure_ids = pad_mask * x[..., 5]
            mea_mask = measure_ids.ne(0).float()[..., None].to(x.device) # ignore eos
            
            pos_emb = rel_pos_mask * self.wRpe(rel_pos) + mea_mask * self.wMpe(measure_ids)

        else:
            # set position ids to exclude padding symbols
            position_ids = pad_mask * (
                torch.arange(1, 1 + seq_len)
                .to(x.device)
                .repeat(bsz, 1)
            )
            pos_emb = self.wpe(position_ids)
        
        x = self.drop(evt_emb+dur_emb+trk_emb+pos_emb)

        doutputs = self.decoder_model(
            x = x,
            memory = encoder_out,
            x_mask = self.attn_mask,
            x_length_mask = len_mask,
            memory_mask = full_mask, #WIP
            memory_length_mask = enc_len_mask #WIP
        )
        # print("Output: ",outputs)
        doutputs = self.ln_f(doutputs)
        
        return doutputs
    
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if log_probs:
            return tuple(utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in net_output)
        else:
            return tuple(utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in net_output)

    def max_positions(self):
        return None

#
#   FULL MODEL DEFINITION
#

@register_model('vivy')
class VIVYNet(FairseqEncoderDecoderModel):
    """Encoder and Decoder Specification for Full Training"""
    
    @staticmethod
    def add_args(parser):
        """Argument Definition class"""
        # Shorten Method
        parser.add_argument('--shorten_method', type=str, metavar='N')
        
        # Shorten Data Split List
        parser.add_argument('--shorten_data_split_list', type=str, metavar='N')
        
        # Token Per Sample
        parser.add_argument('--tokens_per_sample', type=int, metavar='N')
        
        # Sample Break Mode
        parser.add_argument('--sample_break_mode', type=str, metavar='N')
        
        # Ratio
        parser.add_argument('--ratio', type=int, metavar='N')
        
        # Sample Overlap Rate
        parser.add_argument('--sample_overlap_rate', type=int, metavar='N')
        
        # Permutation invariance
        parser.add_argument('--perm_inv', type=int, metavar='N')
        
        # Event Token Size
        parser.add_argument('--evt_voc_size', type=int, metavar='N')
        
        # Track Token Size
        parser.add_argument('--trk_voc_size', type=int, metavar='N')
    
    @classmethod
    def build_model(cls, args, task):
        """Build model function"""
        
        # Create BERT model
        bert = BERT(args=args, dictionary=task.source_dictionary)
        
        # Create SymphonyNet model
        symphony_net = SymphonyNet(args=args, task= task)
        
        # Return 
        return VIVYNet(bert, symphony_net)

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
#   DATASET SPECIFICATIONS
#

def copy_tensor(src, dst):
    """Tensor Copying Function"""
    
    # Check if the source and target tensors are equal in length
    assert dst.numel() == src.numel()
    
    # Copy the target tokens to the source information
    dst.copy_(src)

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
):
    """2D to 3D Tensor Function"""
    # Max batch size
    size = max(v.size(0) for v in values)
    
    # Generate the resulting values from the merge
    res = values[0].new(len(values), size, values[0].size(-1)).fill_(pad_idx)
        
    # Iterate through the provided values for collation and copy the 
    # tensor values to the resulting list 
    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])

    # Return the result 
    return res

def collate(samples, pad_idx, eos_idx):
    """Collater Function"""
    
    def merge(key, is_list=False):
        """Merge inner function"""
        
        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Apped the collated tokens to the resulting list
                res.append(
                    collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )
            
            # Retun the result of the appending 
            return res
        
        # If the given key is not a list, move here 
        else:
            # Just return the collated tokens normally
            return collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )
            
    # Return nothing if samples provided is nothing
    if len(samples) == 0:
        return {}
    
    # Merge the source tokens
    src_tokens = merge("source")
    
    # If the sample's target is empty, merge the target tokens
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    # If not, set the target equal to the source dataset 
    else:
        target = src_tokens

    # Return the resulting information
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(s["source"].size(0)  for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].size(0) for s in samples]),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples)
    }

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
        
        # Create temporary variables
        buffer = []
        cur_len = 0
        
        # Process information
        for idx in range(0, end_ds_idx+1):
            tmp = self.dataset[idx].view(-1, self.ratio)
            if self.perm_inv % 2 == 1:
                all_cc_pos = torch.nonzero(tmp[..., 0] == self.cc_idx).view(-1).tolist()
                all_cc_pos.append(tmp.size(0))
                to_swap = []
                for pos, nexp in zip(all_cc_pos[:-1], all_cc_pos[1:]):
                    to_swap.append(tmp[pos:nexp, ...])
                to_swap_idx = torch.randperm(len(to_swap))
                tmp = torch.cat([tmp[:all_cc_pos[0], ...]] + [to_swap[x] for x in to_swap_idx])
            mea = (idx-1) * 3
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

class MultiheadDataset(MonolingualDataset):
    """Final Preprocessing of the Multiheaded Datapoints"""
    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab,
        add_eos_for_other_targets,
        shuffle,
        targets=None,
        add_bos_token=False,
    ):
        """Contstructor for the class"""
        
        # Variable declaration and initialization
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        
        # Check if the a token in the given dataset 
        # is taken the intended <bos> token
        assert not self.add_bos_token, "<bos> is occupied"

        # Format the target data into correct format where
        # its geared for future format
        assert targets is None or all(
            t in {"self", "future", "past"} for t in targets
        ), "targets must be none or one of 'self', 'future', 'past'"
        if targets is not None and len(targets) == 0:
            targets = None
        assert len(targets) == 1 and targets[0] == 'future'
        
        # Set target data
        self.targets = targets
        
    def collater(self, samples):
        """Token collater method"""
        
        # Return the collated information of the given sample
        return collate(samples, self.vocab.pad(), self.vocab.eos())
        
    def __getitem__(self, index):
        """Get item of an iterable based on its index"""
        
        # Make sure that the target data is not empty
        assert self.targets is not None
        
        # Get the source, target, and on of the passed in dataset
        source, target, on = self.dataset[index]
        
        # Generate the source and target information from the parsed info
        source, target = self._make_source_target(
            source, target, None
        )

        # Add the BOS token 
        source, target = self._maybe_add_bos(source, target)
        
        # Return the processed information
        return {"id": index, "source": source, "target": target, "on": on}

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
        src_vocab = Dictionary.load(os.path.join(args.data + "/features", 'dict.txt'))
        tgt_vocab = Dictionary.load(os.path.join(args.data + "/labels/bin", 'dict.txt'))
        print('| [input] dictionary: {} types'.format(len(src_vocab)))
        print('| [label] dictionary: {} types'.format(len(tgt_vocab)))
        
        # Return the instance of the training class
        return VIVYData(args, tgt_vocab, src_vocab)

    def __init__(self, args, label_vocab, input_vocab):
        """Constructor for VIVYTrain class"""
        
        # Set instance variables
        # super().__init__(args)
        self.args = args
        self.src_vocab = input_vocab
        self.tgt_vocab = label_vocab
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        """
        TARGET DATA HANDLING
        """
        
        # Split the paths to the data
        paths = utils.split_paths(self.args.data  + "/labels/bin")
        assert len(paths) > 0
        
        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        
        # Read and get the information from the .bin and .idx files
        tgt_datasets = data_utils.load_indexed_dataset(
            split_path, self.tgt_vocab, self.args.dataset_impl, combine=combine
        )
        
        # If no dataset instance is created, raise an error
        if tgt_datasets is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # Shorten dataset if need be
        tgt_datasets = maybe_shorten_dataset(
            tgt_datasets,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        
        #
        # Split the combined measures into their corresponding sentences
        #
        
        # Set arrays for splitting
        temp_arr = []
        temp_sizes_arr = []
        tgt_sentences = []
        tgt_sentence_sizes = []
        
        # Iterate through the parsed data and make the splits
        for idx, item in enumerate(tgt_datasets):
            # Save the parsed information into the temporary arrays
            temp_arr.append(item)
            temp_sizes_arr.append(tgt_datasets.sizes[idx])
            
            # Check if the iterated item is an EOS measure
            if item.tolist() == [2]:
                # If so, append the temporary information into the resulting arrays
                tgt_sentences.append(temp_arr)
                tgt_sentence_sizes.append(temp_sizes_arr)
                
                # Reset temporary arrays
                temp_arr = []
                temp_sizes_arr = []
                
                # Continue
                continue
        
        #
        # Generate The Target Tokens for the Target Section of the Data
        #
        
        # Specification for EOS 
        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        
        # Create a list to store the tupled result
        tgt_tupled_sentences = []

        # Iterate through the spliced sentences and get the token
        # representation instances from each iterations
        for idx, item in enumerate(tgt_sentences):
            # Generate the TupleMultiHeadDataset of the dataset
            tmhd = TupleMultiHeadDataset(
                item,
                tgt_sentence_sizes[idx],
                self.args.tokens_per_sample,
                pad=self.tgt_vocab.pad(),
                eos=self.tgt_vocab.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,
                ratio=self.args.ratio + 1,
                sample_overlap_rate=self.args.sample_overlap_rate,
                permutation_invariant=self.args.perm_inv,
                evt_vocab_size=self.args.evt_voc_size,
                trk_vocab_size=self.args.trk_voc_size,
            )
            
            # Generate a MultiheadDataset
            mhd = self._initialize_dataset(
                dataset=tmhd,
                sizes=tmhd.sizes,
                src_vocab=self.src_vocab,
                tgt_vocab=self.tgt_vocab,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=True,
                targets=["future"],
                add_bos_token=False,
            )
            
            # Append the instance to tgt_tupled_sentences
            tgt_tupled_sentences.append(mhd[0]["target"])
        
        """
        SOURCE DATA HANDLING
        """
        
        # Split the paths to the data
        paths = utils.split_paths(self.args.data  + "/features")
        assert len(paths) > 0
        
        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        
        # Create dataset instance
        src_dataset = data_utils.load_indexed_dataset(
            split_path, self.src_vocab, self.args.dataset_impl, combine=combine
        )       
        
        """
        DATASET COMPILATION
        """
        
        # Generate the dataset
        self.dataset = LanguagePairDataset(
            src=src_dataset,    
            src_sizes=src_dataset.sizes,
            src_dict=self.src_vocab,
            tgt=tgt_tupled_sentences,
            tgt_sizes=[len(i) for i in tgt_tupled_sentences],
            tgt_dict=self.tgt_vocab
        )        
        
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_vocab
    
    def _initialize_dataset(self, **kwargs):
        """Method to Initialize the Target Data"""
        return MultiheadDataset(**kwargs)

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