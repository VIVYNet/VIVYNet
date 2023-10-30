# Fairseq Imports
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
from fairseq import utils

# Torch Imports
import torch
from torch.utils.data import Dataset

# Debug imports
from vivynet.utils.debug import Debug

# Miscellaneous Import
import numpy as np
import os
import math


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
        copy_tensor(
            v, res[i][size - len(v) :] if left_pad else res[i][: len(v)]
        )

    # Return the result
    return res


def midi_collate(samples, pad_idx, eos_idx):
    """Midi MultiHeadDataset Collater Function"""

    def merge(key, is_list=False):
        """Merge inner function"""

        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Append the collated tokens to the resulting list
                res.append(
                    collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                )

            # Return the result of the appending
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
        "ntokens": sum(s["source"].size(0) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor(
                [s["source"].size(0) for s in samples]
            ),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples),
    }


def t2m_collate(samples, src_vocab, tgt_vocab):
    """Text2Music PairDataset Collate Function"""

    def merge_midi(key, is_list=False):
        """Merge inner function"""

        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Append the collated tokens to the resulting list
                res.append(
                    collate_tokens(
                        [s[key][i] for s in samples],
                        tgt_vocab.pad(),
                        tgt_vocab.eos(),
                        left_pad=False,
                    )
                )

            # Return the result of the appending
            return res

        # If the given key is not a list, move here
        else:
            # Just return the collated tokens normally
            return collate_tokens(
                [s[key] for s in samples],
                tgt_vocab.pad(),
                tgt_vocab.eos(),
                left_pad=False,
            )

    def merge_text(key, is_list=False):
        # Check if the the provided key's value is a list datatype
        if is_list:
            # If so, append each iterated collated item to a resulting list
            res = []
            for i in range(len(samples[0][key])):
                # Append the collated tokens to the resulting list
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        src_vocab.pad(),
                        src_vocab.eos(),
                        left_pad=False,
                    )
                )

            # Return the result of the appending
            return res

        # If the given key is not a list, move here
        else:
            # Just return the collated tokens normally
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                src_vocab.pad(),
                src_vocab.eos(),
                left_pad=False,
            )

    # Return nothing if samples provided is nothing
    if len(samples) == 0:
        return {}

    # Merge the source midi tokens
    dec_in_tokens = merge_midi("dec_input")

    # Merge the source text tokens
    enc_input = merge_text("enc_input")

    # If the sample's target is empty, merge the target tokens
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge_midi("target", is_target_list)
    # If not, set the target equal to the source dataset
    else:
        target = dec_in_tokens

    # Return the resulting information
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(s["dec_input"].size(0) for s in samples),
        "net_input": {
            "enc_input": enc_input,
            "dec_in_tokens": dec_in_tokens,
            "dec_in_lengths": torch.LongTensor(
                [s["dec_input"].size(0) for s in samples]
            ),
        },
        "target": target,
        "ontokens": sum(s["on"] for s in samples),
    }


class TupleMultiHeadDataset(TokenBlockDataset):
    """Class Specification for Multiheaded Information"""

    def __init__(
        self,
        dataset,
        rand_chosen_cnt_l,
        sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        include_targets=False,
        document_sep_len=1,
        ratio=4 + 1,
        sample_overlap_rate=4,
        permutation_invariant=3,
        trk_idx=2,
        spec_tok_cnt=4,
        evt_vocab_size=425,
        trk_vocab_size=44,
        augmented=False,
    ):
        """Constructor for class"""

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

        # Set value of break_mode
        break_mode = break_mode if break_mode is not None else "complete_doc"
        assert break_mode == "complete_doc", break_mode

        # Transform and process sizes and other attributes
        sizes_cs = np.cumsum(sizes)
        piece_sep_ids = np.where(sizes == document_sep_len)[0].tolist()
        totpieces = len(piece_sep_ids)
        slice_indices = np.zeros((totpieces, 2), dtype=int)
        block_to_dataset_index = np.zeros((totpieces, 3), dtype=int)

        # Process sliced_indices and block_to_dataset_index arrays
        for i in range(len(piece_sep_ids)):
            s = piece_sep_ids[i - 1] if i > 0 else -1
            e = piece_sep_ids[i]
            slice_indices[i, :] = (
                sizes_cs[s] if s >= 0 else 0,
                sizes_cs[e - 1],
            )
            block_to_dataset_index[i, :] = (s + 1, 0, e - 1)

        # Apply data augmentation
        if augmented:
            # Apply data augmentation by creating multiple samples for each data
            # block
            sample_step = max(
                round(self.sample_len_max / sample_overlap_rate), 1
            )
            new_slice_indices = []
            new_block_to_dataset_index = []
            for line, line_piece in zip(slice_indices, block_to_dataset_index):
                l_piece_tot = line[1] - line[0]
                assert l_piece_tot % self.ratio == 0, (line[0], line[1])
                l_toks = l_piece_tot // self.ratio
                chosen_cnt = math.ceil(
                    (l_toks + np.random.randint(sample_step)) / sample_step
                )
                rand_chosen_cnt_l.append(chosen_cnt)
                # chosen_cnt = sum(
                #     1
                #     for _ in range(
                #         0 - np.random.randint(sample_step),
                #         l_toks,
                #         sample_step
                #     )
                # )
                new_slice_indices.append(np.stack([line] * chosen_cnt))
                new_block_to_dataset_index.append(
                    np.stack([line_piece] * chosen_cnt)
                )

            slice_indices = np.concatenate(new_slice_indices)
            block_to_dataset_index = np.concatenate(new_block_to_dataset_index)

        # # Transform the slices, sizes, and block information
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]
        self._sizes[:] = self.sample_len_max
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(
            block_to_dataset_index
        )

    def __getitem__(self, index):
        """Item Retrieval Method"""

        # Create index pointers
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[
            index
        ]
        assert start_offset == 0, (start_ds_idx, start_offset, end_ds_idx)

        # Create temporary variables
        buffer = []
        cur_len = 0

        st = start_ds_idx

        # Process information
        for idx in range(st, end_ds_idx + 1):
            tmp = self.dataset[idx].view(-1, self.ratio)
            if self.perm_inv % 2 == 1:
                all_cc_pos = (
                    torch.nonzero(tmp[..., 0] == self.cc_idx).view(-1).tolist()
                )
                all_cc_pos.append(tmp.size(0))
                to_swap = []
                for pos, nexp in zip(all_cc_pos[:-1], all_cc_pos[1:]):
                    to_swap.append(tmp[pos:nexp, ...])
                to_swap_idx = torch.randperm(len(to_swap))
                tmp = torch.cat(
                    [tmp[: all_cc_pos[0], ...]]
                    + [to_swap[x] for x in to_swap_idx]
                )
            mea = (idx - st + 1) * 3
            mea_num = torch.zeros((tmp.size(0), 1), dtype=int)
            mea_num[2:, 0] = mea
            mea_num[1][0] = mea - 1
            mea_num[0][0] = mea - 2
            buffer.append(torch.cat((tmp, mea_num), dim=1))
            cur_len += tmp.size(0)
            if cur_len >= self.sample_len_max:
                break

        # Create buffer and calculate it
        buffer = torch.cat(buffer)
        if cur_len < self.sample_len_max:
            buffer = torch.cat(
                [buffer, buffer.new([[self.eos] * (self.ratio + 1)])]
            )

        # Get item
        item = buffer[: self.sample_len_max, ...]
        if self.perm_inv > 0:
            perm = torch.cat(
                [
                    torch.arange(self.spec_tok_cnt),
                    torch.randperm(self.max_trk_cnt) + self.spec_tok_cnt,
                ]
            )
            item[..., self.trk_idx].apply_(lambda x: perm[x])

        assert self.include_targets

        # Process item
        source = torch.cat(
            [
                item.new([[self.eos] * (self.ratio - 1) + [0, 0]]),
                item[:-1, ...],
            ]
        )
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
        assert len(targets) == 1 and targets[0] == "future"

        # Set target data
        self.targets = targets

    def collater(self, samples):
        """Token collater method"""

        # Return the collated information of the given sample
        return midi_collate(samples, self.vocab.pad(), self.vocab.eos())

    def __getitem__(self, index):
        """Get item of an iterable based on its index"""

        # Make sure that the target data is not empty
        assert self.targets is not None

        # Get the source, target, and on of the passed in dataset
        source, target, on = self.dataset[index]

        # Generate the source and target information from the parsed info
        source, target = self._make_source_target(source, target, None)

        # Add the BOS token
        source, target = self._maybe_add_bos(source, target)

        # Return the processed information
        return {"id": index, "source": source, "target": target, "on": on}


class PairDataset(LanguagePairDataset):
    """Main dataset structure"""

    def __init__(
        self, src, src_sizes, src_dict, shuffle, tgt=None, tgt_sizes=None, tgt_dict=None
    ):
        """Text2Music Dataset classification"""

        # Super call
        super().__init__(src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict)

        # Variable definitions and initialization
        self.src = src
        self.src_dict = src_dict
        self.tgt = tgt
        self.tgt_dict = tgt_dict
        self.shuffle = shuffle

    def __getitem__(self, index):
        """Get item method"""

        # Extract information at given index
        enc_input = self.src[index]
        tgt_input = self.tgt[index]
        dec_input = tgt_input["source"]
        target = tgt_input["target"]
        on = tgt_input["on"]

        # Return the information
        return {
            "id": index,
            "enc_input": enc_input,
            "dec_input": dec_input,
            "target": target,
            "on": on,
        }

    def collater(self, samples):
        """Token collater method"""
        # Return the collated information of the given sample
        return t2m_collate(samples, self.src_dict, self.tgt_dict)


class TextDataset(Dataset):
    def __init__(self, source, l_chosen_cnt):
        super(TextDataset).__init__()

        # Initializing constructor
        self.source = source
        self.aug_src = []
        self.l_chosen_cnt = l_chosen_cnt
        self.sizes = []
        # Apply data augmentation for text dataset
        for seq, mul in zip(self.source, self.l_chosen_cnt):
            self.aug_src.extend([seq] * mul)
            self.sizes.extend([len(seq)] * mul)
        # # Convert final list to tensor
        # self.aug_src = torch.stack(self.aug_src)

    def __getitem__(self, index):
        return self.aug_src[index]

    def __len__(self):
        return len(self.aug_src)

    def sizes(self):
        return self.sizes


@register_task("text2music")
class VIVYData(LanguageModelingTask):
    """Dataset Class Specification"""

    debug = Debug("VIVYData", 7)

    @staticmethod
    def add_args(parser):
        """Argument parsing"""

        VIVYData.debug.ldf("<< START >>")

        # Get the data
        parser.add_argument("data", metavar="FILE", help="data")
        VIVYData.debug.ldf("data")

        # Shorten Method
        parser.add_argument("--shorten_method", type=str, metavar="N")
        VIVYData.debug.ldf("shorten_method")

        # Shorten Data Split List
        parser.add_argument("--shorten_data_split_list", type=str, metavar="N")
        VIVYData.debug.ldf("shorten_data_split_list")

        # Sample Break Mode
        parser.add_argument("--sample_break_mode", type=str, metavar="N")
        VIVYData.debug.ldf("sample_break_mode")

        # Ratio
        parser.add_argument("--ratio", type=int, metavar="N")
        VIVYData.debug.ldf("ratio")

        # Sample Overlap Rate
        parser.add_argument("--sample_overlap_rate", type=int, metavar="N")
        VIVYData.debug.ldf("sample_overlap_rate")
        VIVYData.debug.ldf("<< END >>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Task setup method"""

        VIVYData.debug.ldf("<< START >>")

        # Load dictionaries from the data
        src_vocab = Dictionary.load(
            os.path.join(args.data + "/features", "dict.txt")
        )
        VIVYData.debug.ldf("src_vocab")
        tgt_vocab = Dictionary.load(
            os.path.join(args.data + "/labels/bin", "dict.txt")
        )
        VIVYData.debug.ldf("tgt_vocab")
        print("| [input] dictionary: {} types".format(len(src_vocab)))
        print("| [label] dictionary: {} types".format(len(tgt_vocab)))

        # Return the instance of the training class
        VIVYData.debug.ldf("<< END >>")
        return VIVYData(args, tgt_vocab, src_vocab)

    def __init__(self, args, label_vocab, input_vocab):
        """Constructor for VIVYTrain class"""

        VIVYData.debug.ldf("<< START >>")

        # Set instance variables
        super().__init__(args, input_vocab, output_dictionary=label_vocab)
        # self.args = args
        self.src_vocab = input_vocab
        self.tgt_vocab = label_vocab
        VIVYData.debug.ldf("var dec")
        VIVYData.debug.ldf("<< END >>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split"""

        """
        TARGET DATA HANDLING
        """

        VIVYData.debug.ldf(f"<< START (split: {split}) >>")

        # Define variables
        augmented_midi = True

        # Split the paths to the data
        paths = utils.split_paths(self.args.data + "/labels/bin")
        assert len(paths) > 0
        VIVYData.debug.ldf("TGT - paths")

        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        VIVYData.debug.ldf("TGT - path split")

        # Read and get the information from the .bin and .idx files
        tgt_datasets = data_utils.load_indexed_dataset(
            split_path, self.tgt_vocab, self.args.dataset_impl, combine=combine
        )
        VIVYData.debug.ldf("TGT - tgt_datasets")

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
        VIVYData.debug.ldf("TGT - maybe_shorten_dataset")

        rand_chosen_cnt_l = []
        tgt_datasets = TupleMultiHeadDataset(
            tgt_datasets,
            rand_chosen_cnt_l,
            tgt_datasets.sizes,
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
            augmented=augmented_midi,
        )
        VIVYData.debug.ldf("TGT - TupleMultiHeadDataset Init")

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        VIVYData.debug.ldf("TGT - Add EOS for other targets")



        final_target = MultiheadDataset(
            dataset=tgt_datasets,
            sizes=tgt_datasets.sizes,
            src_vocab=self.tgt_vocab,
            tgt_vocab=self.tgt_vocab,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=False,
            targets=self.targets,
            add_bos_token=False,  # Note: it should be from args,
        )

        VIVYData.debug.ldf("TGT - MultiheadDataset Init")
        VIVYData.debug.ldf(
            f"TGT - *FINALIZED* (size: {len(final_target.sizes)}) - {split}"
        )

        """
        SOURCE DATA HANDLING
        """

        # Split the paths to the data
        paths = utils.split_paths(self.args.data + "/features")
        assert len(paths) > 0
        VIVYData.debug.ldf("SRC - paths")

        # Get the path splits
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        VIVYData.debug.ldf("SRC - path split")

        # Create dataset instance
        src_dataset = data_utils.load_indexed_dataset(
            split_path, self.src_vocab, self.args.dataset_impl, combine=combine
        )
        VIVYData.debug.ldf(f"SRC - *LOADED* (size: {len(src_dataset.sizes)})")

        src_dataset = (
            TextDataset(src_dataset, rand_chosen_cnt_l)
            if augmented_midi
            else src_dataset
        )

        # print(src_dataset.sizes)
        # print(final_target.sizes)
        # input()

        VIVYData.debug.ldf(f"SRC - *TEXT AUGMENTED* (size: {len(src_dataset)})")
        """
        DATASET COMPILATION
        """

        # Data shortening for debugging
        short_src = []
        short_src_vocab = []
        short_tgt = []
        short_tgt_vocab = []
        for i in range(10):
            short_src.append(src_dataset[i])
            short_src_vocab.append(src_dataset.sizes[i])
            short_tgt.append(final_target[i])
            short_tgt_vocab.append(final_target.sizes[i])
        VIVYData.debug.ldf("DEBUG - SHORTENING")

        # Data compilation
        self.datasets[split] = PairDataset(
            src=short_src or src_dataset,
            src_sizes=short_src_vocab or src_dataset.sizes,
            src_dict=self.src_vocab,
            tgt=short_tgt or final_target,
            tgt_sizes=short_tgt_vocab or final_target.sizes,
            tgt_dict=self.tgt_vocab,
            shuffle= True,
        )
        VIVYData.debug.ldf("COMPILATION")
        VIVYData.debug.ldf(f"<< END (split: {split}) >>")

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        VIVYData.debug.ldf("<< src_vocab >>")
        return self.src_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        VIVYData.debug.ldf("<< tgt_vocab >>")
        return self.tgt_vocab

    def _initialize_dataset(self, **kwargs):
        """Method to Initialize the Pair Dataset (Text, Midi)"""
        return PairDataset(**kwargs)
