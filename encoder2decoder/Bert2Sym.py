from fairseq.models import FairseqEncoder, FairseqEncoderModel, FairseqEncoderDecoderModel, register_model, register_model_architecture
from fairseq.tasks import LegacyFairseqTask, FairseqTask, register_task
from fairseq.data import Dictionary, data_utils, TokenBlockDataset, MonolingualDataset, LanguagePairDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq import utils, metrics
from dataclasses import dataclass, field
from transformers import BertTokenizer, BertModel, BertConfig
from encoder import BertEncoder
from decoder.symphony_net.src.fairseq.modified_linear_transformer import linear_transformer_multi
import logging, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

@register_criterion("multiple_loss")
class MultiplelossCriterion(CrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        losses = self.compute_loss(model, net_output, sample, reduce=reduce) # return a list
        assert not self.sentence_avg
        #TODO: adjust weight of evt losses and other losses by length (current strategy: simple average the losses)
        # weights = [sample["ntokens"]] + [sample["ontokens"]] * (len(losses) - 1)
        loss = torch.mean(torch.stack(losses))
        logging_output = {
            "loss": loss.data,
            "evt_loss": losses[0].data,
            "dur_loss": losses[1].data,
            "trk_loss": losses[2].data,
            "ins_loss": losses[3].data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample["ntokens"],
            "on_sample_size": sample["ntokens"],
        }
        return loss, sample["ntokens"], logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs_tuple = model.get_normalized_probs(net_output, log_probs=True)
        print("SAMPLE: ", sample)
        print("PROBS: ", lprobs_tuple, " + ",lprobs_tuple[0].size())
        print("PROBS_SIZE: ", len(lprobs_tuple))
        losses = []
        for idx, lprobs in enumerate(lprobs_tuple):
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output)[..., idx].view(-1)

            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
            losses.append(loss)
        return losses

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_evt = sum(log.get("evt_loss", 0) for log in logging_outputs)
        loss_dur = sum(log.get("dur_loss", 0) for log in logging_outputs)
        loss_trk = sum(log.get("trk_loss", 0) for log in logging_outputs)
        loss_ins = sum(log.get("ins_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        on_sample_size = sum(log.get("on_sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        # total_losses = 4
        # weighted_size = (sample_size + on_sample_size*(total_losses-1)) / total_losses
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "evt_loss", loss_evt / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dur_loss", loss_dur / on_sample_size / math.log(2), on_sample_size, round=3
        )
        metrics.log_scalar(
            "trk_loss", loss_trk / on_sample_size / math.log(2), on_sample_size, round=3
        )
        metrics.log_scalar(
            "ins_loss", loss_ins / on_sample_size / math.log(2), on_sample_size, round=3
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            metrics.log_derived(
                "evt_ppl", lambda meters: utils.get_perplexity(meters["evt_loss"].avg)
            )
            metrics.log_derived(
                "dur_ppl", lambda meters: utils.get_perplexity(meters["dur_loss"].avg)
            )
            metrics.log_derived(
                "trk_ppl", lambda meters: utils.get_perplexity(meters["trk_loss"].avg)
            )
            metrics.log_derived(
                "ins_ppl", lambda meters: utils.get_perplexity(meters["ins_loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
@dataclass
class SymphonyModelingConfig(FairseqDataclass):
    # Note: Add more configs for Encoder
    ratio: int = field(
        default=4, metadata={"help": "note/metadata attribute amount: default (evt, dur, trk, ins)"}
    )
    evt_voc_size: int = field(
        default=-1, metadata={"help": "event vocab size"}
    )
    dur_voc_size: int = field(
        default=-1, metadata={"help": "duration vocab size"}
    )
    trk_voc_size: int = field(
        default=-1, metadata={"help": "track vocab size"}
    )
    ins_voc_size: int = field(
        default=-1, metadata={"help": "instrument vocab size"}
    )
    max_rel_pos: int = field(
        default=-1, metadata={"help": "maximum relative position index, calculated by make_data.py"}
    )
    max_mea_pos: int = field(
        default=-1, metadata={"help": "maximum measure cnt within a sample, calculated by make_data.py"}
    )
    perm_inv: int = field(
        default=3, metadata={"help": "consider permutation invariance for music, 0: without PI, 1: data augmentation only, 2: positional encoding only, 3: all considered"}
    )
    sample_overlap_rate: int = field(
        default=4, metadata={"help": "sample overlap rate, default is 4 (stride 1024), also needed in make_data.py"}
    )

@register_model('Bert2SymphonyNet')
class Bert2SymphonyNet(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
    
    @staticmethod
    def add_args(parser):
        # WORK IN PROCESS
        # Note: change feature name for encoder and decoder
        parser.add_argument('--enc-fn-dim', type=int, metavar='N',
                            help='encoder feed forward layer dimension')
        parser.add_argument('--enc-dropout', type=float, metavar='D',
                            help='encoder dropout probability for all fully connected layers')
        parser.add_argument('--dec-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--dec-num-attention-heads', type=int, metavar='N',
                            help='decoder num attention heads')
        parser.add_argument('--dec-num-layers', type=int, metavar='N',
                            help='decoder num layers')
        parser.add_argument('--dec-dropout', type=float, metavar='D',
                            help='decoder dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')

    @classmethod
    def build_model(cls, args, task):
        # WORK IN PROCESS
        encoder = BertEncoder.BertEncoder(args)
        decoder = linear_transformer_multi.LinearTransformerMultiHeadDecoder(args, task)
        return Bert2SymphonyNet(encoder, decoder)
    
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

@register_model_architecture('Bert2SymphonyNet', 'Default')
def base_architecture(args):
    args.enc_dropout = getattr(args, 'enc_dropout', 0.5)
    args.enc_fn_dim = getattr(args, 'enc_fn_dim', 768)
    args.dec_embed_dim = getattr(args, "dec_embed_dim", 512)
    args.dec_num_attention_heads = getattr(args, "dec_num_attention_heads", 16)
    args.dec_num_layers = getattr(args, "dec_num_layers", 12)
    args.dec_dropout = getattr(args, "dec_dropout", 0.1)

@register_task('midi_generate')
class MidiGenerateTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        # WORK IN PROCESS: need more argument if needed
        parser.add_argument('data', metavar='FILE',
                            help = 'file prefix for data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # WORK IN PROCESS: need text and processed midi dataset
        input_dummy_name = 'dataset'
        target_dummy_name = 'dataset'

        input_vocab = Dictionary.load(os.path.join(args.data, input_dummy_name))
        target_vocab = Dictionary.load(os.path.join(args.data, target_dummy_name))
        return MidiGenerateTask(
            args= args,
            input_vocab = input_vocab,
            target_vocab = target_vocab
        )


    def __init__(self, args, input_vocab, target_vocab):
        super().__init__(args)
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
    
    def load_dataset(self, split, epoch, **kwargs):
        paths = utils.split_paths(self.args.data)
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)
        # WORK IN PROCESS:
        
        src_path = os.path.join(self.args.data, "{}-{}".format("src",split_path))
        tgt_path = os.path.join(self.args.data, "{}-{}".format("tgt",split_path))

        src = data_utils.load_indexed_dataset(
            src_path,
            self.input_vocab,      #Source Dictionary
            self.args.dataset_impl #Default dataset_impl is used: MMAP (binary files)
        )

        tgt = data_utils.load_indexed_dataset(
            tgt_path, 
            self.target_vocab,     #Target Dictionary
            self.args.dataset_impl #Default dataset_impl is used: MMAP (binary files)
        )
                
        # Process text-based dataset
        text_dataset = src
        # Process music dataset
        music_dataset = linear_transformer_multi.maybe_shorten_dataset(
            tgt,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed
        )

        music_dataset = linear_transformer_multi.TupleMultiHeadDataset(
            music_dataset,
            music_dataset.sizes,
            self.args.tokens_per_sample,
            pad = self.target_dictionary.pad(),
            eos = self.target_dictionary.eos(),
            break_mode = self.args.sample_break_mode,
            ratio=self.args.ratio + 1,
            sample_overlap_rate=self.args.sample_overlap_rate,
            permutation_invariant=self.args.perm_inv,
            #trk_idx=self.args.trk_idx,
            #spec_tok_cnt=self.args.spec_tok_cnt,
            evt_vocab_size=self.args.evt_voc_size,
            trk_vocab_size=self.args.trk_voc_size
        )
        # Pairing
        self.datasets[split] = LanguagePairDataset(
            src = text_dataset,
            src_sizes = text_dataset.sizes,
            src_dict = self.input_vocab,
            tgt = music_dataset,
            tgt_sizes = music_dataset.sizes,
            tgt_dict = self.target_vocab,
            shuffle = True,
            left_pad_source=True
        )

    @property
    def source_dictionary(self):
        return self.input_vocab
    
    @property
    def target_dictionary(self):
        return self.target_vocab