# Fairseq Imports
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion
from fairseq import utils, metrics

# Torch Imports
import torch
import torch.nn.functional as F

# Debug Imports
from vivynet.utils.debug import Debug

# Miscellaneous Imports
import wandb
import math


@register_criterion("nll_loss")
class ModelCriterion(CrossEntropyCriterion):
    """Model criterion class"""

    debug = Debug("ModelCriterion", 5)

    def forward(self, model, sample, reduce=True):
        """Forward function for the criterion"""

        ModelCriterion.debug.ldf("<< START >>")

        # Get output of the model
        net_output = model(
            sample["net_input"]["enc_input"],
            sample["net_input"]["dec_in_tokens"],
        )
        ModelCriterion.debug.ldf("VIVYNet Output")

        # Compute the losses of the output
        losses = self.compute_loss(model, net_output, sample, reduce=reduce)
        ModelCriterion.debug.ldf("Process Losses")

        # Aggregate losses
        loss = torch.mean(torch.stack(losses))
        ModelCriterion.debug.ldf("Aggregate Losses")

        # Create logging output
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
        ModelCriterion.debug.ldf("Generate Logging")

        # # Log with WandB
        # wandb.log(logging_output)
        # ModelCriterion.debug.ldf("WANDB Logged")

        # Return information
        ModelCriterion.debug.ldf("<< END >>")
        return loss, sample["ntokens"], logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """Loss computation"""

        ModelCriterion.debug.ldf("<< START >>")

        # Get normalized probability from the net_output
        lprobs_tuple = model.get_normalized_probs(net_output, log_probs=True)
        losses = []
        ModelCriterion.debug.ldf("Normalized Probability")

        # Iterate through all normalized probability
        for idx, lprobs in enumerate(lprobs_tuple):
            # Change the probability dimension
            lprobs = lprobs.view(-1, lprobs.size(-1))
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
        ModelCriterion.debug.ldf("Losses Calculations")

        # Return the list of losses
        ModelCriterion.debug.ldf("<< END >>")
        return losses

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        # Preprocess logged metrics
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_evt = sum(log.get("evt_loss", 0) for log in logging_outputs)
        loss_dur = sum(log.get("dur_loss", 0) for log in logging_outputs)
        loss_trk = sum(log.get("trk_loss", 0) for log in logging_outputs)
        loss_ins = sum(log.get("ins_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        on_sample_size = sum(
            log.get("on_sample_size", 0) for log in logging_outputs
        )

        # we divide by log(2) to convert the loss from base e to base 2
        # total_losses = 4
        # weighted_size =
        #   (sample_size + on_sample_size*(total_losses-1)) / total_losses

        # Track metrics
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "evt_loss",
            loss_evt / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_scalar(
            "dur_loss",
            loss_dur / on_sample_size / math.log(2),
            on_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "trk_loss",
            loss_trk / on_sample_size / math.log(2),
            on_sample_size,
            round=3,
        )
        metrics.log_scalar(
            "ins_loss",
            loss_ins / on_sample_size / math.log(2),
            on_sample_size,
            round=3,
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl",
                lambda meters: utils.get_perplexity(meters["nll_loss"].avg),
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            metrics.log_derived(
                "evt_ppl",
                lambda meters: utils.get_perplexity(meters["evt_loss"].avg),
            )
            metrics.log_derived(
                "dur_ppl",
                lambda meters: utils.get_perplexity(meters["dur_loss"].avg),
            )
            metrics.log_derived(
                "trk_ppl",
                lambda meters: utils.get_perplexity(meters["trk_loss"].avg),
            )
            metrics.log_derived(
                "ins_ppl",
                lambda meters: utils.get_perplexity(meters["ins_loss"].avg),
            )

        # Track metrics with WANDB
        out = metrics.get_active_aggregators()[0]
        track = {}
        for k in out:
            iter = out[k]
            if hasattr(iter, "avg"):
                track[k] = out[k].avg
            else:
                track[k] = out[k].fn(out)
        wandb.log(track)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
