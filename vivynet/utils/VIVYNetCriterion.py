# Fairseq Imports
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.criterions import register_criterion

# Torch Imports
import torch
import torch.nn.functional as F

# Debug Imports
from vivynet.utils.debug import Debug


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
