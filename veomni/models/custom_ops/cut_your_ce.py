from veomni.models.custom_ops.cut_cross_entropy import linear_cross_entropy, LinearCrossEntropy, VocabParallelOptions
import torch
from typing import Optional
# embeddings = model.compute_embedding(inputs)
# classifier = model.get_classifier_weights()

# # This is the same as manual_shift_loss above
# auto_shift_loss = linear_cross_entropy(embeddings, classifier, labels, shift=1)


class CutYourCE(torch.nn.Module):
    def __init__(
        self,
        is_shifted: bool = True,
        tensor_parallel_size: int = 1,
        tensor_group: Optional[torch.distributed.ProcessGroup] = None,
        reduction: str = "sum",
        ignore_index: int = -100,
        softcap: Optional[float] = None,
        implementation_type: str = "cce_kahan", # I prefer use cc_kahan -> cce_kahan_full_c -> cce_kahan_full_c_full_e
        # this order is recomendation for memory first and precision last: i.e cc_kahan is the fastest and cce_kahan_full_c_full_e is the most precise
    ):
        """
        Uses Kahan summation (or fp32) to improve numerical precision. 
            This comes at the cost of more memory usage (albeit only a temporary buffer in the backward pass).
            This is useful for long sequence lengths or if the model is particularly sensitive to numerical imprecision.
        Args:
            is_shifted (bool): Whether to use shifted logits.
            tensor_parallel_size (int): The size of the tensor parallel group.
            tensor_group (Optional[torch.distributed.ProcessGroup]): The process group for the tensor parallel group.
        Todo: add tensor parallel support
        """
        super().__init__()
        self.is_shifted = is_shifted
        self.ce_fn = LinearCrossEntropy(
            ignore_index=ignore_index,
            softcap=softcap,
            impl=implementation_type,
            reduction=reduction,
            shift=1 if not self.is_shifted else 0
        )
    
    def forward(self, classifier_weights, hidden_states,  labels, vocab_size=None, group=None):
        # Modified order to match the order of the LigerFusedLinearCrossEntropyLoss
        # Note that: we didn't support bias weights here
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states.flatten(0, -2) # [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, hidden_dim]
            labels = labels.flatten()
        if vocab_size is not None and group is not None:
            vp_opts = VocabParallelOptions.from_vocab(vocab_size, group=group)
            return self.ce_fn(
                hidden_states,
                classifier_weights,
                labels,
                vocab_parallel_options=vp_opts
            )
        return self.ce_fn(
            hidden_states,
            classifier_weights,
            labels,
        )