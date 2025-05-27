import torch
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from veomni.models.transformers.tensor_parallelism.indentity import IndentityParallel
from veomni.models.transformers.tensor_parallelism.prepare_input import PrepareModuleInputTuple


def apply_tensor_parallel_plan(model: torch.nn.Module, tp_mesh):
    # Note that the inputs to the model are already sharded 
    # For each shard process, we need apply the same inputs to the model 

    basic_shard_modules = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(), # input is replicated across shards
            output_layouts=Shard(1), # output is sharded split across shards
        ),
        "model.norm": SequenceParallel(), 
        "model.lm_head": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ) 
    }
    # Attention modules: follow the same pattern of megatron-lm
    model = parallelize_module(
        model,
        tp_mesh,
        basic_shard_modules
    )

    parallelize_module(
        model.model,
        tp_mesh,
        {
            "rotary_emb": PrepareModuleOutput(
                output_layouts=(Replicate(), Replicate()), # output is replicated across shards due to the attention make in the all sequence
                desired_output_layouts=(Replicate(), Replicate()),
            )
        }
    )
    # return model
    num_layers = len(model.model.layers)

    for layer_id, transformer_layer in enumerate(model.model.layers):
        layer_tp_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": PrepareModuleInputTuple(
                input_kwarg_layouts={
                    "hidden_states": Shard(1),
                    "position_embeddings": (Replicate(), Replicate()),
                    "attention_mask": Replicate(),
                    "position_ids": Replicate(),
                    "cu_seq_lens_q": Replicate(),
                    "cu_seq_lens_k": Replicate(),
                    "max_length_q": Replicate(),
                    "max_length_k": Replicate(),
                },
                desired_input_kwarg_layouts={
                    "hidden_states": Replicate(),
                    "position_embeddings": (Replicate(), Replicate()),
                    "attention_mask": Replicate(),
                    "position_ids": Replicate(),
                    "cu_seq_lens_q": Replicate(),
                    "cu_seq_lens_k": Replicate(),
                    "max_length_q": Replicate(),
                    "max_length_k": Replicate(),
                },
                # desired_input_layouts=(Replicate(), Replicate(), Replicate()),
            ),
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            "self_attn.q_norm": IndentityParallel(),
            "self_attn.k_norm": IndentityParallel(),
            "post_attention_layernorm": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1)),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
            "mlp.up_proj": ColwiseParallel(),
        }
        parallelize_module(
            module=transformer_layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )
    return model
    # for i in range(num_layers):

    
