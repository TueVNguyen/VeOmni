
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.models import build_foundation_model
import os
import torch
import torch.distributed as dist
from veomni.distributed.torch_parallelize import build_parallelize_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    print(f"world_size: {world_size}, rank: {rank}")
    # torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")

    
    init_parallel_state(dp_size=4, tp_size=2, ep_size=1, pp_size=1, cp_size=1, ulysses_size=1, dp_mode="fsdp1", device_type="cuda")
    parallel_state = get_parallel_state()
    print(parallel_state.ep_enabled)
    # exit(0)
    model = build_foundation_model(
        config_path="/home/slurm/tuenv2/tuenv/model_hub/qwen3/Qwen3-0.6B",
        weights_path=None,
        init_device="cpu",
    )
    model = model.cuda()
    # apply tensor parallel plan 
    from veomni.models.transformers.qwen3.parallel_plan import apply_tensor_parallel_plan
    print(parallel_state.tp_mesh.size())
    apply_tensor_parallel_plan(model, parallel_state.tp_mesh)
    fsdp_config = {
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
        # "cpu_offload": CPUOffloadPolicy(offload_params=False),
        "mesh": parallel_state.fsdp_mesh,
    }
    fully_shard(model, **fsdp_config, reshard_after_forward=True)

    current_dp_rank = parallel_state.dp_rank
    current_tp_rank = parallel_state.tp_rank
    inputs = torch.tensor(list(range(10))) + current_dp_rank
    position_ids = torch.tensor(list(range(10))).long().reshape(1, -1).cuda()
    position_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]).long().reshape(1, -1).cuda()
    # inputs = torch.tensor([current_dp_rank * 10 + current_tp_rank * i for i in range(10)])
    inputs = inputs.reshape(1, -1)
    if current_dp_rank == 0:
        print("tp_rank: ", current_tp_rank, "inputs: ", inputs)
    inputs = inputs.long().cuda()
    outputs = model(inputs, position_ids=position_ids)
        # exit(0)

#     # exit(0)
#     model = build_parallelize_model(
#         model,
#         init_device="cpu",
#         weights_path=None,
#         enable_full_shard=True,
#         enable_mixed_precision=False,
#         enable_gradient_checkpointing=False,
#         enable_fsdp_offload=False,
#         basic_modules=model._no_split_modules
#     )

# print(model)

# from vescale.dtensor.placement_types import Replicate, Shard


# megatron_lm_plan = {
#     "model.embed_tokens.weight": [Replicate()], # it is not sharded


#     r"model.layers.\d+.input_layernorm.weight": [Replicate()],

#     r"model.layers.\d+.self_attn.q_proj.weight": [Shard(0)],
#     r"model.layers.\d+.self_attn.k_proj.weight": [Shard(0)],
#     r"model.layers.\d+.self_attn.v_proj.weight": [Shard(0)],
#     r"model.layers.\d+.self_attn.o_proj.weight": [Shard(1)],
#     r"model.layers.\d+.self_attn.q_norm.weight": [Replicate()],
#     r"model.layers.\d+.self_attn.k_norm.weight": [Replicate()],
#     # mlp layers 
#     r"model.layers.\d+.mlp.experts.\d+.gate_proj.weight": [Shard(1)],
#     r"model.layers.\d+.mlp.experts.\d+.up_proj.weight": [Shard(1)],
#     r"model.layers.\d+.mlp.experts.\d+.down_proj.weight": [Shard(0)],
#     r"model.layers.\d+.post_attention_layernorm.weight": [Replicate()],
#     r"model.layers.\d+.mlp.gate.weight": [Replicate()],

#     r"model.layers.\d+.post_attention_layernorm.weight": [Replicate()],

 
    
# }

# fwd_resharding_plan = {
#     # TODO: buggy: attn mask is torch.Tensor, in training, it's a None
#     r".input": {"input_ids": [Replicate()], "attention_mask": [Replicate()]},
#     "model.embed_tokens.input": [[Replicate()]],
#     # No SP
#     # r"layers.\d+.input_layernorm.input": [[Replicate()]],
#     # r"layers.\d+.input_layernorm.output": [[Replicate()]],
#     # SP
#     r"model.layers.\d+.input_layernorm.input": [[Shard(1)]],
#     r"model.layers.\d+.input_layernorm.output": [[Shard(1)]],
#     r"model.layers.\d+.self_attn.input": [[Replicate()]],
#     r"model.layers.\d+.self_attn.output": {
#         "attn_output": [Replicate()],
#         "attn_weights": None,
#         "past_key_value": None,
#     },
#     r"model.layers.\d+.self_attn.o_proj.output": [[Replicate()]],
#     # No SP
#     # r"model.layers.\d+.post_attention_layernorm.input": [[Replicate()]],
#     # r"model.layers.\d+.post_attention_layernorm.output": [[Replicate()]],
#     # SP
#     r"model.layers.\d+.post_attention_layernorm.input": [[Shard(1)]],
#     r"model.layers.\d+.post_attention_layernorm.output": [[Shard(1)]],
#     r"model.layers.\d+.mlp.input": [[Replicate()]],
#     r"model.layers.\d+.mlp.gate.output": [[Replicate()]],
#     r"model.layers.\d+.mlp.output": {
#         "final_hidden_states": [Replicate()],
#         "router_logits": [Replicate()],
#     },
#     r"model.layers.\d+.mlp.experts.\d+.gate_proj.input": [[Replicate()]],
#     r"model.layers.\d+.mlp.experts.\d+.up_proj.input": [[Replicate()]],
#     r"model.layers.\d+.mlp.experts.\d+.down_proj.output": [[Replicate()]],
#     "model.norm.input": [[Replicate()]],
# }

# mixtral_plan = {"parameter": megatron_lm_plan, "forward": fwd_resharding_plan}

# from vescale.dmodule import parallelize_module
# from vescale.devicemesh_api import VESCALE_DEVICE_MESH 
# VESCALE_DEVICE_MESH.init_device_mesh("cuda", mesh_shape=(2, 4), mesh_dim_names=("DP", "TP"))
# # mesh_2d = VESCALE_DEVICE_MESH.get_device_mesh()

# module = parallelize_module(model,VESCALE_DEVICE_MESH['TP'], mixtral_plan)