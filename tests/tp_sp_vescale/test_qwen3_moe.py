
import torch._dynamo 
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True
torch._inductor.config.reorder_for_peak_memory = False
torch._dynamo.config.capture_scalar_outputs = True
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.models import build_foundation_model
import os
import torch
import torch.distributed as dist
from veomni.distributed.torch_parallelize import build_parallelize_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    print(f"world_size: {world_size}, rank: {rank}")
    # torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")

    
    init_parallel_state(dp_size=8, tp_size=1, ep_size=1, pp_size=1, cp_size=1, ulysses_size=1, dp_mode="fsdp2", device_type="cuda")
    parallel_state = get_parallel_state()
    print(parallel_state.ep_enabled)
    # exit(0)
    model = build_foundation_model(
        config_path="model_hub/qwen3/Qwen3-30B-A3B",
        weights_path="model_hub/qwen3/Qwen3-30B-A3B",
        init_device="cpu",
        config_kwargs=dict(num_hidden_layers=4)
        
    )
    if torch.distributed.get_rank() == 0:
        model.save_pretrained("model_hub/qwen3/Qwen3-30B-A3B-4-layers")
    exit(0)
    model = build_parallelize_model(
        model,
        init_device="cpu",
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=model._no_split_modules,
        weights_path="model_hub/qwen3/Qwen3-0.6B",
        enable_compile=False,
        use_orig_params=True
    )
    # model = model.cuda()s
    if torch.distributed.get_rank() == 0:
        print(model)
    model.train()
    
    # model = torch.compile(model, fullgraph=False, disable=False, dynamic=False)
    
    
    current_dp_rank = parallel_state.dp_rank
    current_tp_rank = parallel_state.tp_rank
    # inputs = torch.tensor(list(range(10))) + current_dp_rank
    # random create inputs with lengths is 16384
    inputs = torch.randint(0, 10000, (1, 20000))
    # position_ids = torch.zeros_like(inputs)
    position_ids = []
    current_i = 0
    for i in range(20000):
        position_ids.append(current_i)
        current_i += 1
        if i in [32, 512, 1024, 5182, 8192, 10000, 12384, 16384]:
            current_i = 0
    position_ids = torch.tensor(position_ids).long().reshape(1, -1).cuda()
    # position_ids = torch.tensor(list(range(10))).long().reshape(1, -1).cuda()
    # position_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]).long().reshape(1, -1).cuda()
    attention_mask = torch.ones_like(inputs).long().reshape(1, -1).cuda()
    position_ids_ = position_ids.flatten()
    indices_q = torch.arange(position_ids_.size(0), device=position_ids_.device, dtype=torch.int32)
    cu_seq_lens = torch.cat(
        (
            indices_q[position_ids_ == 0],
            torch.tensor(position_ids_.size(), device=position_ids_.device, dtype=torch.int32),
        )
    )
    max_length_q = position_ids_.max() + 1
    max_length_k = position_ids_.max() + 1
    cu_seq_lens_q = cu_seq_lens
    cu_seq_lens_k = cu_seq_lens
    # inputs = torch.tensor([current_dp_rank * 10 + current_tp_rank * i for i in range(10)])
    inputs = inputs.reshape(1, -1)
    if current_dp_rank == 0:
        print("tp_rank: ", current_tp_rank, "inputs: ", inputs)
    inputs = inputs.long().cuda()
    outputs = model(inputs, position_ids=position_ids, attention_mask=attention_mask, cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k, max_length_q=max_length_q, max_length_k=max_length_k)
    list_compile_time = []
    for _ in range(100):
        _, compile_time = timed(lambda: model(inputs, position_ids=position_ids, attention_mask=attention_mask, cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k, max_length_q=max_length_q, max_length_k=max_length_k))
        list_compile_time.append(compile_time)
        if torch.distributed.get_rank() == 0:
            print(f"compile_time: {compile_time} ms")
    list_compile_time.pop(0)
    if torch.distributed.get_rank() == 0:
        print(f"compile_time mean: {sum(list_compile_time) / len(list_compile_time)} ms")
    