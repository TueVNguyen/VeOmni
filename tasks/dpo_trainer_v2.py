import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
    build_dataloader_v2
)
from veomni.data.simple_dataloader.dataset import DistributedBatchMultiTurnSFTDatasetSampler, MultiTurnSFTDataset
from veomni.data.batching_strategy import KeepInOrderStrategy

from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from functools import partial

logger = helper.create_logger(__name__)



@dataclass
class DPOArguments:
    name: str = "dpo"
    train_split: str = "train"
    stage: str = "precompute"

@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    dpo: "DPOArguments" = field(default_factory=DPOArguments)

def tokenizer_chatml(tokenizer, row):
    prompt = row["prompt"]
    chosen = row["chosen"]
    rejected = row["rejected"]
    system = None 
    if "system" in row and row["system"] is not None and len(row["system"].strip()) > 0:
        system = row["system"]
    
    chosen_messages = [
        {"role": "system", "content": system} if system is not None else None,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": chosen}
    ]

    rejected_messages = [
        {"role": "system", "content": system} if system is not None else None,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": rejected}
    ]

    def get_offset(messages):
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        input_ids = full_tokens[0] 
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        start_pos, end_pos = None, None
        for i, msg in enumerate(messages):
            prefix_messages = messages[:i+1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
            
            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None
            
            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]
            
            if msg['role'] == 'assistant':
                start_pos = start_pos
                end_pos = end_pos
                loss_mask[start_pos:end_pos] = 1

        if start_pos is None or end_pos is None:
            raise ValueError("No assistant message found in the messages")

        return input_ids, loss_mask, start_pos, end_pos

    chosen_input_ids, chosen_loss_mask, chosen_start_pos, chosen_end_pos = get_offset(chosen_messages)
    rejected_input_ids, rejected_loss_mask, rejected_start_pos, rejected_end_pos = get_offset(rejected_messages)
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_loss_mask": chosen_loss_mask,
        "chosen_start_pos": chosen_start_pos,
        "chosen_end_pos": chosen_end_pos,
        "chosen_position_ids": torch.arange(len(chosen_input_ids)),
        "rejected_input_ids": rejected_input_ids,
        "rejected_loss_mask": rejected_loss_mask,
        "rejected_start_pos": rejected_start_pos,
        "rejected_end_pos": rejected_end_pos,
        "rejected_position_ids": torch.arange(len(rejected_input_ids)),
    }

def collated_fn(batch):
    # We will pad to [2, flatten_length]
    index = []
    chosen_input_ids = []
    chosen_position_ids = []
    choosen_start_pos = []
    choosen_end_pos = []
    rejected_input_ids = []
    rejected_position_ids = []
    rejected_start_pos = []
    rejected_end_pos = []
    
    max_chosen_length = max([item["chosen_input_ids"].shape[0] for item in batch])
    max_rejected_length = max([item["rejected_input_ids"].shape[0] for item in batch]) 

    input_ids_choosen_orign = torch.zeros(len(batch), max_chosen_length)
    input_ids_rejected_orign = torch.zeros(len(batch), max_rejected_length) 

    input_ids_choosen_mask = torch.zeros(len(batch), max_chosen_length)
    input_ids_rejected_mask = torch.zeros(len(batch), max_rejected_length)

    for batch_idx in range(len(batch)):
        batch_item = batch[batch_idx]
        chosen_input_ids.append(batch_item["chosen_input_ids"].view(-1,))
        chosen_position_ids.append(batch_item["chosen_position_ids"].view(-1,))
        rejected_input_ids.append(batch_item["rejected_input_ids"].view(-1,))
        rejected_position_ids.append(batch_item["rejected_position_ids"].view(-1,))
        input_ids_choosen_orign[batch_idx, :chosen_input_ids.shape[0]] = 1
        input_ids_rejected_orign[batch_idx, :rejected_input_ids.shape[0]] = 1

        input_ids_choosen_mask[batch_idx, batch_item["chosen_start_pos"]:batch_item["chosen_end_pos"]] = 1
        input_ids_rejected_mask[batch_idx, batch_item["rejected_start_pos"]:batch_item["rejected_end_pos"]] = 1

        choosen_start_pos.append(batch_item["chosen_start_pos"])
        choosen_end_pos.append(batch_item["chosen_end_pos"])
        rejected_start_pos.append(batch_item["rejected_start_pos"])
        rejected_end_pos.append(batch_item["rejected_end_pos"])
        index.append(batch_item['index'])

    
    return {
        "input_ids_choosen": torch.cat(chosen_input_ids, dim=0),
        "input_ids_rejected": torch.cat(rejected_input_ids, dim=0),
        "choosen_start_pos": choosen_start_pos,
        "choosen_end_pos": choosen_end_pos,
        "rejected_start_pos": rejected_start_pos,
        "rejected_end_pos": rejected_end_pos,

        "position_ids_choosen": torch.cat(chosen_position_ids, dim=0),
        "position_ids_rejected": torch.cat(rejected_position_ids, dim=0),

        "input_ids_choosen_orign": input_ids_choosen_orign,
        "input_ids_rejected_orign": input_ids_rejected_orign,

        "input_ids_choosen_mask": input_ids_choosen_mask,
        "input_ids_rejected_mask": input_ids_rejected_mask,

        "index": index,

    }
    
def main():
    import datetime
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)

    # First preprocess the data 

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    train_dataset_path = args.data.train_path 

    train_dataset = load_dataset(train_dataset_path)[
        args.dpo.train_split
    ]
    
    if "prompt" not in train_dataset.column_names:
        raise ValueError("prompt column not found in dataset")
    
    if "chosen" not in train_dataset.column_names:
        raise ValueError("chosen column not found in dataset")
    
    if "rejected" not in train_dataset.column_names:
        raise ValueError("rejected column not found in dataset")
    
    logger.info_rank0(f"Found {len(train_dataset)} examples in the dataset")

    logger.info_rank0("Build chat template for the dataset")


    train_dataset = train_dataset.map(partial(tokenizer_chatml, tokenizer=tokenizer), num_proc=32)
    train_dataset = train_dataset.add_column("index", list(range(len(train_dataset))))
    parallel_state = get_parallel_state()
    # We will distributed sampler here 
    sampler = DistributedSampler(train_dataset, batch_size=args.train.batch_size, shuffle=False, num_replicas=parallel_state.dp_size, rank=parallel_state.dp_rank, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size, shuffle=False, collate_fn=collated_fn, sampler=sampler)

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")
    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        n_layer_gradient_checkpointing=args.train.n_layer_gradient_checkpointing,
    )

    model = model.eval()

    precompute_outs = []
    precompute_index = []
    if args.dpo.stage == "precompute":
        logger.info_rank0("Start precompute")
        for step, batch in enumerate(train_loader):
            index = batch["index"]
            logger.info_rank0(f"Precompute step {step} / {len(train_loader)}")
            with torch.no_grad():
                outputs = model(
                    input_ids=batch["input_ids_choosen"],
                    position_ids=batch["position_ids_choosen"],
                    attention_mask=batch["input_ids_choosen_mask"],
                )
                    # gather all the outputs from all the ranks 
                tensor_output = outputs.logits  # bs, 1 

                tensor_output = tensor_output.view(-1,)

                dp_size = parallel_state.dp_size
            
                tensor_output_list = [torch.zeros_like(tensor_output) for _ in range(dp_size)]
                dist.all_gather(tensor_output_list, tensor_output)
                index  = torch.tensor(index, dtype=torch.long).reshape(-1,)
                index_list = [torch.zeros_like(index) for _ in range(dp_size)]
                dist.all_gather(index_list, index)
                if args.train.global_rank == 0:
                    precompute_outs.extend(tensor_output_list.detach().cpu())
                    precompute_index.extend(index_list.detach().cpu().tolist())
        

                    
                    
                    
                    

                    
                








