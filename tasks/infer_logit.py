import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List
import sys
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
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)
def setup_ddp():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def collated_fn(examples):
    input_ids = []
    attention_mask = []
    lengths = []
    for example in examples:
        assert len(example) == 1
        input_ids.append(example[0]['input_ids'])
        attention_mask.append(example[0]['attention_mask'])
        lengths.append(example[0]['input_ids'].shape[0])
    max_length = max(lengths)
    input_ids = [torch.cat([x, torch.zeros(max_length - x.shape[0], dtype=torch.long)]) for x in input_ids]
    attention_mask = [torch.cat([x, torch.zeros(max_length - x.shape[0], dtype=torch.long)]) for x in attention_mask]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "lengths": torch.tensor(lengths),
    }
    
def main():
    dataset_name_or_path = "/home/slurm/tuenv2/tuenv/hf/r1_math_only/"
    model_name = "Qwen/QwQ-32B"
    chat_template = "chatml"
    max_seq_len = 32768
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_template = build_chat_template(chat_template, tokenizer)
    transform = partial(
        process_sft_example,
        chat_template=chat_template,
        max_seq_len=max_seq_len,
        text_keys=["messages"],
    )
    from datasets import load_from_disk, load_dataset
    dataset = load_dataset(dataset_name_or_path)
    dataset = dataset.map(lambda x: transform(x)[0], batched=False, num_proc=10)
    total_len = len(dataset)
    list_indices = list(range(total_len)) 

    part = np.array_split(list_indices, 8)[int(sys.argv[1])]
    
    dataset = dataset.select(part)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collated_fn, drop_last=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to("cuda")
    p = int(sys.argv[1])
    FOLDER_SAVE = "/home/slurm/tuenv2/tuenv/hf/r1_math_only/chunks_qwq32b/"
    os.makedirs(FOLDER_SAVE, exist_ok=True)
    save_path = os.path.join(FOLDER_SAVE, f"{p}.npy")
    
    with open(save_path, "wb") as f:
        for index_batch, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            lengths = batch["lengths"].to("cuda")
            if index_batch % 100 == 0:
                logger.info(f"Processing Part {p}: batch {index_batch} of {len(dataloader)}")
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                for index, l in enumerate(lengths):
                    np.save(f, logits[index, :l])

if __name__ == "__main__":
    main()