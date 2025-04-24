"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union, Dict, Any
import torch
import math
import torch.distributed as dist
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from typing import Optional
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer
import warnings
from veomni.utils.dist_utils import main_process_first

def hf_tokenizer(name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f'tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f'tokenizer.pad_token is None. Now set to {tokenizer.eos_token}')

def process_item_row(row, tokenizer, messages_key, is_chatml):
    messages = row[messages_key]
    if is_chatml:
        text = "\n".join([msg['content'] for msg in messages])
        inputs = tokenizer(text, return_tensors='pt')['input_ids'][0]
        return {
            'input_ids': inputs,
            'attention_mask': torch.ones_like(inputs),
            'loss_mask': torch.ones_like(inputs, dtype=torch.long),
            "position_ids": torch.arange(len(inputs)),
            'lengths': len(inputs)
        }
    full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
    input_ids = full_tokens[0]  # The output is already a tensor
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
    # Process each message to find assistant responses
    for i, msg in enumerate(messages):
        # Get tokens for messages up to this point to find the start position
        prefix_messages = messages[:i+1]
        prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        
        # Get tokens for messages up to previous point
        prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None
        
        # Calculate start and end positions
        start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
        end_pos = prefix_tokens[0].shape[0]
        
        # If this is an assistant message, set loss mask
        if msg['role'] == 'assistant':
            loss_mask[start_pos:end_pos] = 1
    # Create position IDs
    # position_ids = torch.arange(len(input_ids), dtype=torch.long)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    # Zero out position IDs for padding
    position_ids = position_ids * attention_mask
    if hasattr(row, 'length'):
        length = row['length'] 
        assert len(input_ids) == length
    else:
        length = len(input_ids)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
        "lengths": length
    }


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self,
                 data_files: Union[str, List[str]],
                 tokenizer,
                 messages_key='messages',  # Key for the messages list in the parquet file
                 max_length=1024,
                 key="train",
                 device_mesh=None,
                 cut_of_length=1024,
                 recompute_input_ids=False,
                 cache_path=None,
                 is_chatml=False):
        
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.messages_key = messages_key
        if isinstance(data_files, str) == True:
            data_files = [data_files] 
        if cache_path is None:
            try:
                hf_datasets = [load_dataset(data_file, split=key) for data_file in data_files] 
            except Exception as e:
                print(e)
                hf_datasets = [load_from_disk(data_file) for data_file in data_files] 
        else:
            try:
                hf_datasets = [load_dataset(data_file, split=key, cache_dir=cache_path) for data_file in data_files] 
            except Exception as e:
                print(e)
                hf_datasets = [load_from_disk(data_file) for data_file in data_files] 
        from functools import partial
        self.is_chatml = is_chatml
        with main_process_first():
            hf_datasets[0 ] = hf_datasets[0].map(
                partial(process_item_row, tokenizer=tokenizer, messages_key=messages_key, is_chatml=is_chatml),
                batched=False,
                num_proc=256
            )
        print(max(hf_datasets[0]['lengths']), " Max length")
        with main_process_first():
            hf_datasets[0] = hf_datasets[0].filter(
                lambda x: x['lengths'] <= max_length,
                num_proc=256
            )
        self.final_dataset = hf_datasets[0] #concatenate_datasets(hf_datasets)#.select(range(10000))
        total_tokens = sum(self.final_dataset['lengths'])
        print(f"Total tokens: {total_tokens}")
       
        from functools import partial
        # self.final_dataset = self.final_dataset.map(partial(process_item_row, tokenizer=tokenizer, messages_key=messages_key), batched=False, num_proc=128)
        self.lengths = self.final_dataset['lengths']
        self.max_length = max_length
        if torch.distributed.get_rank() == 0:
            print(self.final_dataset)
            print(self.tokenizer.decode(self.final_dataset[0]['input_ids']))
            loss_mask = self.final_dataset[0]['loss_mask']
            # labels = [i for i in self.final_dataset[0]['input_ids'] if i != 0 else 0]
            labels = []
            for i, mask in zip(self.final_dataset[0]['input_ids'], loss_mask):
                if mask == 1:
                    labels.append(i)
                else:
                    labels.append(0)
            print(self.tokenizer.decode(labels))


    def __len__(self):
        return len(self.final_dataset)

    def process_item_row(self, row):
        messages = row[self.messages_key]
        tokenizer = self.tokenizer
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        # Process each message to find assistant responses
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[:i+1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
            
            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None
            
            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]
            
            # If this is an assistant message, set loss mask
            if msg['role'] == 'assistant':
                loss_mask[start_pos:end_pos] = 1
        # Create position IDs
        # position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask
        if hasattr(row, 'length'):
            length = row['length'] 
            assert len(input_ids) == length
        else:
            length = len(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            "lengths": length
        }
    

    def __getitem__(self, items):
        
        items = [self.final_dataset[item] for item in items]

        for i in range(len(items)):
            if len(items[i]['input_ids']) <= self.max_length:
                items[i]['input_ids'] = torch.LongTensor(items[i]['input_ids'])
                items[i]['attention_mask'] = torch.LongTensor(items[i]['attention_mask'])
                items[i]['loss_mask'][-1] = 0 # mask out the last token in the response
                items[i]['loss_mask'] = torch.tensor(items[i]['loss_mask'])
                items[i]['labels'] = items[i]['input_ids']#[ 1:] 
                # items[i]['labels'] = torch.cat([items[i]['labels'], torch.zeros(size=(1,), dtype=items[i]['labels'].dtype) - 100], dim=0)
                items[i]['position_ids'] =torch.tensor(items[i]['position_ids'])
                # if len(items[i]['input_ids']) < self.max_length:
                #     items[i]['input_ids'] = torch.cat([items[i]['input_ids'], torch.zeros(size=(self.max_length - len(items[i]['input_ids']),), dtype=items[i]['input_ids'].dtype)], dim=0)
                #     items[i]['attention_mask'] = torch.cat([items[i]['attention_mask'], torch.zeros(size=(self.max_length - len(items[i]['attention_mask']),), dtype=items[i]['attention_mask'].dtype)], dim=0)
                #     items[i]['loss_mask'] = torch.cat([items[i]['loss_mask'], torch.zeros(size=(self.max_length - len(items[i]['loss_mask']),), dtype=items[i]['loss_mask'].dtype)], dim=0)
                #     items[i]['labels'] = torch.cat([items[i]['labels'], torch.zeros(size=(self.max_length - len(items[i]['labels']),), dtype=items[i]['labels'].dtype) - 100], dim=0)
                #     items[i]['position_ids'] = torch.cat([items[i]['position_ids'], torch.zeros(size=(self.max_length - len(items[i]['position_ids']),), dtype=items[i]['position_ids'].dtype)], dim=0)
                items[i]['labels'] = torch.where(items[i]['loss_mask'] == 1, items[i]['labels'], torch.zeros_like(items[i]['labels']) - 100)
            else:   
                raise ValueError(f"Input ids length is greater than max length: {len(items[i]['input_ids'])}")
            items[i] = {
                "input_ids": items[i]['input_ids'],
                "attention_mask": items[i]['attention_mask'],
                "labels": items[i]['labels'],
            }
        return items
    
# def collate_fn(batch, micro_batch_size_per_gpu=1):
#     input_ids = [] 
#     attention_mask = []
#     position_ids = []
#     loss_mask = []
#     lengths = []
#     labels = []
#     current = []
#     new_batch = []
#     for item in batch:
#         current.append(len(item))
#         if len(current) == micro_batch_size_per_gpu:
#             lengths.append(sum(current))
#             current = []
#         new_batch.extend(item)
#     if len(current) > 0:
#         raise ValueError(f"Current batch size is not equal to micro_batch_size_per_gpu: {len(current)}")
#     input_ids = torch.stack([item['input_ids'] for item in new_batch], dim=0)
#     attention_mask = torch.stack([item['attention_mask'] for item in new_batch], dim=0)
#     position_ids = torch.stack([item['position_ids'] for item in new_batch], dim=0)
#     loss_mask = torch.stack([item['loss_mask'] for item in new_batch], dim=0)
#     labels = torch.stack([item['labels'] for item in new_batch], dim=0)
#     labels = torch.where(loss_mask == 1, labels, -100)
#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'position_ids': position_ids,
#         'loss_mask': loss_mask,
#         "labels": labels.long(),
#         "split_micro_batch_size": torch.LongTensor(lengths)
#     }

class DistributedBatchMultiTurnSFTDatasetSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_length: int = 1024,
        batch_size: int = 1,
        corss_pack: int = 100000
    ) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.corss_pack = corss_pack
        self.packing_manager = DynBszBuffer()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def __len__(self):
        return self.total

    def prepare2(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset))) 
        
        import numpy as np
        lengths = np.array(self.dataset.lengths)
        length_part = self.corss_pack
        indices_parts = [indices[i:i+length_part] for i in range(0, len(indices), length_part)] 

        packs_indices = []
        from tqdm import tqdm
        for index, indices_part in enumerate(tqdm(indices_parts)):
            for item in indices_part:
                self.packing_manager.append({
                    "index": item,
                    "length": lengths[item]
                })
            while self.packing_manager.all_token_cnt > self.max_length:
                packs = self.packing_manager.get_samples(self.max_length, force=True)
                packs_indices.append([pack["index"] for pack in packs])
        batch_size = self.batch_size

        # We always append the last batch for simplicity 
        # Todo: support drop last
        
        while (len(packs_indices)) % self.num_replicas != 0:
            packs_indices.append(packs_indices[-1])

        print(f"Packs indices: {len(packs_indices)}")
        packs_indices_rank = []
        for index, pack in enumerate(packs_indices):
            if index % self.num_replicas == self.rank:
                packs_indices_rank.append(pack)

        assert all([isinstance(pack, list) for pack in packs_indices_rank])
        self.total = len(packs_indices_rank)
        for batch_indices in packs_indices_rank:
            yield batch_indices

    def __iter__(self):
        # if self.shuffle:
        #     g = torch.Generator()
        #     g.manual_seed(self.seed + self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        # else:
        #     indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset))) 
        
        import numpy as np
        lengths = np.array(self.dataset.lengths)
        length_part = self.corss_pack
        indices_parts = [indices[i:i+length_part] for i in range(0, len(indices), length_part)] 

        packs_indices = []
        cnt = 0
        for index, indices_part in enumerate(indices_parts):
            for item in indices_part:
                self.packing_manager.append({
                    "index": item,
                    "length": lengths[item]
                })
            while self.packing_manager.all_token_cnt >= self.max_length:
                packs = self.packing_manager.get_samples(self.max_length, force=True)
                # packs_indices.append([pack["index"] for pack in packs])
                packs_indices = [pack["index"] for pack in packs]
                if cnt % self.num_replicas == self.rank:
                    yield packs_indices
                cnt += 1
        remain_rank = cnt % self.num_replicas # 1, 2 
        if remain_rank != 0:
            remain_rank = range(remain_rank, self.num_replicas) 
            for i in remain_rank:
                if cnt % self.num_replicas == self.rank:
                    yield packs_indices
                cnt += 1

        # if cnt % self.num_replicas !=0:
        #     yield packs_indices
        #     cnt += 1
        # return 
        # batch_size = self.batch_size

        # # We always append the last batch for simplicity 
        # # Todo: support drop last
        
        # while (len(packs_indices)) % self.num_replicas != 0:
        #     packs_indices.append(packs_indices[-1])

        # print(f"Packs indices: {len(packs_indices)}")
        # packs_indices_rank = []
        # for index, pack in enumerate(packs_indices):
        #     if index % self.num_replicas == self.rank:
        #         packs_indices_rank.append(pack)

        # assert all([isinstance(pack, list) for pack in packs_indices_rank])
        # self.total = len(packs_indices_rank)
        # for batch_indices in packs_indices_rank:
        #     yield batch_indices

        
        # # Do sampling Lenghts here
        # # First we  split the indices to 32 parts and sort the indices by the lengths and merge them
        # import numpy as np
        # lengths = np.array(self.dataset.lengths)
        # length_part = self.corss_pack
        # indices_parts = [indices[i:i+length_part] for i in range(0, len(indices), length_part)] 
        # indices_parts = [sorted(indices_part, key=lambda x: lengths[x]) for indices_part in indices_parts]
        # indices = []
        # for indices_part in indices_parts:
        #     indices.extend(indices_part)
        # packs_indices = []
        # packs = []

        # current_pack = []
        # current_pack_index = []

        # for index in indices:
        #     length = lengths[index]
        #     if sum(current_pack) + length <= self.max_length:
        #         current_pack.append(length)
        #         current_pack_index.append(index)
        #     else:
        #         packs.append(current_pack[:])
        #         packs_indices.append(current_pack_index[:])
        #         current_pack = [length]
        #         current_pack_index = [index]
        # if len(current_pack) > 0:
        #     packs.append(current_pack[:])
        #     packs_indices.append(current_pack_index[:])
        
        
        # batch_size = self.batch_size

        # # We always append the last batch for simplicity 
        # # Todo: support drop last
        
        # while (len(packs_indices)) % self.num_replicas != 0:
        #     packs_indices.append(packs_indices[0])

        # print(f"Packs indices: {len(packs_indices)}")
        # packs_indices_rank = []
        # for index, pack in enumerate(packs_indices):
        #     if index % self.num_replicas == self.rank:
        #         packs_indices_rank.append(pack)

        # assert all([isinstance(pack, list) for pack in packs_indices_rank])
        # self.total = len(packs_indices_rank)
        # for batch_indices in packs_indices_rank:
        #     yield batch_indices
    
    def get_state_dict(self):
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "drop_last": self.drop_last,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
        }
    
    def load_state_dict(self, state_dict):
        self.seed = state_dict["seed"]
        self.epoch = state_dict["epoch"]
        self.drop_last = state_dict["drop_last"]
        self.max_length = state_dict["max_length"]
        self.batch_size = state_dict["batch_size"]
        self.shuffle = state_dict["shuffle"]



import bisect
from typing import List, Sequence, Tuple


class DynBszBuffer:
    """
    A buffer to store samples for dynamic batch size.
    """

    def __init__(self):
        self._buffer = []
        self._buffer_sample_lens = []
        self.del_idxs = []
        self.cur_idx = 0
        self.all_token_cnt = 0

    def append(self, item: Dict[str, Any]):
        """
        Append a sample to the buffer.
        Args:
            item: a sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        self._buffer_sample_lens.append(item["length"])
        self.all_token_cnt += self._buffer_sample_lens[-1]

    def get_samples(self, n_token_per_iter: int, force: bool = True):
        """
        get samples from the buffer.
        Args:
            n_token_per_iter: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.
        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < n_token_per_iter:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            if self.cur_idx not in self.del_idxs and (
                (force is True and cum_seq_len == 0) or (seq_len <= n_token_per_iter - cum_seq_len)
            ):
                cum_seq_len += seq_len
                samples.append(self._buffer[self.cur_idx])
                self.del_idxs.append(self.cur_idx)
            self.cur_idx += 1
        # print(cum_seq_len)
        if force:
            assert len(samples) > 0
        self.flush()
        return samples

    def __len__(self):
        return len(self._buffer)

    def flush(self):
        """ "
        Flush the buffer.
        """
        self.cur_idx = 0
        self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in self.del_idxs])
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx] for idx in range(buffer_len) if idx not in self.del_idxs
        ]
        self.del_idxs = []

    def merge(self, buffer_to_merge: "DynBszBuffer"):
        """ "
        Merge the buffer with another buffer.
        Args:
            buffer_to_merge: the buffer to merge.
        """
        self.flush()
        buffer_to_merge.flush()
        for item in buffer_to_merge._buffer:
            self.append(item)


            