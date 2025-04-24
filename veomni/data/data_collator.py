# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from ..distributed.parallel_state import get_parallel_state
from ..utils.seqlen_pos_transform_utils import len2culen, pos2culen
from .constants import IGNORE_INDEX


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


class CollatePipeline:
    def __init__(self, data_collators: Optional[Union[Callable, List[Callable]]] = None):
        """
        Args:
            data_collators: a list of data collators or a single data collator
        """

        if not isinstance(data_collators, (list, tuple)):
            data_collators = [data_collators]
        self.data_collators = data_collators

    def __call__(self, batch: Sequence[Dict[str, Any]]):
        """
        process data batch through data collators.

        Args:
            batch: the original input data batch

        Returns:
            batch: the processed data batch

        """
        for data_collator in self.data_collators:
            batch = data_collator(batch)
        return batch


@dataclass
class DataCollatorWithPadding(DataCollator):
    """
    Data collator with padding.
    """

    pad_token_id: int = 0

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = defaultdict(list)

        # batching features
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            # process padding features
            if key in ["input_ids", "attention_mask", "position_ids", "images_seq_mask"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=0)
            elif key in ["labels", "labels_image"]:
                batch[key] = pad_sequence(batch[key], batch_first=True, padding_value=IGNORE_INDEX)
            else:
                batch[key] = default_collate(batch[key])

        return batch


@dataclass
class DataCollatorWithPacking(DataCollator):
    """
    Data collator with packing.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        seqlens = torch.tensor([len(feature["input_ids"]) for feature in features], dtype=torch.long)
        batch = {"cu_seqlens": len2culen(seqlens)}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features])
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        return batch


@dataclass
class DataCollatorWithPositionIDs(DataCollator):
    """
    Data collator with packing by position ids.
    """
    max_seq_len: int = -1
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        batch = {}
        for input_name in features[0].keys():
            if input_name in ("input_ids", "attention_mask", "labels", "position_ids", "text_embedding"):
                batch[input_name] = torch.cat([feature[input_name] for feature in features], dim=-1).unsqueeze(0)
            else:
                batch[input_name] = default_collate([feature[input_name] for feature in features])

        if "position_ids" not in batch:
            batch["position_ids"] = torch.cat(
                [torch.arange(len(feature["input_ids"])) for feature in features]
            ).unsqueeze(0)

        if "labels" in batch:
            cu_seqlens = pos2culen(batch["position_ids"])
            batch["labels"][:, cu_seqlens[1:-1]] = IGNORE_INDEX
        # if self.max_seq_len != -1:
        #     if len(batch["input_ids"]) < self.max_seq_len:
        #         batch["input_ids"] = torch.cat([batch["input_ids"], torch.full((self.max_seq_len - len(batch["input_ids"]),), 0)])
        #         batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.full((self.max_seq_len - len(batch["attention_mask"]),), 0)])
        #         batch["position_ids"] = torch.cat([batch["position_ids"], torch.full((self.max_seq_len - len(batch["position_ids"]),), 0)])
        #         batch["labels"] = torch.cat([batch["labels"], torch.full((self.max_seq_len - len(batch["labels"]),), IGNORE_INDEX)])
        return batch

@dataclass
class DataCollatorWithExternalAPI(DataCollator):
    """
    Data collator with external API.
    """

    def __init__(self, use_external_api_url: str):
        self.use_external_api_url = use_external_api_url

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        texts = [feature["text"] for feature in features]
        text_embeddings = [self.call_embeddings(text) for text in texts]
        return text_embeddings

    def call_embeddings(self, text: str) -> List[float]:
        response = requests.post(
            self.use_external_api_url,
            json={"model": "", "input": text},
        )
        text_embedding = response.json()["data"][0]["embedding"]
        return text_embedding

@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used in dynamic batch dataloader at main process.
    """
    max_seq_len: int = -1
    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        features_out = [] 
        import torch
        # if torch.distributed.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()
        for feature in features: 
            if isinstance(feature, dict):
                features_out.append({
                    k:v.clone() for k,v in feature.items()
                })
            else:
                clone_feature = []
                for f in feature:
                    clone_feature.append({
                        k:v.clone() for k,v in f.items()
                    })
                features_out.extend(clone_feature)
                
        def padding_tensor(tensors, pad_value=-2, max_seq_len: int = -1):
            max_length = max(len(t) for t in tensors)
            
            list_padded_tensors = []
            for t in tensors:
                if len(t) < max_length:
                    t = torch.cat([t, torch.full((max_length - len(t),), pad_value)])
                else:
                    t = t[:max_length]
                list_padded_tensors.append(t)
            return torch.stack(list_padded_tensors, 0)
        
        final_features = {
            k: padding_tensor([feature[k] for feature in features_out]) for k in features_out[0].keys()
        }
        return final_features


@dataclass
class DataCollatorRemovePadding(DataCollator):
    """
    Data collator to remove padding, used in dynamic batch dataloader at main process.
    """

    def __call__(self, features: Dict[str, "torch.Tensor"]) -> List[Dict[str, "torch.Tensor"]]:
        def unpadding_tensor(tensor, pad_value=-2):
            return tensor[tensor != pad_value]
        keys = list(features.keys())
        list_features = []
        for index in range(features[keys[0]].shape[0]):
            list_features.append({
                k: unpadding_tensor(features[k][index]) for k in keys
            })
        return list_features

@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader at worker process.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        return features[0]


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        micro_batch_size = len(features) // self.num_micro_batch
        for i in range(len(features)):
            features[i] = features[i][0]  # 1-to-N inverse transform

        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            micro_batches.append(self.internal_data_collator(features[i : i + micro_batch_size]))

        return micro_batches


@dataclass
class TextSequenceShardCollator(DataCollator):
    """
    Data collator to chunk inputs according to sequence parallelism.
    Args:
        rmpad: whether the samples is packing or not.
        rmpad_with_pos_ids: whether the samples is packing by position ids or not.
        pad_token_id: the id of the padding token.
    """

    rmpad: bool
    rmpad_with_pos_ids: bool
    pad_token_id: int = 0
    max_seq_len: int = -1
    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """
        Slices a tensor along the specified dimension for sequence parallelism.
        """
        seq_length = tensor.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        # if torch.distributed.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()   
        # print(torch.distributed.get_rank(), self.sp_rank, sp_chunk_size)
        return tensor.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_length: int = 0
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        if pad_length == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_length
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        # if torch.distributed.get_rank() == 0:
        #     import ipdb; ipdb.set_trace()
        input_ids = batch.pop("input_ids")
        text_embedding = batch.pop("text_embedding", None)
        if text_embedding is not None:
            assert len(text_embedding) == len(input_ids), f'{len(text_embedding)} != {len(input_ids)}, length of text_embedding and input_ids must be the same'
        labels = batch.pop("labels")[..., 1:].contiguous()  # shift labels
        labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)

        if self.rmpad_with_pos_ids:  # mask the last token of each sequence
            cu_seqlens = pos2culen(batch["position_ids"])
            labels[:, cu_seqlens[1:-1] - 1] = IGNORE_INDEX
        elif self.rmpad:
            labels = labels.view(-1)
            labels[batch["cu_seqlens"][1:-1] - 1] = IGNORE_INDEX
        else:
            if "position_ids" not in batch:  # we should calculate the position ids before chunking
                batch["position_ids"] = torch.arange(0, input_ids.size(-1)).unsqueeze(0)

        # sp padding
        seq_length = input_ids.size(-1)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        pad_length = sp_chunk_size * self.sp_size - seq_length

        input_ids = self.sp_padding(input_ids, dim=-1, pad_value=self.pad_token_id, pad_length=pad_length)
        if text_embedding is not None:
            text_embedding = self.sp_padding(text_embedding, dim=-1, pad_value=0, pad_length=pad_length)
        labels = self.sp_padding(labels, dim=-1, pad_value=IGNORE_INDEX, pad_length=pad_length)

        if self.rmpad_with_pos_ids:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=1, pad_length=pad_length
            )
        else:
            batch["attention_mask"] = self.sp_padding(
                batch["attention_mask"], dim=-1, pad_value=0, pad_length=pad_length
            )

        if self.rmpad:
            if pad_length > 0:
                batch["cu_seqlens"] = F.pad(
                    batch["cu_seqlens"], (0, 1), "constant", batch["cu_seqlens"][-1].item() + pad_length
                )
        else:
            batch["position_ids"] = self.sp_padding(batch["position_ids"], dim=-1, pad_value=0, pad_length=pad_length)

        # sp slice
        batch["input_ids"] = self.sp_slice(input_ids, dim=-1)
        batch["labels"] = self.sp_slice(labels, dim=-1)
        if text_embedding is not None:
            batch["text_embedding"] = self.sp_slice(text_embedding, dim=-1)
        return batch
