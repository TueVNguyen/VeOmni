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


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

import os
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal, Optional

from torch import distributed as dist
from torch.distributed import get_process_group_ranks
import torch
from ..utils import logging
from ..utils.import_utils import is_torch_version_greater_than


if is_torch_version_greater_than("2.4"):
    from torch.distributed.device_mesh import init_device_mesh


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

_PARALLEL_STATE: "ParallelState" = None
_MESH_DIM_MAP_NAME_VESCALE = {
    "dp": "DP",
    "tp": "TP",
    "pp": "PP",
    "cp": "CP",
    "ulysses": "ULY",
    "sp": "ULY",
    "ep": "EP",
    "ep_dp": "EP_DP",
    
}

def requires_mesh(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(self: "ParallelState", *args, **kwargs):
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized.")

        return fn(self, *args, **kwargs)

    return _inner


@dataclass(frozen=True)
class ParallelState:
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ulysses_size: int = 1
    hsdp_size: int = 1
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp1"
    device_type: str = "cuda"
    include_sp_in_fsdp: bool = True
    device_mesh: Optional["DeviceMesh"] = None
    sp_device_mesh: Optional["DeviceMesh"] = None
    usp_device_mesh: Optional["DeviceMesh"] = None
    ep_device_mesh: Optional["DeviceMesh"] = None
    device_mesh_fsdp: Optional["DeviceMesh"] = None

    def __post_init__(self):
        if not self.include_sp_in_fsdp:
            raise NotImplementedError("Decoupled sequence parallel has not been implemented.")

        if self.cp_size > 1:
            raise NotImplementedError("Ring attention is not supported yet.")

        if self.pp_size * self.dp_size * self.tp_size * self.cp_size * self.ulysses_size != self.world_size:
            raise ValueError("The product of parallel sizes should be equal to the world size.")

        if self.sp_enabled:
            from ..distributed.sequence_parallel import (
                init_sequence_parallel,
                set_context_parallel_group,
                set_data_parallel_group,
                set_ulysses_sequence_parallel_group,
                set_unified_sequence_parallel_group,
            )

            if self.sp_device_mesh is not None:
                set_data_parallel_group(self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["dp"]))
                if self.ulysses_size > 1:
                    set_ulysses_sequence_parallel_group(self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["ulysses"]))
                if self.cp_size > 1:
                    set_context_parallel_group(self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["cp"]))
                # set unified sequence parallel group
                set_unified_sequence_parallel_group(self.usp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["sp"]))
            else:
                init_sequence_parallel(
                    ulysses_size=self.ulysses_size,
                    sep_dp=True,
                    ulysses_group_key="default",
                    cp_size=self.cp_size,
                )

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        if self.is_initialized:
            return dist.get_rank()
        return -1

    @property
    def world_size(self) -> int:
        if self.is_initialized:
            return dist.get_world_size()
        return 1

    # ------------------------------ DP ------------------------------ #
    @property
    def dp_group(self) -> Optional["ProcessGroup"]:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["dp"])

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_group

            return get_data_parallel_group()

        return self.fsdp_group

    @property
    def dp_rank(self) -> int:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["dp"])

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_rank

            return get_data_parallel_rank()

        return self.fsdp_rank

    @property
    @requires_mesh
    def dp_mesh(self) -> "DeviceMesh":

        if self.sp_device_mesh is not None:
            return self.sp_device_mesh[_MESH_DIM_MAP_NAME_VESCALE["dp"]]

        raise self.fsdp_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    # ----------------------------- FSDP ----------------------------- #
    @property
    def fsdp_group(self) -> Optional["ProcessGroup"]:
        # For fsdp, we will use the device mesh from dp group and resize it to [hsdp_size, fsdp_size]
        # To do this, efficiently,
        if self.device_mesh is not None:

            return self.device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["dp"])

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["dp"])

        return self.global_rank

    @property
    @requires_mesh
    def fsdp_mesh(self) -> "DeviceMesh":
        if self.hsdp_size > 1:
            return self.device_mesh_fsdp['hsdp', _MESH_DIM_MAP_NAME_VESCALE["dp"]]
            # from torch.distributed.device_mesh import DeviceMesh
            # dp_mesh = self.device_mesh[_MESH_DIM_MAP_NAME_VESCALE["dp"]]
            # list_ranks_dp = get_process_group_ranks(self.fsdp_group)
            # mesh = torch.tensor(list_ranks_dp, device="cpu", dtype=torch.int)
            # device_mesh = DeviceMesh(
            #         device_type=dp_mesh.device_type,
            #         mesh=mesh,
            #         mesh_dim_names=['hsdp', 'fsdp'],
            #         _init_backend=False,
            #     )
            # return device_mesh
                
        return self.device_mesh[_MESH_DIM_MAP_NAME_VESCALE["dp"]]

    @property
    def fsdp_enabled(self) -> bool:
        return self.fsdp_size > 1

    @property
    def fsdp_size(self) -> int:
        return self.world_size // (self.pp_size * self.tp_size)

    # ------------------------------ TP ------------------------------ #
    @property
    @requires_mesh
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["tp"])

    @property
    @requires_mesh
    def tp_mesh(self) -> "DeviceMesh":
        return self.device_mesh[_MESH_DIM_MAP_NAME_VESCALE["tp"]]

    @property
    def tp_group(self) -> Optional["ProcessGroup"]:
        return self.device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["tp"])

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    # ------------------------------ PP ------------------------------ #
    @property
    @requires_mesh
    def pp_rank(self) -> int:
        return self.device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["pp"])

    @property
    @requires_mesh
    def pp_mesh(self) -> "DeviceMesh":
        return self.device_mesh[_MESH_DIM_MAP_NAME_VESCALE["pp"]]

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    @requires_mesh
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    @requires_mesh
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)

    # ------------------------------ EP ------------------------------ #
    @property
    @requires_mesh
    def ep_mesh(self) -> "DeviceMesh":
        return self.ep_device_mesh[_MESH_DIM_MAP_NAME_VESCALE["ep"]]

    @property
    @requires_mesh
    def ep_group(self) -> "ProcessGroup":
        return self.ep_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["ep"])

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    # ------------------------------ SP ------------------------------ #
    @property
    def sp_group(self) -> Optional["ProcessGroup"]:
        if self.usp_device_mesh is not None:
            return self.usp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["sp"])

        if self.sp_enabled:
            from .sequence_parallel import get_unified_sequence_parallel_group

            return get_unified_sequence_parallel_group()

        return None

    @property
    def sp_rank(self) -> int:
        if self.usp_device_mesh is not None:
            return self.usp_device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["sp"])

        if self.sp_enabled:
            from .sequence_parallel import get_unified_sequence_parallel_rank

            return get_unified_sequence_parallel_rank()

        return -1

    @property
    def sp_enabled(self) -> bool:
        return self.cp_size > 1 or self.ulysses_size > 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_size * self.cp_size

    @property
    def ulysses_group(self) -> Optional["ProcessGroup"]:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["ulysses"])

        if self.sp_enabled:
            from .sequence_parallel import get_ulysses_sequence_parallel_group

            return get_ulysses_sequence_parallel_group()

        return None

    @property
    def ulysses_rank(self) -> int:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["ulysses"])

        if self.sp_enabled:
            from .sequence_parallel import get_ulysses_sequence_parallel_rank

            return get_ulysses_sequence_parallel_rank()

        return -1

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_group(self) -> Optional["ProcessGroup"]:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_group(_MESH_DIM_MAP_NAME_VESCALE["cp"])

        if self.sp_enabled:
            from .sequence_parallel import get_context_parallel_group

            return get_context_parallel_group()

        return None

    @property
    def cp_rank(self) -> int:
        if self.sp_device_mesh is not None:
            return self.sp_device_mesh.get_local_rank(_MESH_DIM_MAP_NAME_VESCALE["cp"])

        if self.sp_enabled:
            from .sequence_parallel import get_context_parallel_rank

            return get_context_parallel_rank()

        return -1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1


def init_parallel_state(
    dp_size: int,
    tp_size: int,
    ep_size: int,
    pp_size: int,
    cp_size: int,
    ulysses_size: int,
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"],
    hsdp_size: int=1,
    device_type: str = "cuda",
    include_sp_in_fsdp: bool = True,
) -> None:
    """
    Initializes global parallel state.
    """
    global _PARALLEL_STATE
    if _PARALLEL_STATE is not None:
        logger.warning("Parallel state has already been initialized.")
        return

    device_mesh, sp_device_mesh, usp_device_mesh, ep_device_mesh = None, None, None, None
    if is_torch_version_greater_than("2.4"):
        print("init_parallel_state here")
        fsdp_size = dist.get_world_size() // (pp_size * tp_size)
        assert fsdp_size % hsdp_size == 0, "hsdp_size must be a factor of fsdp_size"
        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=(pp_size, fsdp_size, tp_size),
            mesh_dim_names=(_MESH_DIM_MAP_NAME_VESCALE["pp"], _MESH_DIM_MAP_NAME_VESCALE["dp"], _MESH_DIM_MAP_NAME_VESCALE["tp"]),
        )
        device_mesh2 =init_device_mesh(
            device_type=device_type,
            mesh_shape=(pp_size, hsdp_size, fsdp_size // hsdp_size, tp_size),
            mesh_dim_names=(_MESH_DIM_MAP_NAME_VESCALE["pp"], "hsdp" ,_MESH_DIM_MAP_NAME_VESCALE["dp"], _MESH_DIM_MAP_NAME_VESCALE["tp"]),
        )
        if ulysses_size > 1 or cp_size > 1:
            sp_device_mesh = init_device_mesh(
                device_type=device_type,
                mesh_shape=(dp_size, cp_size, ulysses_size),
                mesh_dim_names=(_MESH_DIM_MAP_NAME_VESCALE["dp"], _MESH_DIM_MAP_NAME_VESCALE["cp"], _MESH_DIM_MAP_NAME_VESCALE["ulysses"]),
            )
            usp_device_mesh = init_device_mesh(
                device_type=device_type,
                mesh_shape=(dp_size, cp_size * ulysses_size),
                mesh_dim_names=(_MESH_DIM_MAP_NAME_VESCALE["dp"], _MESH_DIM_MAP_NAME_VESCALE["sp"]),
            )
        # TODO: support ep_size != dp_size
        if ep_size > 1:
            assert fsdp_size % ep_size == 0, "ep_size must be a factor of dp_size"
            ep_dp_size = fsdp_size // ep_size
            ep_device_mesh = init_device_mesh(
                device_type=device_type,
                mesh_shape=(pp_size, ep_dp_size, ep_size, tp_size),
                mesh_dim_names=(_MESH_DIM_MAP_NAME_VESCALE["pp"], _MESH_DIM_MAP_NAME_VESCALE["ep_dp"], _MESH_DIM_MAP_NAME_VESCALE["ep"], _MESH_DIM_MAP_NAME_VESCALE["tp"]),
            )

    _PARALLEL_STATE = ParallelState(
        dp_size=dp_size,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode=dp_mode,
        hsdp_size=hsdp_size,
        device_type=device_type,
        include_sp_in_fsdp=include_sp_in_fsdp,
        device_mesh=device_mesh,
        sp_device_mesh=sp_device_mesh,
        usp_device_mesh=usp_device_mesh,
        ep_device_mesh=ep_device_mesh,
        device_mesh_fsdp=device_mesh2,
    )


def get_parallel_state() -> "ParallelState":
    """
    Returns global parallel state.
    """
    if _PARALLEL_STATE is None:
        logger.warning_once("Parallel state has not been initialized. returning default Single-process state.")
        return ParallelState()

    return _PARALLEL_STATE
