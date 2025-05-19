#!/bin/bash
torchrun --nproc_per_node=8 tests/tp_sp_vescale/test_qwen3_moe.py