#!/bin/bash

# qwen3-tensor-parallel
torchrun --nproc_per_node=4 tests/tp_sp_vescale/test_qwen3.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tests/test_load_large_model_meta_device_fsdp2.py configs/sft/medical/17_05_2025/qwen3_8b_it_v2_23_05.yaml
# qwen3-moe-tensor-parallel
# TORCH_COMPILE_DEBUG=1 INDUCTOR_POST_FUSION_SVG=1 TORCH_LOGS="+inductor,+dynamo,+graph_breaks" TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 
# torchrun --nproc_per_node=8 tests/tp_sp_vescale/test_qwen3_moe.py
