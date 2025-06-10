import argparse
import os

from transformers import AutoConfig, AutoProcessor

from veomni.checkpoint import bytecheckpoint_ckpt_to_state_dict, ckpt_to_state_dict
from veomni.models import save_model_weights
from veomni.utils import helper
import torch
from tqdm import tqdm

logger = helper.create_logger(__name__)


def merge_to_hf_pt(load_dirs: str, save_path: str, model_assets_dir: str = None):
    # save model in huggingface's format
    state_dicts = [ckpt_to_state_dict(
        save_checkpoint_path=load_dir,
        output_dir=save_path,
        ckpt_manager="dcp" #="bytecheckpoint" # dcp
        ) for load_dir in tqdm(load_dirs, desc="Loading checkpoints")]
    def avg_state_dicts(state_dicts):
        state_dict = {key: [d[key] for d in state_dicts] for key in state_dicts[0].keys()}
        for key in tqdm(list(state_dict.keys()), desc="Averaging checkpoints"):
            state_dict[key] = torch.mean(torch.stack(state_dict[key]), dim=0).to(state_dict[key][0].dtype)
        return state_dict
    state_dict = avg_state_dicts(state_dicts)
    # state_dict = {k: sum(d[k] for d in state_dicts) / len(state_dicts) for k in state_dicts[0].keys()}
    if model_assets_dir is not None:
        config = AutoConfig.from_pretrained(model_assets_dir)
        processor = AutoProcessor.from_pretrained(model_assets_dir, trust_remote_code=True)

        save_model_weights(save_path, state_dict, model_assets=[config, processor])
    else:
        save_model_weights(save_path, state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-dirs", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model_assets_dir", type=str, default=None)
    args = parser.parse_args()
    load_dirs = args.load_dirs.split(" ")
    print(load_dirs)
    save_dir = args.save_dir
    model_assets_dir = args.model_assets_dir
    logger.info(f"Merge Args: {args}")
    merge_to_hf_pt(load_dirs, save_dir, model_assets_dir)
    logger.info(f"Merge to hf pt success! Save to: {save_dir}")
