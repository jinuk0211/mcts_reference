from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse
from datetime import datetime

from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from agents import SBSREACT
from agents import MCTS
from solver import Solver
from config import BaseConfig
# from react_demo import load_qaf

from typing import List, Any, Dict

def load_qaf(filename: str) -> List[Dict[str, Any]]:
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
        if "example" in data:
            data = data["example"]
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data


from react_batch_demo import batch


def batch(iterable, n=-1):
    l = len(iterable)
    if n <= 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file", 
        type=str, 
        required=True,
        help="the file includes question / partial solution (optional) / answer (optional)")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    data = load_qaf(args.qaf)
    solver = Solver(config=config)

    # init method
    if config.mode == "mcts":
        method = MCTS
    elif config.mode == "sbs":
        method = SBSREACT
    else:
        raise NotImplementedError

    saved_jsonl_file = f"{args.qaf}.{config.mode}.{llm_version}.{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl" 
    with open(saved_jsonl_file, "w") as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [method(config=config, question=d["question"], ground_truth=d["answer"] if config.is_sampling else None) 
                      for d in cur_data]
            jsonlines = solver.solve(agents)
            for d in cur_data:
                question = d["question"]
                d["react"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
