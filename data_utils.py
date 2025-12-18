from typing import List, Any, Dict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
from dataclasses import dataclass


@dataclass
class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, Rq: int, Rc: int, processor: AutoProcessor = None, problem_col_name: str = "prompt"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.Rq = Rq
        self.Rc = Rc
        self.problem_col_name = problem_col_name
        self.processor = processor
        if self.processor is not None:
            self.processor.tokenizer.padding_side = 'left'


    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_text = [ex[self.problem_col_name] for ex in batch]
        c_text = [ex["generations_rh"] for ex in batch]
        c_neg_text = [ex["generations"] for ex in batch]
        bsz = len(batch)

        img = [ex["image"] for ex in batch]
        q_messages = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text", "text": DataCollator.SYSTEM_PROMPT
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", "image": img[i],
                        },
                        {"type": "text", "text": DataCollator.USER_PROMPT_TEMPLATE.format(problem=ex)},
                    ],
                }
            ] for i, ex in enumerate(q_text)]
        c_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ex},
                    ],
                }
            ] for i, ex in enumerate(c_text)]
        c_neg_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ex},
                    ],
                }
            ] for i, ex in enumerate(c_neg_text)]
        messages = q_messages + c_messages + c_neg_messages
        all_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True  # padding should be set for batch generation!
        )
        return {
            "q_input_ids": all_inputs["input_ids"][:bsz],
            "c_input_ids": all_inputs["input_ids"][bsz:2 * bsz],
            "c_neg_input_ids": all_inputs["input_ids"][2 * bsz:],
            "image_grid_thw": all_inputs["image_grid_thw"],
            "pixel_values": all_inputs["pixel_values"],
            "attention_mask": all_inputs["attention_mask"]
        }


def make_dataloader(tokenizer, batch_size, split, args, groups_q=None, groups_c=None, processor=None, problem_col_name="prompt", shuffle=True):
    """
    Replace this with your own dataset mapping function if needed.
    """
    ds_tr = load_dataset(args.dataset, "default", split=split)
    ds = ds_tr.with_format("torch")

    if groups_q is None:
        groups_q = args.groups_q
    if groups_c is None:
        groups_c = args.groups_c
    Rq = max(groups_q)
    Rc = max(groups_c)
    collator = DataCollator(tokenizer, args.max_length, Rq, Rc, processor=processor, problem_col_name=problem_col_name)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
    return loader
