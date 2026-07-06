import logging

import torch
from utils.utils import build_metric_summary

logger = logging.getLogger(__name__)

def clipdata2gpu(batch_input, use_cuda=True, device=None):
    if batch_input is None:
        raise ValueError("clipdata2gpu received None batch_input.")

    if isinstance(batch_input, dict):
        batch_dict = batch_input
    elif isinstance(batch_input, (list, tuple)) and len(batch_input) == 8:
        keys = [
            "content",
            "content_masks",
            "label",
            "category",
            "image",
            "clip_image",
            "clip_text",
            "clip_attention_mask",
        ]
        batch_dict = dict(zip(keys, batch_input))
    else:
        raise TypeError(f"Unexpected batch format for FineFake: {type(batch_input)}")

    if device is None:
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    out = {}
    for key, value in batch_dict.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out

class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class Recorder:
    def __init__(self, early_stop_patience=10, metric_key="F1"):
        self.max = {metric_key: 0.0}
        self.cur = {metric_key: 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop_patience = early_stop_patience
        self.metric_key = metric_key

    def add(self, res):
        self.cur = res
        self.curindex += 1
        return self.judge()

    def judge(self):
        if self.cur[self.metric_key] > self.max.get(self.metric_key, 0.0):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return "save"
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_stop_patience:
            return "esc"
        return "continue"

    def showfinal(self):
        return self.max

def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None, threshold=0.5):
    results = build_metric_summary(label_list, pred_probs, category_list, category_dict, threshold=threshold)
    return results
