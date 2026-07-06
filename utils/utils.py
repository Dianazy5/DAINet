import torch
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils.domain_labels import display_domain_name

logger = logging.getLogger(__name__)

def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score)

def _category_value(category_item):
    return category_item.item() if hasattr(category_item, "item") else category_item

def build_metric_summary(y_true, y_score, category=None, category_dict=None, threshold=0.5):
    y_true_np = np.asarray(y_true).astype(int)
    y_score_np = np.asarray(y_score).astype(float)
    if y_true_np.size == 0 or y_score_np.size == 0:
        raise ValueError("Labels or prediction scores are empty.")
    if len(y_true_np) != len(y_score_np):
        raise ValueError(f"Label/prediction length mismatch: {len(y_true_np)} vs {len(y_score_np)}.")
    y_pred_int = (y_score_np >= threshold).astype(int)
    results = {
        "F1": f1_score(y_true_np, y_pred_int, labels=[0, 1], average="macro", zero_division=0),
        "acc": accuracy_score(y_true_np, y_pred_int),
        "auc": safe_auc(y_true_np, y_score_np),
    }
    domain_macro_f1 = {}
    if category is not None and category_dict is not None and len(category) == len(y_true_np):
        categories = np.asarray([_category_value(item) for item in category], dtype=object)
        for category_name, category_id in category_dict.items():
            mask = categories == category_id
            if np.any(mask):
                domain_macro_f1[display_domain_name(category_name)] = f1_score(
                    y_true_np[mask],
                    y_pred_int[mask],
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
    results["domain_macro_F1"] = domain_macro_f1
    return results

def clipdata2gpu(batch_input):
    if batch_input is None:
        raise ValueError("clipdata2gpu received None batch_input.")

    batch_dict = None

    if isinstance(batch_input, dict):
        batch_dict = batch_input
        logger.debug("clipdata2gpu received a dictionary batch.")
    elif isinstance(batch_input, (list, tuple)):
        logger.debug(f"clipdata2gpu received a tuple/list batch with {len(batch_input)} items.")

        if len(batch_input) == 8:
            keys = [
                'content', 'content_masks', 'label', 'category',
                'image', 'clip_image', 'clip_text', 'clip_attention_mask'
            ]
            batch_dict = dict(zip(keys, batch_input))
            logger.debug("clipdata2gpu converted tuple/list batch to dictionary.")
        else:
            raise ValueError(f"clipdata2gpu received a tuple/list with unexpected number of items: {len(batch_input)}.")
    else:
        raise TypeError(f"clipdata2gpu expects a dictionary, tuple, or list, but received {type(batch_input)}.")

    if batch_dict is None:
        raise RuntimeError("clipdata2gpu: batch_dict is None after type checking.")

    gpu_batch = {}
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            gpu_batch[key] = value.cuda()
        else:
            gpu_batch[key] = value
            logger.debug(f"Key '{key}' in batch is not a Tensor (type: {type(value)}), not moved to GPU.")
    return gpu_batch

class Averager:
     def __init__(self): self.n=0.0; self.v=0.0
     def add(self, x): self.v=(self.v*self.n+x)/(self.n+1); self.n+=1
     def item(self): return self.v

class Recorder:
    def __init__(self, early_stop_patience=10, metric_key='F1'):
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
        if self.cur.get(self.metric_key, 0.0) > self.max.get(self.metric_key, 0.0):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_stop_patience:
            return 'esc'
        return 'continue'

    def showfinal(self):
        return self.max

def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None):
    metrics = build_metric_summary(label_list, pred_probs, category_list, category_dict)
    return metrics