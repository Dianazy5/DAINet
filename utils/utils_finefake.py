import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def clipdata2gpu(batch_input, use_cuda=True, device=None):
    """Move FineFake/GossipCop-style batches to the target device."""
    if batch_input is None:
        logger.warning("clipdata2gpu received None batch_input.")
        return None

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
        logger.error("Unexpected batch format for FineFake: %s", type(batch_input))
        return None

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
    def __init__(self, early_stop_patience=10, metric_key="metric"):
        self.max = {metric_key: 0.0}
        self.cur = {metric_key: 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop_patience = early_stop_patience
        self.metric_key = metric_key

    def add(self, res):
        self.cur = res
        self.curindex += 1
        print("current", self.cur)
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
        print(f"Max epoch {self.maxindex}", self.max)


def _safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) > 1:
            return roc_auc_score(y_true, y_score)
    except ValueError:
        pass
    return 0.0


def _confusion_counts(y_true, y_pred_int, positive_label=0):
    y_true = np.asarray(y_true).astype(int)
    y_pred_int = np.asarray(y_pred_int).astype(int)
    negative_label = 1 - positive_label
    return {
        "TP": int(np.sum((y_true == positive_label) & (y_pred_int == positive_label))),
        "FP": int(np.sum((y_true == negative_label) & (y_pred_int == positive_label))),
        "TN": int(np.sum((y_true == negative_label) & (y_pred_int == negative_label))),
        "FN": int(np.sum((y_true == positive_label) & (y_pred_int == negative_label))),
    }


def _real_fake_confusion_counts(y_true, y_pred_int, real_label=1, fake_label=0):
    fake_counts = _confusion_counts(y_true, y_pred_int, positive_label=fake_label)
    real_counts = _confusion_counts(y_true, y_pred_int, positive_label=real_label)
    counts = dict(fake_counts)
    counts.update({f"Fake_{key}": value for key, value in fake_counts.items()})
    counts.update({f"Real_{key}": value for key, value in real_counts.items()})
    return counts


def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None, threshold=0.5):
    """
    FineFake metrics with Weibo-style output keys.

    Label convention for FineFake in this project:
    - 0 = Fake
    - 1 = Real
    """
    y_true = np.asarray(label_list).astype(int)
    y_pred = np.asarray(pred_probs).astype(float)
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("calculate_metrics received empty labels or predictions.")
        return {}
    if len(y_true) != len(y_pred):
        logger.error("label/pred length mismatch: %d vs %d", len(y_true), len(y_pred))
        return {}

    y_pred_int = (y_pred >= threshold).astype(int)
    macro_precision = precision_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)

    results = {
        "auc": _safe_auc(y_true, y_pred),
        "metric": macro_f1,
        "recall": macro_recall,
        "precision": macro_precision,
        "acc": accuracy_score(y_true, y_pred_int),
    }

    results.update(_real_fake_confusion_counts(y_true, y_pred_int, real_label=1, fake_label=0))
    results["f1"] = macro_f1
    results["F1"] = macro_f1



    real_mask = y_true == 1
    fake_mask = y_true == 0

    results["Real_Acc"] = accuracy_score(y_true[real_mask], y_pred_int[real_mask]) if np.any(real_mask) else 0.0
    results["Real_Pre"] = precision_score(y_true, y_pred_int, pos_label=1, zero_division=0)
    results["Real_Rec"] = recall_score(y_true, y_pred_int, pos_label=1, zero_division=0)
    results["Real_F1"] = f1_score(y_true, y_pred_int, pos_label=1, zero_division=0)

    results["Fake_Acc"] = accuracy_score(y_true[fake_mask], y_pred_int[fake_mask]) if np.any(fake_mask) else 0.0
    results["Fake_Pre"] = precision_score(y_true, y_pred_int, pos_label=0, zero_division=0)
    results["Fake_Rec"] = recall_score(y_true, y_pred_int, pos_label=0, zero_division=0)
    results["Fake_F1"] = f1_score(y_true, y_pred_int, pos_label=0, zero_division=0)

    real_counts = _confusion_counts(y_true, y_pred_int, positive_label=1)
    fake_counts = _confusion_counts(y_true, y_pred_int, positive_label=0)
    results["Real"] = {
        "precision": results["Real_Pre"],
        "recall": results["Real_Rec"],
        "F1": results["Real_F1"],
        "support": int(np.sum(real_mask)),
        **real_counts,
    }
    results["Fake"] = {
        "precision": results["Fake_Pre"],
        "recall": results["Fake_Rec"],
        "F1": results["Fake_F1"],
        "support": int(np.sum(fake_mask)),
        **fake_counts,
    }

    results["Macro_Acc"] = results["acc"]
    results["Macro_Pre"] = macro_precision
    results["Macro_Rec"] = macro_recall
    results["Macro_F1"] = macro_f1
    results["metric"] = results["Macro_F1"]
    results["f1"] = results["Macro_F1"]
    results["F1"] = results["Macro_F1"]

    if category_list is not None and category_dict is not None and len(category_list) == len(y_true):
        categories = np.asarray(category_list)
        for category_name, category_id in category_dict.items():
            mask = categories == category_id
            if not np.any(mask):
                continue
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            cat_pred_int = y_pred_int[mask]
            results[category_name] = {
                "precision": precision_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "recall": recall_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "fscore": f1_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "F1": f1_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "auc": _safe_auc(cat_true, cat_pred),
                "acc": accuracy_score(cat_true, cat_pred_int),
                "support": int(mask.sum()),
                **_real_fake_confusion_counts(cat_true, cat_pred_int, real_label=1, fake_label=0),
            }

    return results
import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)


def clipdata2gpu(batch_input, use_cuda=True, device=None):
    """Move FineFake/GossipCop-style batches to the target device."""
    if batch_input is None:
        logger.warning("clipdata2gpu received None batch_input.")
        return None

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
        logger.error("Unexpected batch format for FineFake: %s", type(batch_input))
        return None

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
    def __init__(self, early_stop_patience=10, metric_key="metric"):
        self.max = {metric_key: 0.0}
        self.cur = {metric_key: 0.0}
        self.maxindex = 0
        self.curindex = 0
        self.early_stop_patience = early_stop_patience
        self.metric_key = metric_key

    def add(self, res):
        self.cur = res
        self.curindex += 1
        print("current", self.cur)
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
        print(f"Max epoch {self.maxindex}", self.max)


def _safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) > 1:
            return roc_auc_score(y_true, y_score)
    except ValueError:
        pass
    return 0.0


def _confusion_counts(y_true, y_pred_int, positive_label=0):
    y_true = np.asarray(y_true).astype(int)
    y_pred_int = np.asarray(y_pred_int).astype(int)
    negative_label = 1 - positive_label
    return {
        "TP": int(np.sum((y_true == positive_label) & (y_pred_int == positive_label))),
        "FP": int(np.sum((y_true == negative_label) & (y_pred_int == positive_label))),
        "TN": int(np.sum((y_true == negative_label) & (y_pred_int == negative_label))),
        "FN": int(np.sum((y_true == positive_label) & (y_pred_int == negative_label))),
    }


def _real_fake_confusion_counts(y_true, y_pred_int, real_label=1, fake_label=0):
    fake_counts = _confusion_counts(y_true, y_pred_int, positive_label=fake_label)
    real_counts = _confusion_counts(y_true, y_pred_int, positive_label=real_label)
    counts = dict(fake_counts)
    counts.update({f"Fake_{key}": value for key, value in fake_counts.items()})
    counts.update({f"Real_{key}": value for key, value in real_counts.items()})
    return counts


def calculate_metrics(label_list, pred_probs, category_list=None, category_dict=None, threshold=0.5):
    """
    FineFake metrics with Weibo-style output keys.

    Label convention for FineFake in this project:
    - 0 = Fake
    - 1 = Real
    """
    y_true = np.asarray(label_list).astype(int)
    y_pred = np.asarray(pred_probs).astype(float)
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning("calculate_metrics received empty labels or predictions.")
        return {}
    if len(y_true) != len(y_pred):
        logger.error("label/pred length mismatch: %d vs %d", len(y_true), len(y_pred))
        return {}

    y_pred_int = (y_pred >= threshold).astype(int)
    macro_precision = precision_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_int, labels=[0, 1], average="macro", zero_division=0)

    results = {
        "auc": _safe_auc(y_true, y_pred),
        "metric": macro_f1,
        "recall": macro_recall,
        "precision": macro_precision,
        "acc": accuracy_score(y_true, y_pred_int),
    }

    results.update(_real_fake_confusion_counts(y_true, y_pred_int, real_label=1, fake_label=0))
    results["f1"] = macro_f1
    results["F1"] = macro_f1



    real_mask = y_true == 1
    fake_mask = y_true == 0

    results["Real_Acc"] = accuracy_score(y_true[real_mask], y_pred_int[real_mask]) if np.any(real_mask) else 0.0
    results["Real_Pre"] = precision_score(y_true, y_pred_int, pos_label=1, zero_division=0)
    results["Real_Rec"] = recall_score(y_true, y_pred_int, pos_label=1, zero_division=0)
    results["Real_F1"] = f1_score(y_true, y_pred_int, pos_label=1, zero_division=0)

    results["Fake_Acc"] = accuracy_score(y_true[fake_mask], y_pred_int[fake_mask]) if np.any(fake_mask) else 0.0
    results["Fake_Pre"] = precision_score(y_true, y_pred_int, pos_label=0, zero_division=0)
    results["Fake_Rec"] = recall_score(y_true, y_pred_int, pos_label=0, zero_division=0)
    results["Fake_F1"] = f1_score(y_true, y_pred_int, pos_label=0, zero_division=0)

    real_counts = _confusion_counts(y_true, y_pred_int, positive_label=1)
    fake_counts = _confusion_counts(y_true, y_pred_int, positive_label=0)
    results["Real"] = {
        "precision": results["Real_Pre"],
        "recall": results["Real_Rec"],
        "F1": results["Real_F1"],
        "support": int(np.sum(real_mask)),
        **real_counts,
    }
    results["Fake"] = {
        "precision": results["Fake_Pre"],
        "recall": results["Fake_Rec"],
        "F1": results["Fake_F1"],
        "support": int(np.sum(fake_mask)),
        **fake_counts,
    }

    results["Macro_Acc"] = results["acc"]
    results["Macro_Pre"] = macro_precision
    results["Macro_Rec"] = macro_recall
    results["Macro_F1"] = macro_f1
    results["metric"] = results["Macro_F1"]
    results["f1"] = results["Macro_F1"]
    results["F1"] = results["Macro_F1"]

    if category_list is not None and category_dict is not None and len(category_list) == len(y_true):
        categories = np.asarray(category_list)
        for category_name, category_id in category_dict.items():
            mask = categories == category_id
            if not np.any(mask):
                continue
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            cat_pred_int = y_pred_int[mask]
            results[category_name] = {
                "precision": precision_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "recall": recall_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "fscore": f1_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "F1": f1_score(cat_true, cat_pred_int, average="macro", zero_division=0),
                "auc": _safe_auc(cat_true, cat_pred),
                "acc": accuracy_score(cat_true, cat_pred_int),
                "support": int(mask.sum()),
                **_real_fake_confusion_counts(cat_true, cat_pred_int, real_label=1, fake_label=0),
            }

    return results
