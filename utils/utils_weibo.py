# # -*-codeing = utf-8 -*-

# # -*-codeing = utf-8 -*-

import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

# --- START: MODIFIED CODE ---


def clipdata2gpu(batch, use_cuda):



    batch_data = {
        'content': batch[0],
        'content_masks': batch[1],
        'label': batch[2],
        'category': batch[3],
        'image': batch[4],
        'clip_image': batch[5],
        'clip_text': batch[6]
    }

    if use_cuda:
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cuda()
    


    if len(batch) > 7 and batch[7] is not None:
        batch_data['teacher_reasoning_text_emb'] = batch[7].cuda() if use_cuda else batch[7]
    if len(batch) > 8 and batch[8] is not None:
        batch_data['teacher_reasoning_image_emb'] = batch[8].cuda() if use_cuda else batch[8]
    if len(batch) > 9 and batch[9] is not None:
        batch_data['teacher_reasoning_cross_emb'] = batch[9].cuda() if use_cuda else batch[9]

    return batch_data

def data2gpu(batch, use_cuda):



    batch_data = {
        'content': batch[0],
        'content_masks': batch[1],
        'label': batch[2],
        'category': batch[3],
        'image': batch[4]
    }
    if use_cuda:
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.cuda()
    return batch_data

# --- END: MODIFIED CODE ---


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def _confusion_counts(y_true, y_pred_int, positive_label=1):
    y_true = np.array(y_true).astype(int)
    y_pred_int = np.array(y_pred_int).astype(int)
    negative_label = 1 - positive_label
    return {
        'TP': int(np.sum((y_true == positive_label) & (y_pred_int == positive_label))),
        'FP': int(np.sum((y_true == negative_label) & (y_pred_int == positive_label))),
        'TN': int(np.sum((y_true == negative_label) & (y_pred_int == negative_label))),
        'FN': int(np.sum((y_true == positive_label) & (y_pred_int == negative_label))),
    }


def _real_fake_confusion_counts(y_true, y_pred_int, real_label=0, fake_label=1):
    fake_counts = _confusion_counts(y_true, y_pred_int, positive_label=fake_label)
    real_counts = _confusion_counts(y_true, y_pred_int, positive_label=real_label)
    counts = dict(fake_counts)
    counts.update({f'Fake_{key}': value for key, value in fake_counts.items()})
    counts.update({f'Real_{key}': value for key, value in real_counts.items()})
    return counts


def metrics(y_true, y_pred, category, category_dict, threshold=0.5):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}


    if category:
        for i, c in enumerate(category):
            c_val = c.item() if hasattr(c, 'item') else c
            category_name = reverse_category_dict[c_val]
            res_by_category[category_name]['y_true'].append(y_true[i])
            res_by_category[category_name]['y_pred'].append(y_pred[i])


    try:
        metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        metrics_by_category['auc'] = 0.0

    y_pred_int = (np.array(y_pred) >= threshold).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
    metrics_by_category['f1'] = metrics_by_category['metric']

    metrics_by_category.update(_real_fake_confusion_counts(y_true, y_pred_int, real_label=0, fake_label=1))


    for c, res in res_by_category.items():
        if not res['y_true']:
            continue
        
        y_pred_cat_int = (np.array(res['y_pred']) >= threshold).astype(int)
        
        cat_auc = 0.0
        try:

            if len(np.unique(res['y_true'])) > 1:
                cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
        except ValueError:
            pass

        metrics_by_category[c] = {
            'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
            'auc': cat_auc,
            'acc': accuracy_score(res['y_true'], y_pred_cat_int),
            **_real_fake_confusion_counts(res['y_true'], y_pred_cat_int, real_label=0, fake_label=1),
        }
    return metrics_by_category


def metricsTrueFalse(y_true, y_pred, category, category_dict, threshold=0.5):



    results = metrics(y_true, y_pred, category, category_dict, threshold=threshold)


    y_true_np = np.array(y_true)
    y_pred_int = (np.array(y_pred) >= threshold).astype(int)



    real_mask = y_true_np == 0
    fake_mask = y_true_np == 1

    results['Real_Acc'] = accuracy_score(y_true_np[real_mask], y_pred_int[real_mask]) if np.any(real_mask) else 0.0
    results['Real_Pre'] = precision_score(y_true_np, y_pred_int, pos_label=0, zero_division=0)
    results['Real_Rec'] = recall_score(y_true_np, y_pred_int, pos_label=0, zero_division=0)
    results['Real_F1'] = f1_score(y_true_np, y_pred_int, pos_label=0, zero_division=0)

    results['Fake_Acc'] = accuracy_score(y_true_np[fake_mask], y_pred_int[fake_mask]) if np.any(fake_mask) else 0.0
    results['Fake_Pre'] = precision_score(y_true_np, y_pred_int, pos_label=1, zero_division=0)
    results['Fake_Rec'] = recall_score(y_true_np, y_pred_int, pos_label=1, zero_division=0)
    results['Fake_F1'] = f1_score(y_true_np, y_pred_int, pos_label=1, zero_division=0)

    real_counts = _confusion_counts(y_true_np, y_pred_int, positive_label=0)
    fake_counts = _confusion_counts(y_true_np, y_pred_int, positive_label=1)
    results['Real'] = {
        'precision': results['Real_Pre'],
        'recall': results['Real_Rec'],
        'F1': results['Real_F1'],
        'support': int(np.sum(real_mask)),
        **real_counts,
    }
    results['Fake'] = {
        'precision': results['Fake_Pre'],
        'recall': results['Fake_Rec'],
        'F1': results['Fake_F1'],
        'support': int(np.sum(fake_mask)),
        **fake_counts,
    }


    results['Macro_Acc'] = results.get('acc', 0)
    results['Macro_Pre'] = precision_score(y_true_np, y_pred_int, labels=[0, 1], average='macro', zero_division=0)
    results['Macro_Rec'] = recall_score(y_true_np, y_pred_int, labels=[0, 1], average='macro', zero_division=0)
    results['Macro_F1'] = f1_score(y_true_np, y_pred_int, labels=[0, 1], average='macro', zero_division=0)
    results['metric'] = results['Macro_F1']
    results['f1'] = results['Macro_F1']
    
    return results


class Recorder():

    def __init__(self, early_step, metric_key='metric'):
        self.max = {metric_key: 0}
        self.cur = {metric_key: 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step
        self.metric_key = metric_key

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):

        if self.cur.get(self.metric_key, 0) > self.max.get(self.metric_key, 0):
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print(f"Max epoch {self.maxindex}", self.max)

# # -*-codeing = utf-8 -*-

# import torch
# from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
# import numpy as np

# ## Data Handling
# def data_to_gpu(data, use_cuda):
#     """
#     Moves a dictionary or list containing tensors to the GPU, if available.
#     """
#     if not use_cuda:
#         return data

#     if isinstance(data, dict):
#         for key, value in data.items():
#             if isinstance(value, torch.Tensor):
#                 data[key] = value.cuda()
#     elif isinstance(data, list):
#         data = [item.cuda() if isinstance(item, torch.Tensor) else item for item in data]
    
#     return data

# ## Metrics Calculation
# def metrics(y_true, y_pred, category, category_dict):
#     """
#     This helper function calculates various metrics and organizes them by category.
#     """
#     res_by_category = {}
#     metrics_by_category = {}
#     reverse_category_dict = {}
#     for k, v in category_dict.items():
#         reverse_category_dict[v] = k
#         res_by_category[k] = {"y_true": [], "y_pred": []}

#     # Populates the res_by_category dictionary with true and predicted values for each category
#     if category:
#         for i, c in enumerate(category):
#             c_val = c.item() if hasattr(c, 'item') else c
#             category_name = reverse_category_dict[c_val]
#             res_by_category[category_name]['y_true'].append(y_true[i])
#             res_by_category[category_name]['y_pred'].append(y_pred[i])

#     # Calculates overall metrics
#     try:
#         metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
#     except ValueError:
#         metrics_by_category['auc'] = 0.0
#         pass
    
#     y_pred_int = np.around(np.array(y_pred)).astype(int)
#     metrics_by_category['metric'] = f1_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['recall'] = recall_score(y_true, y_pred_int, average='macro')
#     metrics_by_category['precision'] = precision_score(y_true, y_pred_int, average='macro', zero_division=0)
#     metrics_by_category['acc'] = accuracy_score(y_true, y_pred_int)
#     metrics_by_category['f1'] = metrics_by_category['metric']

#     # Calculates metrics by category
#     for c, res in res_by_category.items():
#         if not res['y_true']:
#             continue
        
#         y_pred_cat_int = np.around(np.array(res['y_pred'])).astype(int)
        
#         cat_auc = 0.0
#         try:
#             if len(np.unique(res['y_true'])) > 1:
#                 cat_auc = roc_auc_score(res['y_true'], res['y_pred'])
#         except ValueError:
#             pass

#         metrics_by_category[c] = {
#             'precision': precision_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'recall': recall_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'fscore': f1_score(res['y_true'], y_pred_cat_int, average='macro', zero_division=0),
#             'auc': cat_auc,
#             'acc': accuracy_score(res['y_true'], y_pred_cat_int)
#         }

#     return metrics_by_category


# def metricsTrueFalse(y_true, y_pred, category, category_dict):
#     """
#     This function calls the metrics function to get the output format.
#     """
#     return metrics(y_true, y_pred, category, category_dict)

# ## Training Utilities
# class Averager():
#     """
#     A class to calculate the running average of a value.
#     """
#     def __init__(self):
#         self.n = 0
#         self.v = 0

#     def add(self, x):
#         self.v = (self.v * self.n + x) / (self.n + 1)
#         self.n += 1

#     def item(self):
#         return self.v


# class Recorder():
#     """
#     A class to record and manage model training progress for early stopping.
#     """
#     def __init__(self, early_step):
#         self.max = {'metric': 0}
#         self.cur = {'metric': 0}
#         self.maxindex = 0
#         self.curindex = 0
#         self.early_step = early_step

#     def add(self, x):
#         self.cur = x
#         self.curindex += 1
#         print("curent", self.cur)
#         return self.judge()

#     def judge(self):
#         if self.cur['metric'] > self.max['metric']:
#             self.max = self.cur
#             self.maxindex = self.curindex
#             self.showfinal()
#             return 'save'
#         self.showfinal()
#         if self.curindex - self.maxindex >= self.early_step:
#             return 'esc'
#         else:
#             return 'continue'

#     def showfinal(self):
#         print("Max", self.max)
