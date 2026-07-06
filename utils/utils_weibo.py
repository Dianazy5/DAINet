import torch
from utils.utils import build_metric_summary

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

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def metrics(y_true, y_pred, category, category_dict, threshold=0.5):
    return build_metric_summary(y_true, y_pred, category, category_dict, threshold=threshold)

def metricsTrueFalse(y_true, y_pred, category, category_dict, threshold=0.5):
    return metrics(y_true, y_pred, category, category_dict, threshold=threshold)

class Recorder():
    def __init__(self, early_step, metric_key='F1'):
        self.max = {metric_key: 0}
        self.cur = {metric_key: 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step
        self.metric_key = metric_key

    def add(self, x):
        self.cur = x
        self.curindex += 1
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
        return self.max
