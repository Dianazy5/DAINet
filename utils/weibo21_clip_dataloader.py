from utils.domain_labels import category_to_id

import pickle
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import cn_clip.clip as cn_clip_module
from cn_clip.clip import load_from_name as cn_clip_load_from_name
import logging
import numpy as np

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _init_fn(worker_id):
    np.random.seed(2024 + worker_id)

def word2input_updated(texts, tokenizer: BertTokenizer, max_len: int):
    token_ids_list = []
    attention_masks_list = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            max_length=max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        token_ids_list.append(encoded_dict['input_ids'].squeeze(0))
        attention_masks_list.append(encoded_dict['attention_mask'].squeeze(0))

    if not token_ids_list:
        return torch.empty((0, max_len), dtype=torch.long), torch.empty((0, max_len), dtype=torch.long)

    token_ids_tensor = torch.stack(token_ids_list)
    masks_tensor = torch.stack(attention_masks_list)
    return token_ids_tensor, masks_tensor

class bert_data:
    def __init__(self,
                 max_len: int,
                 batch_size: int,
                 vocab_file: str,
                 category_dict: dict,
                 num_workers: int = 2,

                 cn_clip_model_name: str = "ViT-B-16",
                 cn_clip_download_root: str = './'):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file_path = vocab_file
        self.category_dict = category_dict
        self.cn_clip_model = None

        logger.info(f"Weibo21 DataLoader (bert_data) initialized:")
        logger.info(f"  max_len: {self.max_len}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  vocab_file (for BERT): {self.vocab_file_path}")
        logger.info(f"  category_dict: {self.category_dict}")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  cn_clip_model_name: {cn_clip_model_name}")

        bert_tokenizer_path = os.path.dirname(self.vocab_file_path) if os.path.isfile(
            self.vocab_file_path) else self.vocab_file_path
        if not os.path.isdir(bert_tokenizer_path):
            raise FileNotFoundError(f"BERT tokenizer path is not a valid directory: {bert_tokenizer_path}")
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        logger.info(f"BERT Tokenizer loaded successfully: {bert_tokenizer_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Trying to load with device '{device}' load cn_clip model '{cn_clip_model_name}'...")
        self.cn_clip_model, _ = cn_clip_load_from_name(
            cn_clip_model_name,
            device=device,
            download_root=cn_clip_download_root
        )
        logger.info(f"cn_clip model '{cn_clip_model_name}' loaded successfully.")

    def load_data(self,
                  excel_path: str,
                  mae_image_pkl_path: str,
                  clip_image_pkl_path: str,
                  shuffle: bool):
        logger.info(f"Start loading data: excel_path='{excel_path}'")
        logger.info(f"  MAE PKL: '{mae_image_pkl_path}', CLIP PKL: '{clip_image_pkl_path}'")

        data_df = pd.read_csv(excel_path)
        text_col = 'content'
        label_col = 'label'
        category_col = 'category'
        if text_col not in data_df.columns:
            raise KeyError(f"Data file {excel_path} is missing text column '{text_col}'.")
        if label_col not in data_df.columns:
            raise KeyError(f"Data file {excel_path} is missing label column '{label_col}'.")
        if category_col not in data_df.columns:
            raise KeyError(f"Data file {excel_path} is missing category column '{category_col}'.")
        data_df[text_col] = data_df[text_col].fillna('')
        logger.info(f"from {excel_path} loaded {len(data_df)} rows.")

        content_texts = data_df[text_col].tolist()
        bert_token_ids, bert_masks = word2input_updated(content_texts, self.bert_tokenizer, self.max_len)
        logger.info(f"BERT text processing finished. Shape: {bert_token_ids.shape}")

        labels_tensor = torch.tensor(data_df[label_col].astype(int).to_numpy(), dtype=torch.long)
        logger.info(f"Label processing finished. Shape: {labels_tensor.shape}")

        categories_tensor = torch.tensor(
            data_df[category_col].astype(str).apply(lambda c: category_to_id(c, self.category_dict)).to_numpy(),
            dtype=torch.long
        )
        logger.info(f"Category processing finished. Shape: {categories_tensor.shape}")

        with open(mae_image_pkl_path, 'rb') as f:
            mae_image_tensor = pickle.load(f)
        if not isinstance(mae_image_tensor, torch.Tensor):
            raise TypeError(
                f"MAE PKL file ({mae_image_pkl_path}) does not contain a PyTorch tensor. Type: {type(mae_image_tensor)}")
        logger.info(f"MAE image feature PKL loaded successfully.Shape: {mae_image_tensor.shape}")

        with open(clip_image_pkl_path, 'rb') as f:
            clip_image_tensor = pickle.load(f)
        if not isinstance(clip_image_tensor, torch.Tensor):
            raise TypeError(
                f"CLIP PKL file ({clip_image_pkl_path}) does not contain a PyTorch tensor. Type: {type(clip_image_tensor)}")
        logger.info(f"CLIP image feature PKL loaded successfully.Shape: {clip_image_tensor.shape}")

        logger.info(f"Start processing with cn_clip.tokenize {len(content_texts)} CLIP texts...")
        clip_text_tensor = cn_clip_module.tokenize(content_texts, context_length=77)
        if not isinstance(clip_text_tensor, torch.Tensor):
            clip_text_tensor = torch.tensor(clip_text_tensor, dtype=torch.long)
        logger.info(f"CLIP text processing finished. Shape: {clip_text_tensor.shape}")

        num_samples = len(data_df)
        if not (bert_token_ids.shape[0] == num_samples and
                bert_masks.shape[0] == num_samples and
                labels_tensor.shape[0] == num_samples and
                categories_tensor.shape[0] == num_samples and
                mae_image_tensor.shape[0] == num_samples and
                clip_image_tensor.shape[0] == num_samples and
                clip_text_tensor.shape[0] == num_samples):
            raise ValueError(
                "One or more processed tensors have a sample-count mismatch with the source data: "
                f"rows={num_samples}, bert_ids={bert_token_ids.shape[0]}, bert_masks={bert_masks.shape[0]}, "
                f"labels={labels_tensor.shape[0]}, categories={categories_tensor.shape[0]}, "
                f"mae_images={mae_image_tensor.shape[0]}, clip_images={clip_image_tensor.shape[0]}, "
                f"clip_text={clip_text_tensor.shape[0]}"
            )

        clip_attention_mask_tensor = (clip_text_tensor != 0).long()

        datasets = TensorDataset(
            bert_token_ids,
            bert_masks,
            labels_tensor,
            categories_tensor,
            mae_image_tensor,
            clip_image_tensor,
            clip_text_tensor,
            clip_attention_mask_tensor
        )
        logger.info(f"TensorDataset created successfully with {len(datasets)} samples.")

        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        logger.info(
            f"DataLoader created successfully with batch size {self.batch_size},and {len(dataloader)} batches.")
        return dataloader


