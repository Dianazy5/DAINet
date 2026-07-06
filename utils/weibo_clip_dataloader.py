# # utils/weibo21_clip_dataloader.py

# import pickle
# import os
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader


# from cn_clip.clip import load_from_name as cn_clip_load_from_name
# import logging



# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


# def _init_fn(worker_id):


# def word2input_updated(texts, tokenizer: BertTokenizer, max_len: int):
#     """

#     """
#     token_ids_list = []
#     attention_masks_list = []
#     for text in texts:
#         text_str = str(text) if text is not None else ""
#         encoded_dict = tokenizer.encode_plus(
#             text_str,
#             max_length=max_len,
#             add_special_tokens=True,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,

#         )
#         token_ids_list.append(encoded_dict['input_ids'].squeeze(0))
#         attention_masks_list.append(encoded_dict['attention_mask'].squeeze(0))

#     if not token_ids_list:
#         return torch.empty((0, max_len), dtype=torch.long), torch.empty((0, max_len), dtype=torch.long)

#     token_ids_tensor = torch.stack(token_ids_list)
#     masks_tensor = torch.stack(attention_masks_list)
#     return token_ids_tensor, masks_tensor



#     def __init__(self,
#                  max_len: int,
#                  batch_size: int,

#                  category_dict: dict,
#                  num_workers: int = 2,

#                  cn_clip_model_name: str = "ViT-B-16",


#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         self.category_dict = category_dict
#         self.cn_clip_model = None



#         logger.info(f"  max_len: {self.max_len}")
#         logger.info(f"  batch_size: {self.batch_size}")
#         logger.info(f"  vocab_file (for BERT): {self.vocab_file_path}")
#         logger.info(f"  category_dict: {self.category_dict}")
#         logger.info(f"  num_workers: {self.num_workers}")
#         logger.info(f"  cn_clip_model_name: {cn_clip_model_name}")

#         try:




#             bert_tokenizer_path = os.path.dirname(self.vocab_file_path) if os.path.isfile(self.vocab_file_path) else self.vocab_file_path




#             self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)

#         except Exception as e:

#             raise


#         try:
#             device = "cuda" if torch.cuda.is_available() else "cpu"



#             self.cn_clip_model, _ = cn_clip_load_from_name(
#                 cn_clip_model_name,
#                 device=device,
#                 download_root=cn_clip_download_root
#             )


#         except Exception as e:

#             # self.cn_clip_preprocess = None









#     def load_data(self,



#                   shuffle: bool):
#         """


#         """

#         logger.info(f"  MAE PKL: '{mae_image_pkl_path}', CLIP PKL: '{clip_image_pkl_path}'")

#         try:
#             data_df = pd.read_excel(excel_path)


#             text_col = 'content'
#             label_col = 'label'
#             category_col = 'category'

#             if text_col not in data_df.columns:

#                 return None
#             if label_col not in data_df.columns:

#                 return None






#         except FileNotFoundError:

#             return None
#         except Exception as e:

#             return None


#         content_texts = data_df[text_col].tolist()
#         bert_token_ids, bert_masks = word2input_updated(content_texts, self.bert_tokenizer, self.max_len)



#         try:
#             labels_tensor = torch.tensor(data_df[label_col].astype(int).to_numpy(), dtype=torch.long)

#         except Exception as e:

#             return None


#         if category_col in data_df.columns:
#             try:
#                 categories_tensor = torch.tensor(

#                     dtype=torch.long
#                 )

#             except Exception as e:

#                 categories_tensor = torch.zeros(len(data_df), dtype=torch.long)
#         else:

#             categories_tensor = torch.zeros(len(data_df), dtype=torch.long)



#         try:
#             with open(mae_image_pkl_path, 'rb') as f:
#                 mae_image_tensor = pickle.load(f)
#             if not isinstance(mae_image_tensor, torch.Tensor):

#                 return None

#         except FileNotFoundError:

#             return None
#         except Exception as e:

#             return None


#         try:
#             with open(clip_image_pkl_path, 'rb') as f:
#                 clip_image_tensor = pickle.load(f)
#             if not isinstance(clip_image_tensor, torch.Tensor):

#                 return None

#         except FileNotFoundError:

#             return None
#         except Exception as e:

#             return None






#         try:






#                 clip_text_tensor = torch.tensor(clip_text_tensor, dtype=torch.long)

#         except Exception as e:




#             return None



#         num_samples = len(data_df)
#         if not (bert_token_ids.shape[0] == num_samples and
#                 bert_masks.shape[0] == num_samples and
#                 labels_tensor.shape[0] == num_samples and
#                 categories_tensor.shape[0] == num_samples and
#                 mae_image_tensor.shape[0] == num_samples and
#                 clip_image_tensor.shape[0] == num_samples and
#                 clip_text_tensor.shape[0] == num_samples):


#             logger.error(f"  BERT IDs: {bert_token_ids.shape[0]}, BERT Masks: {bert_masks.shape[0]}")
#             logger.error(f"  Labels: {labels_tensor.shape[0]}, Categories: {categories_tensor.shape[0]}")
#             logger.error(f"  MAE Images: {mae_image_tensor.shape[0]}, CLIP Images: {clip_image_tensor.shape[0]}")
#             logger.error(f"  CLIP Text: {clip_text_tensor.shape[0]}")
#             return None



#         # 'content': batch[0] -> bert_token_ids
#         # 'content_masks': batch[1] -> bert_masks
#         # 'label': batch[2] -> labels_tensor
#         # 'category': batch[3] -> categories_tensor















#         datasets = TensorDataset(
#             bert_token_ids,
#             bert_masks,
#             labels_tensor,
#             categories_tensor,
#             mae_image_tensor,
#             clip_image_tensor,
#             clip_text_tensor,

#         )



#         dataloader = DataLoader(
#             dataset=datasets,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             shuffle=shuffle,
#             worker_init_fn=_init_fn
#         )

#         return dataloader

# utils/weibo_clip_dataloader.py

# import pickle
# import os
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset, DataLoader


# from cn_clip.clip import load_from_name as cn_clip_load_from_name
# import logging



# logger = logging.getLogger(__name__)
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)


# def _init_fn(worker_id):


# def word2input_updated(texts, tokenizer: BertTokenizer, max_len: int):
#     """

#     """
#     token_ids_list = []
#     attention_masks_list = []
#     for text in texts:
#         text_str = str(text) if text is not None else ""
#         encoded_dict = tokenizer.encode_plus(
#             text_str,
#             max_length=max_len,
#             add_special_tokens=True,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,

#         )
#         token_ids_list.append(encoded_dict['input_ids'].squeeze(0))
#         attention_masks_list.append(encoded_dict['attention_mask'].squeeze(0))

#     if not token_ids_list:
#         return torch.empty((0, max_len), dtype=torch.long), torch.empty((0, max_len), dtype=torch.long)

#     token_ids_tensor = torch.stack(token_ids_list)
#     masks_tensor = torch.stack(attention_masks_list)
#     return token_ids_tensor, masks_tensor



#     def __init__(self,
#                  max_len: int,
#                  batch_size: int,

#                  category_dict: dict,
#                  num_workers: int = 2,

#                  cn_clip_model_name: str = "ViT-B-16",


#         self.max_len = max_len
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         self.category_dict = category_dict
#         self.cn_clip_model = None



#         logger.info(f"  max_len: {self.max_len}")
#         logger.info(f"  batch_size: {self.batch_size}")
#         logger.info(f"  vocab_file (for BERT): {self.vocab_file_path}")
#         logger.info(f"  category_dict: {self.category_dict}")
#         logger.info(f"  num_workers: {self.num_workers}")
#         logger.info(f"  cn_clip_model_name: {cn_clip_model_name}")
#         logger.info(f"  cn_clip_download_root: {cn_clip_download_root}")


#         try:

#             bert_tokenizer_path = os.path.dirname(self.vocab_file_path) if os.path.isfile(self.vocab_file_path) else self.vocab_file_path




#             self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)

#         except Exception as e:

#             raise


#         try:
#             device = "cuda" if torch.cuda.is_available() else "cpu"

#             self.cn_clip_model, _ = cn_clip_load_from_name(
#                 cn_clip_model_name,
#                 device=device,
#                 download_root=cn_clip_download_root
#             )

#         except Exception as e:




#     def load_data(self,

#                   mae_image_pkl_path: str,
#                   clip_image_pkl_path: str,
#                   shuffle: bool):
#         """


#         """

#         logger.info(f"  MAE PKL: '{mae_image_pkl_path}', CLIP PKL: '{clip_image_pkl_path}'")

#         try:

#             if data_file_path.lower().endswith('.csv'):
#                 data_df = pd.read_csv(data_file_path)

#             elif data_file_path.lower().endswith(('.xls', '.xlsx')):
#                 data_df = pd.read_excel(data_file_path)

#             else:



#                 try:
#                     data_df = pd.read_csv(data_file_path)

#                 except Exception as e_csv:

#                     try:


#                     except Exception as e_excel:







#             text_col = 'content'
#             label_col = 'label'
#             category_col = 'category'

#             if text_col not in data_df.columns:

#                 return None
#             if label_col not in data_df.columns:

#                 return None
#             if category_col not in data_df.columns:





#         except FileNotFoundError:

#             return None


#             return None


#         content_texts = data_df[text_col].tolist()
#         bert_token_ids, bert_masks = word2input_updated(content_texts, self.bert_tokenizer, self.max_len)



#         try:
#             labels_tensor = torch.tensor(data_df[label_col].astype(int).to_numpy(), dtype=torch.long)

#         except Exception as e:

#             return None


#         if category_col in data_df.columns:
#             try:
#                 categories_tensor = torch.tensor(

#                     dtype=torch.long
#                 )

#             except Exception as e:

#                 categories_tensor = torch.zeros(len(data_df), dtype=torch.long)
#         else:

#             categories_tensor = torch.zeros(len(data_df), dtype=torch.long)



#         try:
#             with open(mae_image_pkl_path, 'rb') as f:
#                 mae_image_tensor = pickle.load(f)
#             if not isinstance(mae_image_tensor, torch.Tensor):

#                 return None

#         except FileNotFoundError:

#             return None
#         except Exception as e:

#             return None


#         try:
#             with open(clip_image_pkl_path, 'rb') as f:
#                 clip_image_tensor = pickle.load(f)
#             if not isinstance(clip_image_tensor, torch.Tensor):

#                 return None

#         except FileNotFoundError:

#             return None
#         except Exception as e:

#             return None


#         try:



#                 clip_text_tensor = torch.tensor(clip_text_tensor, dtype=torch.long)

#         except Exception as e:


#             return None



#         num_samples = len(data_df)
#         if not (bert_token_ids.shape[0] == num_samples and
#                 bert_masks.shape[0] == num_samples and
#                 labels_tensor.shape[0] == num_samples and
#                 categories_tensor.shape[0] == num_samples and
#                 mae_image_tensor.shape[0] == num_samples and
#                 clip_image_tensor.shape[0] == num_samples and
#                 clip_text_tensor.shape[0] == num_samples):


#             logger.error(f"  BERT IDs: {bert_token_ids.shape[0]}, BERT Masks: {bert_masks.shape[0]}")
#             logger.error(f"  Labels: {labels_tensor.shape[0]}, Categories: {categories_tensor.shape[0]}")
#             logger.error(f"  MAE Images: {mae_image_tensor.shape[0]}, CLIP Images: {clip_image_tensor.shape[0]}")
#             logger.error(f"  CLIP Text: {clip_text_tensor.shape[0]}")
#             return None




#         datasets = TensorDataset(
#             bert_token_ids,
#             bert_masks,
#             labels_tensor,
#             categories_tensor,
#             mae_image_tensor,
#             clip_image_tensor,
#             clip_text_tensor,

#         )



#         dataloader = DataLoader(
#             dataset=datasets,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=True,
#             shuffle=shuffle,
#             worker_init_fn=_init_fn
#         )

#         return dataloader


# utils/weibo21_clip_dataloader.py

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
        text_str = str(text) if text is not None else ""
        encoded_dict = tokenizer.encode_plus(
            text_str,
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


        logger.info(f"Weibo21 DataLoader (bert_data) 初始化:")
        logger.info(f"  max_len: {self.max_len}")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  vocab_file (for BERT): {self.vocab_file_path}")
        logger.info(f"  category_dict: {self.category_dict}")
        logger.info(f"  num_workers: {self.num_workers}")
        logger.info(f"  cn_clip_model_name: {cn_clip_model_name}")

        try:




            bert_tokenizer_path = os.path.dirname(self.vocab_file_path) if os.path.isfile(self.vocab_file_path) else self.vocab_file_path
            if not os.path.isdir(bert_tokenizer_path):
                 logger.warning(f"BERT tokenizer 路径 '{bert_tokenizer_path}' 不是一个有效目录。尝试直接使用 '{self.vocab_file_path}' 作为标识符。")
                 bert_tokenizer_path = self.vocab_file_path

            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
            logger.info(f"BERT Tokenizer 成功加载: {bert_tokenizer_path}")
        except Exception as e:
            logger.error(f"从 {bert_tokenizer_path} (源自 {self.vocab_file_path}) 加载 BERT Tokenizer 失败: {e}")
            raise


        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"尝试使用设备 '{device}' 加载 cn_clip 模型 '{cn_clip_model_name}'...")


            self.cn_clip_model, _ = cn_clip_load_from_name(
                cn_clip_model_name,
                device=device,
                download_root=cn_clip_download_root
            )

            logger.info(f"cn_clip 模型 '{cn_clip_model_name}' 加载成功。")
        except Exception as e:
            self.cn_clip_model = None
            # self.cn_clip_preprocess = None
            logger.warning(f"加载 cn_clip 模型 '{cn_clip_model_name}' 失败: {e}. "
                           "CLIP 文本分词将不可用，除非 cn_clip.tokenize 能在无模型实例情况下工作（不太可能）。")







    def load_data(self,
                  excel_path: str,
                  mae_image_pkl_path: str,
                  clip_image_pkl_path: str,
                  shuffle: bool):




        logger.info(f"开始加载数据: excel_path='{excel_path}'")
        logger.info(f"  MAE PKL: '{mae_image_pkl_path}', CLIP PKL: '{clip_image_pkl_path}'")

        try:
            data_df = pd.read_csv(excel_path)


            text_col = 'content'
            label_col = 'label'
            category_col = 'category'

            if text_col not in data_df.columns:
                logger.error(f"Excel 文件 {excel_path} 中缺少文本列 '{text_col}'。")
                return None
            if label_col not in data_df.columns:
                logger.error(f"Excel 文件 {excel_path} 中缺少标签列 '{label_col}'。")
                return None
            if category_col not in data_df.columns:
                logger.warning(f"Excel 文件 {excel_path} 中缺少类别列 '{category_col}'。将使用默认分类。")


            data_df[text_col] = data_df[text_col].fillna('')
            logger.info(f"从 {excel_path} 加载了 {len(data_df)} 行。")
        except FileNotFoundError:
            logger.error(f"Excel 文件未找到: {excel_path}")
            return None
        except Exception as e:
            logger.error(f"读取 Excel 文件 {excel_path} 失败: {e}")
            return None


        content_texts = data_df[text_col].tolist()
        bert_token_ids, bert_masks = word2input_updated(content_texts, self.bert_tokenizer, self.max_len)
        logger.info(f"BERT 文本处理完成。Shape: {bert_token_ids.shape}")


        try:
            labels_tensor = torch.tensor(data_df[label_col].astype(int).to_numpy(), dtype=torch.long)
            logger.info(f"标签处理完成。Shape: {labels_tensor.shape}")
        except Exception as e:
            logger.error(f"处理标签失败: {e}")
            return None


        if category_col in data_df.columns:
            try:
                categories_tensor = torch.tensor(
                    data_df[category_col].astype(str).apply(lambda c: self.category_dict.get(c, 0)).to_numpy(),
                    dtype=torch.long
                )
                logger.info(f"类别处理完成。Shape: {categories_tensor.shape}")
            except Exception as e:
                logger.error(f"处理类别失败: {e}. 将使用默认类别0。")
                categories_tensor = torch.zeros(len(data_df), dtype=torch.long)
        else:
            logger.info("未找到类别列，使用默认类别0。")
            categories_tensor = torch.zeros(len(data_df), dtype=torch.long)



        try:
            with open(mae_image_pkl_path, 'rb') as f:
                mae_image_tensor = pickle.load(f)
            if not isinstance(mae_image_tensor, torch.Tensor):
                logger.error(f"MAE PKL 文件 ({mae_image_pkl_path}) 未包含有效的 PyTorch 张量。类型: {type(mae_image_tensor)}")
                return None
            logger.info(f"MAE 图像特征 PKL 加载成功。Shape: {mae_image_tensor.shape}")
        except FileNotFoundError:
            logger.error(f"MAE 图像 PKL 文件未找到: {mae_image_pkl_path}")
            return None
        except Exception as e:
            logger.error(f"加载 MAE 图像 PKL ({mae_image_pkl_path}) 失败: {e}")
            return None


        try:
            with open(clip_image_pkl_path, 'rb') as f:
                clip_image_tensor = pickle.load(f)
            if not isinstance(clip_image_tensor, torch.Tensor):
                logger.error(f"CLIP PKL 文件 ({clip_image_pkl_path}) 未包含有效的 PyTorch 张量。类型: {type(clip_image_tensor)}")
                return None
            logger.info(f"CLIP 图像特征 PKL 加载成功。Shape: {clip_image_tensor.shape}")
        except FileNotFoundError:
            logger.error(f"CLIP 图像 PKL 文件未找到: {clip_image_pkl_path}")
            return None
        except Exception as e:
            logger.error(f"加载 CLIP 图像 PKL ({clip_image_pkl_path}) 失败: {e}")
            return None






        try:



            logger.info(f"开始使用 cn_clip.tokenize 处理 {len(content_texts)} 条 CLIP 文本...")
            clip_text_tensor = cn_clip_module.tokenize(content_texts, context_length=77)
            if not isinstance(clip_text_tensor, torch.Tensor):
                clip_text_tensor = torch.tensor(clip_text_tensor, dtype=torch.long)
            logger.info(f"CLIP 文本处理完成。Shape: {clip_text_tensor.shape}")
        except Exception as e:
            logger.error(f"使用 cn_clip.tokenize 处理文本失败: {e}")
            logger.error("确保 cn_clip 库已正确安装并且其依赖项（如词汇表）可用。")


            return None



        num_samples = len(data_df)
        if not (bert_token_ids.shape[0] == num_samples and
                bert_masks.shape[0] == num_samples and
                labels_tensor.shape[0] == num_samples and
                categories_tensor.shape[0] == num_samples and
                mae_image_tensor.shape[0] == num_samples and
                clip_image_tensor.shape[0] == num_samples and
                clip_text_tensor.shape[0] == num_samples):
            logger.error("一个或多个处理后的数据张量样本数量与源数据不匹配！")
            logger.error(f"  Excel行数: {num_samples}")
            logger.error(f"  BERT IDs: {bert_token_ids.shape[0]}, BERT Masks: {bert_masks.shape[0]}")
            logger.error(f"  Labels: {labels_tensor.shape[0]}, Categories: {categories_tensor.shape[0]}")
            logger.error(f"  MAE Images: {mae_image_tensor.shape[0]}, CLIP Images: {clip_image_tensor.shape[0]}")
            logger.error(f"  CLIP Text: {clip_text_tensor.shape[0]}")
            return None



        # 'content': batch[0] -> bert_token_ids
        # 'content_masks': batch[1] -> bert_masks
        # 'label': batch[2] -> labels_tensor
        # 'category': batch[3] -> categories_tensor













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
        logger.info(f"TensorDataset 创建成功，包含 {len(datasets)} 个样本。")


        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        logger.info(f"DataLoader 创建成功，批大小 {self.batch_size}，共 {len(dataloader)} 个批次。")
        return dataloader