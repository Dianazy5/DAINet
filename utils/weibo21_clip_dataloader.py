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

# -*-codeing = utf-8 -*-

import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image

def read_image():
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print("wrong"+filename)
    print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

class bert_data():
    def __init__(self,max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self,path,imagepath,clipimagepath,shuffle,text_only = False):
        self.data = pd.read_excel(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = word2input(content,self.vocab_file,self.max_len)
        ordered_image = pickle.load(open(imagepath,'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))
        clip_text = clip.tokenize(content)
        #("token_ids",token_ids.size())
        #print("masks", masks.size())
        #print("label", label.size())
        #print("category", category.size())
        #print("ordered_image", ordered_image.size())
        #print("clip_image", clip_image.size())
        #print("clip_text", clip_text.size())
        datasets =TensorDataset(token_ids,
                                masks,
                                label,
                                category,
                                ordered_image,
                                clip_image,
                                clip_text
        )
        dataloader = DataLoader(
            dataset = datasets,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = True,
            shuffle = shuffle,
            worker_init_fn = _init_fn
        )
        return dataloader
