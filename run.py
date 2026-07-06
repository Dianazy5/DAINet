import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor
import logging
from utils.domain_labels import FINEFAKE_CATEGORY_DICT, WEIBO21_CATEGORY_DICT, WEIBO_CATEGORY_DICT
from FineFake_dataset import FineFakeDataset
from utils.weibo_clip_dataloader import bert_data as WeiboDataLoaderClass
from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderClass
from model.domain_finefake import Trainer as DOMAINTrainerFineFake
from model.domain_weibo import Trainer as DOMAINTrainerWeibo

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

bert_tokenizer_finefake = None
clip_processor_finefake = None

def collate_fn_finefake(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        values = [item[key] for item in batch]
        if all(isinstance(v, torch.Tensor) for v in values):
            collated_batch[key] = torch.stack(values, dim=0)
        else:
            collated_batch[key] = values
    return collated_batch

class Run():
    def __init__(self, config):
        self.config = config
        self.use_cuda = config['use_cuda']
        self.dataset = config['dataset']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.early_stop = config['early_stop']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        requested_metric = self.config.get('early_stop_metric', 'F1')
        self.early_stop_metric_key = 'F1' if requested_metric == 'metric' else requested_metric

        if self.dataset == "finefake":
            self.root_path = self.config.get('finefake_data_dir')
            self.bert_model_path = self.config.get('bert_model_path_finefake')
            self.clip_model_path = self.config.get('clip_model_path_finefake')
            self.category_dict = FINEFAKE_CATEGORY_DICT

            global bert_tokenizer_finefake, clip_processor_finefake
            logger.info(f"Loading BERT tokenizer from local path (FineFake): {self.bert_model_path}")
            bert_tokenizer_finefake = BertTokenizer.from_pretrained(self.bert_model_path)
            logger.info(f"Loading CLIP processor from local path (FineFake): {self.clip_model_path}")
            clip_processor_finefake = CLIPProcessor.from_pretrained(self.clip_model_path)

        elif self.dataset == "weibo":
            self.root_path = self.config.get('weibo_data_dir')
            self.train_path = os.path.join(self.root_path, 'train_origin.csv')
            self.val_path = os.path.join(self.root_path, 'val_origin.csv')
            self.test_path = os.path.join(self.root_path, 'test_origin.csv')
            self.category_dict = WEIBO_CATEGORY_DICT
            self.bert_model_path = self.config.get('bert_model_path_weibo')
            self.vocab_file = self.config.get('vocab_file')

        elif self.dataset == "weibo21":
            self.root_path = self.config.get('weibo21_data_dir')
            self.train_path = os.path.join(self.root_path, 'train_datasets.xlsx')
            self.val_path = os.path.join(self.root_path, 'val_datasets.xlsx')
            self.test_path = os.path.join(self.root_path, 'test_datasets.xlsx')
            self.category_dict = WEIBO21_CATEGORY_DICT
            self.bert_model_path = self.config.get('bert_model_path_weibo')
            self.vocab_file = self.config.get('vocab_file')
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        logger.info(f"Dataset: {self.dataset}, root path: {self.root_path}")

    def get_dataloader(self):
        train_loader, val_loader, test_loader = None, None, None

        if self.dataset == "finefake":
            logger.info("Loading FineFakeDataset with an in-memory 6:2:2 split.")
            if bert_tokenizer_finefake is None or clip_processor_finefake is None:
                raise RuntimeError("The global tokenizer/processor required by FineFake is not loaded.")

            img_size = 224
            clip_max_len = 77
            common_kwargs = {
                "root_path": self.root_path,
                "bert_tokenizer_instance": bert_tokenizer_finefake,
                "clip_processor_instance": clip_processor_finefake,
                "image_size": img_size,
                "bert_max_len": self.max_len,
                "clip_max_len": clip_max_len,
                "seed": self.config.get("finefake_split_seed", 2026),
                "include_knowledge_text": self.config.get("finefake_include_knowledge_text", True),
            }

            train_dataset = FineFakeDataset(split="train", **common_kwargs)
            val_dataset = FineFakeDataset(split="val", **common_kwargs)
            test_dataset = FineFakeDataset(split="test", **common_kwargs)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batchsize,
                shuffle=True,
                collate_fn=collate_fn_finefake,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batchsize,
                shuffle=False,
                collate_fn=collate_fn_finefake,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batchsize,
                shuffle=False,
                collate_fn=collate_fn_finefake,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True,
            )

        elif self.dataset == "weibo":
            logger.info("Loading utils.weibo_clip_dataloader.bert_data for the Weibo dataset.")
            loader = WeiboDataLoaderClass(max_len=self.max_len,
                                          batch_size=self.batchsize,
                                          vocab_file=self.vocab_file,
                                          category_dict=self.category_dict,
                                          num_workers=self.num_workers,

                                          cn_clip_model_name=self.config.get('cn_clip_model_name', "ViT-B-16"),
                                          cn_clip_download_root=self.config.get('cn_clip_download_root', './')
                                          )

            train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
            train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
            val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
            val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
            test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
            test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

            train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
            val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
            test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)

        elif self.dataset == "weibo21":
            logger.info("Loading utils.weibo21_clip_dataloader.bert_data for the Weibo21 dataset.")
            loader = Weibo21DataLoaderClass(max_len=self.max_len,
                                             batch_size=self.batchsize,
                                             vocab_file=self.vocab_file,
                                             category_dict=self.category_dict,
                                             num_workers=self.num_workers,

                                             )

            train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
            train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
            val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
            val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
            test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
            test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

            train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
            val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
            test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)
        else:
            raise ValueError(f"Dataset {self.dataset} has no dataloader logic defined.")

        logger.info(f"Number of training batches: {len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'}")
        logger.info(f"Number of validation batches: {len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'}")
        logger.info(f"Number of test batches: {len(test_loader) if hasattr(test_loader, '__len__') else 'N/A'}")
        return train_loader, val_loader, test_loader

    def main(self):
        logger.info(f"Start run: Dataset {self.dataset}, model config name {self.model_name}")
        train_loader, val_loader, test_loader = self.get_dataloader()

        trainer = None
        if self.dataset == "finefake":
            if self.model_name != 'domain_finefake':
                logger.warning(f"FineFake usually uses 'domain_finefake' model, current model is '{self.model_name}'.")
            trainer = DOMAINTrainerFineFake(
                emb_dim=self.emb_dim,
                mlp_dims=self.config['model_params']['mlp']['dims'],
                bert_path_or_name=self.bert_model_path,
                clip_path_or_name=self.clip_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                dropout=self.config['model_params']['mlp']['dropout'],
                weight_decay=self.config.get('weight_decay', 5e-5),
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                metric_key_for_early_stop=self.early_stop_metric_key,
                epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
            )
        elif self.dataset == "weibo" or self.dataset == "weibo21":
            if self.model_name != 'domain_weibo':
                 logger.warning(f"Weibo/Weibo21 usually uses 'domain_weibo' model, current model is '{self.model_name}'.")
            trainer = DOMAINTrainerWeibo(
                emb_dim=self.emb_dim,
                mlp_dims=self.config['model_params']['mlp']['dims'],
                bert=self.bert_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=self.config['model_params']['mlp']['dropout'],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=self.config.get('weight_decay', 5e-5),
                save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}"),
                early_stop=self.early_stop,
                epoches=self.epoch,
                metric_key_for_early_stop=self.early_stop_metric_key
            )
        else:
            raise ValueError(f"Dataset {self.dataset} has no matched trainer initialization logic.")

        logger.info("Trainer initialized. Start training...")
        result = trainer.train()
        logger.info("Training process finished.")
        return result
