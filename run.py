



import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPProcessor
import logging

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


try:
    from FineFake_dataset import FineFakeDataset
    logger.info("FineFake_dataset 导入成功。")
except ImportError as e:
    logger.error(f"无法导入 FineFake_dataset: {e}. 如果未使用 FineFake 数据集则忽略。")
    FineFakeDataset = None


WeiboDataLoaderClass = None
Weibo21DataLoaderClass = None

# DataLoader for "weibo" dataset
try:
    from utils.weibo_clip_dataloader import bert_data as WeiboDataLoaderForWeibo
    logger.info("成功从 utils.weibo_clip_dataloader 导入 Weibo 数据加载器 (WeiboDataLoaderForWeibo)。")
    WeiboDataLoaderClass = WeiboDataLoaderForWeibo
except ImportError as e:
    logger.error(f"错误：无法从 utils.weibo_clip_dataloader 导入 Weibo 数据加载器。错误: {e}")


# DataLoader for "weibo21" dataset
try:
    from utils.weibo21_clip_dataloader import bert_data as Weibo21DataLoaderForWeibo21 # [cite: 39]
    logger.info("成功从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器 (Weibo21DataLoaderForWeibo21)。")
    Weibo21DataLoaderClass = Weibo21DataLoaderForWeibo21
except ImportError as e:
    logger.error(f"错误：无法从 utils.weibo21_clip_dataloader 导入 Weibo21 数据加载器。错误: {e}")




DOMAINTrainerFineFake = None
DOMAINTrainerWeibo = None

try:
    from model.domain_finefake import Trainer as DOMAINTrainerFineFake
    logger.info("model.domain_finefake.Trainer 导入成功。")
except ImportError as e:
    logger.warning(f"无法导入 model.domain_finefake.Trainer。如果未使用FineFake数据集则忽略。错误: {e}")

try:
    from model.domain_weibo import Trainer as DOMAINTrainerWeibo
    logger.info("model.domain_weibo.Trainer 导入成功。")
except ImportError as e:
    logger.warning(f"无法导入 model.domain_weibo.Trainer。如果未使用Weibo/Weibo21数据集则忽略。错误: {e}")



bert_tokenizer_finefake = None
clip_processor_finefake = None


def collate_fn_finefake(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        values = [item[key] for item in batch] #
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                collated_batch[key] = torch.stack(values, dim=0) # [cite: 41]
            except RuntimeError as e:
                logger.error(f"Collate (finefake) 无法为键 '{key}' 堆叠张量 (可能是尺寸不匹配): {e}")
                for i, v_item in enumerate(values): #
                    logger.error(f"  Item {i} shape: {v_item.shape}") # [cite: 42]
                return None
            except Exception as e:
                 logger.error(f"Collate (finefake) 为键 '{key}' 堆叠张量时发生未知错误: {e}")
                 return None
        else:
            collated_batch[key] = values # [cite: 43]
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
        self.max_len = config['max_len'] # BERT max_len
        self.num_workers = config['num_workers'] # [cite: 44]
        self.early_stop = config['early_stop']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']

        if self.dataset == "finefake":
            self.root_path = self.config.get('finefake_data_dir')
            self.bert_model_path = self.config.get('bert_model_path_finefake')
            self.clip_model_path = self.config.get('clip_model_path_finefake')
            self.category_dict = {
                "Business": 0,
                "Conflict": 1,
                "Entertainment": 2,
                "Health": 3,
                "Politics": 4,
                "Society": 5,
                "Uncategorized": 6,
            }
            self.early_stop_metric_key = (
                'metric'
                if self.config.get('early_stop_metric', 'acc') == 'acc'
                else self.config.get('early_stop_metric', 'metric')
            )

            global bert_tokenizer_finefake, clip_processor_finefake
            try:
                logger.info(f"尝试从本地路径加载 BERT tokenizer (FineFake): {self.bert_model_path}")
                bert_tokenizer_finefake = BertTokenizer.from_pretrained(self.bert_model_path)
            except Exception as e:
                logger.error(f"从本地路径加载 BERT tokenizer (FineFake) 失败: {e}") # [cite: 46]
            try:
                logger.info(f"尝试从本地路径加载 CLIP processor (FineFake): {self.clip_model_path}")
                clip_processor_finefake = CLIPProcessor.from_pretrained(self.clip_model_path)
            except Exception as e:
                logger.error(f"从本地路径加载 CLIP processor (FineFake) 失败: {e}") # [cite: 47]

            if bert_tokenizer_finefake is None or clip_processor_finefake is None:
                 logger.warning("FineFake 所需的 BERT tokenizer 或 CLIP processor 未能成功加载。")

        elif self.dataset == "weibo":
            self.root_path = self.config.get('weibo_data_dir')
            self.train_path = os.path.join(self.root_path, 'train_origin.csv')
            self.val_path = os.path.join(self.root_path, 'val_origin.csv') # [cite: 48]
            self.test_path = os.path.join(self.root_path, 'test_origin.csv')
            self.category_dict = { "经济": 0, "健康": 1, "军事": 2, "科学": 3, "政治": 4, "国际": 5, "教育": 6, "娱乐": 7, "社会": 8 }
            self.bert_model_path = self.config.get('bert_model_path_weibo') # [cite: 48]
            self.vocab_file = self.config.get('vocab_file') # This should be bert_vocab_file_weibo from main.py config
            self.early_stop_metric_key = 'metric'


        elif self.dataset == "weibo21": # [cite: 49]
            self.root_path = self.config.get('weibo21_data_dir')
            self.train_path = os.path.join(self.root_path, 'train_datasets.xlsx')
            self.val_path = os.path.join(self.root_path, 'val_datasets.xlsx')
            self.test_path = os.path.join(self.root_path, 'test_datasets.xlsx')
            self.category_dict = { "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3, "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8 }
            self.bert_model_path = self.config.get('bert_model_path_weibo') # [cite: 50]
            self.vocab_file = self.config.get('vocab_file') # This should be bert_vocab_file_weibo from main.py config
            self.early_stop_metric_key = 'metric'
        else:
            raise ValueError(f"未知数据集: {self.dataset}")

        logger.info(f"数据集: {self.dataset}, 根目录: {self.root_path}")


    def get_dataloader(self):
        train_loader, val_loader, test_loader = None, None, None

        if self.dataset == "finefake":
            logger.info("为 FineFake 加载 FineFakeDataset，采用 6:2:2 内存划分。")
            if FineFakeDataset is None:
                raise ImportError("FineFakeDataset 未成功导入，无法为 FineFake 加载数据。")
            if bert_tokenizer_finefake is None or clip_processor_finefake is None:
                raise RuntimeError("FineFake 数据集所需的全局 tokenizer/processor 未加载。")

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
            if WeiboDataLoaderClass is None:
                raise ImportError("Weibo 数据加载器 (WeiboDataLoaderClass from utils.weibo_clip_dataloader) 未成功导入。")
            logger.info("为 Weibo 数据集加载 utils.weibo_clip_dataloader.bert_data")
            loader = WeiboDataLoaderClass(max_len=self.max_len, # BERT max_len [cite: 57]
                                          batch_size=self.batchsize, # [cite: 58]
                                          vocab_file=self.vocab_file, # [cite: 58]
                                          category_dict=self.category_dict, # [cite: 59]
                                          num_workers=self.num_workers, # [cite: 59]

                                          cn_clip_model_name=self.config.get('cn_clip_model_name', "ViT-B-16"), # [cite: 5]
                                          cn_clip_download_root=self.config.get('cn_clip_download_root', './') # [cite: 5]
                                          )

            train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl') # [cite: 61]
            train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl')
            val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
            val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
            test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
            test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

            train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True) # [cite: 62]
            val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
            test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False)


        elif self.dataset == "weibo21":
            if Weibo21DataLoaderClass is None:
                raise ImportError("Weibo21 数据加载器 (Weibo21DataLoaderClass from utils.weibo21_clip_dataloader) 未成功导入。")
            logger.info("为 Weibo21 数据集加载 utils.weibo21_clip_dataloader.bert_data") # [cite: 63]
            loader = Weibo21DataLoaderClass(max_len=self.max_len, # BERT max_len
                                             batch_size=self.batchsize,
                                             vocab_file=self.vocab_file, # [cite: 64]
                                             category_dict=self.category_dict,
                                             num_workers=self.num_workers, # [cite: 65]

                                             # cn_clip_model_name=self.config.get('cn_clip_model_name', "ViT-B-16"), # [cite: 5]
                                             # cn_clip_download_root=self.config.get('cn_clip_download_root', './') # [cite: 5]
                                             )

            train_pkl_path = os.path.join(self.root_path, 'train_loader.pkl')
            train_clip_pkl_path = os.path.join(self.root_path, 'train_clip_loader.pkl') # [cite: 67]
            val_pkl_path = os.path.join(self.root_path, 'val_loader.pkl')
            val_clip_pkl_path = os.path.join(self.root_path, 'val_clip_loader.pkl')
            test_pkl_path = os.path.join(self.root_path, 'test_loader.pkl')
            test_clip_pkl_path = os.path.join(self.root_path, 'test_clip_loader.pkl')

            train_loader = loader.load_data(self.train_path, train_pkl_path, train_clip_pkl_path, True)
            val_loader = loader.load_data(self.val_path, val_pkl_path, val_clip_pkl_path, False)
            test_loader = loader.load_data(self.test_path, test_pkl_path, test_clip_pkl_path, False) # [cite: 68]
        else:
            raise ValueError(f"数据集 {self.dataset} 的 Dataloader 逻辑未定义。")

        if train_loader is None or val_loader is None or test_loader is None:
            logger.error("一个或多个 dataloader 初始化失败。正在退出。")
            raise RuntimeError("Dataloader 初始化失败。")

        logger.info(f"训练加载器批次数: {len(train_loader) if hasattr(train_loader, '__len__') else 'N/A'}") # [cite: 68, 69]
        logger.info(f"验证加载器批次数: {len(val_loader) if hasattr(val_loader, '__len__') else 'N/A'}")
        logger.info(f"测试加载器批次数: {len(test_loader) if hasattr(test_loader, '__len__') else 'N/A'}")
        return train_loader, val_loader, test_loader

    def main(self):
        logger.info(f"开始运行: 数据集 {self.dataset}, 模型配置名 {self.model_name}")
        train_loader, val_loader, test_loader = self.get_dataloader()

        trainer = None
        if self.dataset == "finefake":
            if DOMAINTrainerFineFake is None: # [cite: 70]
                raise ImportError("FineFake 对应的 Trainer 未加载，无法训练。")
            if self.model_name != 'domain_finefake':
                logger.warning(f"FineFake 通常使用 'domain_finefake' 模型, 当前为 '{self.model_name}'.")
            trainer = DOMAINTrainerFineFake(
                emb_dim=self.emb_dim,
                mlp_dims=self.config['model_params']['mlp']['dims'], # [cite: 71]
                bert_path_or_name=self.bert_model_path,
                clip_path_or_name=self.clip_model_path,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                val_loader=val_loader, # [cite: 72]
                test_loader=test_loader,
                dropout=self.config['model_params']['mlp']['dropout'],
                weight_decay=self.config.get('weight_decay', 5e-5),
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                metric_key_for_early_stop=self.early_stop_metric_key,
                epoches=self.epoch, # [cite: 73]
                save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}")
            )
        elif self.dataset == "weibo" or self.dataset == "weibo21":
            if DOMAINTrainerWeibo is None:
                raise ImportError(f"DOMAINTrainerWeibo 未加载，无法训练 {self.dataset}。")
            if self.model_name != 'domain_weibo': # [cite: 74]
                 logger.warning(f"Weibo/Weibo21 通常使用 'domain_weibo' 模型, 当前为 '{self.model_name}'.")
            trainer = DOMAINTrainerWeibo(
                emb_dim=self.emb_dim,
                mlp_dims=self.config['model_params']['mlp']['dims'],
                bert=self.bert_model_path,
                use_cuda=self.use_cuda, # [cite: 75]
                lr=self.lr,
                dropout=self.config['model_params']['mlp']['dropout'],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict, # [cite: 76]
                weight_decay=self.config.get('weight_decay', 5e-5),
                save_param_dir=os.path.join(self.save_param_dir, f"{self.dataset}_{self.model_name}"),
                early_stop=self.early_stop,
                epoches=self.epoch
            )
        else:
            raise ValueError(f"数据集 {self.dataset} 没有匹配的 Trainer 初始化逻辑。") # [cite: 77]

        if trainer:
            logger.info("Trainer 初始化完成。开始训练...")
            try:
                trainer.train()
                logger.info("训练过程结束。")
            except Exception as train_e:
                logger.exception(f"训练过程中发生严重错误: {train_e}") # [cite: 78]
        else:
            logger.error("Trainer 未初始化。")

