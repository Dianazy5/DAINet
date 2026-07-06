

import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='domain', help="要使用的模型架构名称。FineFake使用'domain_finefake'，weibo/weibo21使用'domain_weibo'。")
parser.add_argument('--dataset', default='finefake', choices=['finefake', 'weibo', 'weibo21'], help="要使用的数据集。")
parser.add_argument('--epoch', type=int, default=50, help="训练周期数。")
parser.add_argument('--max_len', type=int, default=197, help="BERT 的最大序列长度。")
parser.add_argument('--num_workers', type=int, default=4, help="dataloader 的 worker 数量。")
parser.add_argument('--early_stop', type=int, default=100, help="早停的耐心值。对于Weibo默认为6，FineFake为10。")
parser.add_argument('--early_stop_metric', default='acc', choices=['acc', 'F1', 'metric'], help="用于早停的指标 ('acc'、'F1' 或 'metric')，FineFake/Weibo可使用'metric'（宏F1）。")


parser.add_argument('--bert_model_path_finefake', default='./pretrained_model/bert-base-uncased', help="FineFake 使用的英文 BERT 模型本地路径。")
parser.add_argument('--clip_model_path_finefake', default='./pretrained_model/clip-vit-base-patch16', help="FineFake 使用的英文 CLIP 模型本地路径。")


parser.add_argument('--bert_model_path_weibo', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch', help="Weibo/Weibo21 BERT 模型本地路径。")
parser.add_argument('--bert_vocab_file_weibo', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt', help="Weibo/Weibo21 BERT 词汇表文件。")

parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv', help="Weibo/Weibo21 word2vec 词汇表文件路径。")



parser.add_argument('--finefake_data_dir', default='./FineFake/', help="FineFake 数据集根目录。")
parser.add_argument('--weibo_data_dir', default='./data/', help="Weibo 数据集根目录。")
parser.add_argument('--weibo21_data_dir', default='./Weibo_21/', help="Weibo21 数据集根目录。")

parser.add_argument('--batchsize', type=int, default=64, help="批处理大小。")
parser.add_argument('--seed', type=int, default=2024, help="随机种子。Weibo默认为3074，FineFake为2024。")
parser.add_argument('--gpu', default='0', help="要使用的 GPU ID。")
parser.add_argument('--bert_emb_dim', type=int, default=768, help="BERT 嵌入维度。")
parser.add_argument('--lr', type=float, default=0.0003, help="学习率。FineFake默认为0.0001。")
parser.add_argument('--emb_type', default='bert', help="嵌入类型 (主要为Weibo流程保留，FineFake固定使用bert)。")
parser.add_argument('--save_param_dir', default= './param_model', help="保存模型参数的目录。")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


try:
    from run import Run
except ImportError as e:
     logger.error(f"无法导入 Run 类: {e}. 请确保 run.py 文件存在且无误。")
     exit()

import torch
import numpy as np
import random


current_seed = args.seed
current_batchsize = args.batchsize
current_lr = args.lr
current_early_stop = args.early_stop
if args.dataset == 'finefake':
    current_model_name = args.model_name if args.model_name != 'domain' else 'domain_finefake'
else: # weibo or weibo21
    current_model_name = args.model_name if args.model_name != 'domain' else 'domain_weibo'



random.seed(current_seed)
np.random.seed(current_seed)
torch.manual_seed(current_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logger.info(f"使用数据集: {args.dataset}")
logger.info(f"使用模型: {current_model_name}")
logger.info(f"使用 GPU: {args.gpu}")
logger.info(f"种子设置为: {current_seed}")

emb_dim = args.bert_emb_dim


config = {
        'use_cuda': True if torch.cuda.is_available() else False,
        'dataset': args.dataset,
        'model_name': current_model_name,

        'finefake_data_dir': args.finefake_data_dir,
        'weibo_data_dir': args.weibo_data_dir,
        'weibo21_data_dir': args.weibo21_data_dir,

        'bert_model_path_finefake': args.bert_model_path_finefake,
        'clip_model_path_finefake': args.clip_model_path_finefake,
        'bert_model_path_weibo': args.bert_model_path_weibo,
        'bert_vocab_file_weibo': args.bert_vocab_file_weibo,
        # CLIP for Weibo (cn_clip) is usually handled by its own loading mechanism or a model_path string

        'batchsize': current_batchsize,
  
        'max_len': args.max_len,
        'early_stop': current_early_stop,
        'early_stop_metric': args.early_stop_metric,
        'num_workers': args.num_workers,
        'emb_type': args.emb_type,
        'weight_decay': 5e-5,
        'model_params': {'mlp': {'dims': [512], 'dropout': 0.2}},
        'emb_dim': emb_dim,
        'lr': current_lr,
        'epoch': args.epoch,
    
        'seed': current_seed,
        'save_param_dir': args.save_param_dir,
        'finefake_split_seed': 2026,
        'finefake_include_knowledge_text': True,
        }




if args.dataset == 'weibo' or args.dataset == 'weibo21':
    config['vocab_file'] = args.bert_vocab_file_weibo
    config['bert'] = args.bert_model_path_weibo
    config['w2v_vocab_file'] = args.w2v_vocab_file

logger.info(f"--- 完整配置 ---")
for key, value in config.items():
    logger.info(f"{key}: {value}")
logger.info(f"--------------------")

if __name__ == '__main__':
    if config['use_cuda']:
        logger.info(f"CUDA 可用。正在使用 GPU {args.gpu}。")
    else:
    
        logger.warning("CUDA 不可用。正在使用 CPU 运行。")

    runner = Run(config=config)
    runner.main()
    logger.info("运行结束。")
