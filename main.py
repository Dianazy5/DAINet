import argparse
import logging
import os
import random

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="domain", help="Model architecture name. Use domain_finefake for FineFake and domain_weibo for Weibo/Weibo21.")
parser.add_argument("--dataset", default="finefake", choices=["finefake", "weibo", "weibo21"], help="Dataset name.")
parser.add_argument("--epoch", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--max_len", type=int, default=197, help="Maximum BERT sequence length.")
parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
parser.add_argument("--early_stop", type=int, default=100, help="Early-stopping patience.")
parser.add_argument("--early_stop_metric", default="acc", choices=["acc", "F1", "metric"], help="Metric used for early stopping.")

parser.add_argument("--bert_model_path_finefake", default="./pretrained_model/bert-base-uncased", help="Local path of the English BERT model used by FineFake.")
parser.add_argument("--clip_model_path_finefake", default="./pretrained_model/clip-vit-base-patch16", help="Local path of the English CLIP model used by FineFake.")
parser.add_argument("--bert_model_path_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch", help="Local path of the BERT model used by Weibo/Weibo21.")
parser.add_argument("--bert_vocab_file_weibo", default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt", help="BERT vocabulary file used by Weibo/Weibo21.")
parser.add_argument("--w2v_vocab_file", default="./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv", help="Word2Vec vocabulary path for the Weibo/Weibo21 pipeline.")

parser.add_argument("--finefake_data_dir", default="./FineFake/", help="FineFake dataset root directory.")
parser.add_argument("--weibo_data_dir", default="./data/", help="Weibo dataset root directory.")
parser.add_argument("--weibo21_data_dir", default="./Weibo_21/", help="Weibo21 dataset root directory.")
parser.add_argument("--batchsize", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=2024, help="Random seed.")
parser.add_argument("--gpu", default="0", help="GPU id.")
parser.add_argument("--bert_emb_dim", type=int, default=768, help="BERT embedding dimension.")
parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
parser.add_argument("--emb_type", default="bert", help="Embedding type. Kept mainly for compatibility with the Weibo pipeline.")
parser.add_argument("--save_param_dir", default="./param_model", help="Directory for saved model parameters.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run

current_seed = args.seed
current_batchsize = args.batchsize
current_lr = args.lr
current_early_stop = args.early_stop
if args.dataset == "finefake":
    current_model_name = args.model_name if args.model_name != "domain" else "domain_finefake"
else:
    current_model_name = args.model_name if args.model_name != "domain" else "domain_weibo"

random.seed(current_seed)
np.random.seed(current_seed)
torch.manual_seed(current_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(current_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

emb_dim = args.bert_emb_dim

config = {
    "use_cuda": True if torch.cuda.is_available() else False,
    "dataset": args.dataset,
    "model_name": current_model_name,
    "finefake_data_dir": args.finefake_data_dir,
    "weibo_data_dir": args.weibo_data_dir,
    "weibo21_data_dir": args.weibo21_data_dir,
    "bert_model_path_finefake": args.bert_model_path_finefake,
    "clip_model_path_finefake": args.clip_model_path_finefake,
    "bert_model_path_weibo": args.bert_model_path_weibo,
    "bert_vocab_file_weibo": args.bert_vocab_file_weibo,
    "batchsize": current_batchsize,
    "max_len": args.max_len,
    "early_stop": current_early_stop,
    "early_stop_metric": args.early_stop_metric,
    "num_workers": args.num_workers,
    "emb_type": args.emb_type,
    "weight_decay": 5e-5,
    "model_params": {"mlp": {"dims": [512], "dropout": 0.2}},
    "emb_dim": emb_dim,
    "lr": current_lr,
    "epoch": args.epoch,
    "seed": current_seed,
    "save_param_dir": args.save_param_dir,
    "finefake_split_seed": 2026,
    "finefake_include_knowledge_text": True,
}

if args.dataset in {"weibo", "weibo21"}:
    config["vocab_file"] = args.bert_vocab_file_weibo
    config["bert"] = args.bert_model_path_weibo
    config["w2v_vocab_file"] = args.w2v_vocab_file

if __name__ == "__main__":
    runner = Run(config=config)
    runner.main()