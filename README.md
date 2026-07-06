# DAINet

This repository contains the implementation of a multi-modal multi-domain fake news detection project. It supports FineFake, Weibo, and Weibo21 datasets.

## Requirements

Run the following command to deploy the environment:

```bash
pip install -r requirements.txt
```

## Directory Structure

```text
|-- data/
|-- FineFake/
|-- model/
|   |-- bert.py
|   |-- domain_finefake.py
|   |-- domain_weibo.py
|   |-- layers.py
|   |-- models_mae.py
|-- pretrained_model/
|-- util/
|-- utils/
|   |-- clip_dataloader.py
|   |-- dataloader.py
|   |-- domain_labels.py
|   |-- utils.py
|   |-- utils_finefake.py
|   |-- utils_weibo.py
|   |-- weibo21_clip_dataloader.py
|   |-- weibo_clip_dataloader.py
|-- Weibo_21/
|-- clip_cn_vit-b-16.pt
|-- clip_data_pre.py
|-- data_pre.py
|-- FineFake_dataset.py
|-- mae_pretrain_vit_base.pth
|-- main.py
|-- models_mae.py
|-- run.py
|-- weibo21_clip_data_pre.py
|-- weibo21_data_pre.py
```

## Data Preparation

Data Splitting: In the experiments, we maintain the same data splitting scheme as the benchmarks.

FineFake Dataset: For the FineFake dataset, we follow the work from [(Zhou et al., 2024)](https://arxiv.org/abs/2404.01336). You can download the dataset from the official [FineFake repository](https://github.com/Accuser907/FineFake), then place the data in the `./FineFake` directory.

Weibo21 Dataset: For the Weibo21 dataset, we follow the work from [(Ying et al., 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to Dr. Qiong Nan to get the complete multimodal multi-domain dataset Weibo21.

Weibo Dataset: For the Weibo dataset, we adhere to the work from [(Wang et al., 2022)](https://github.com/yaqingwang/EANN-KDD18). In addition, domain labels are incorporated into this dataset. You can download the final processed data from [Baidu Netdisk](https://pan.baidu.com/s/1TGc-8RUt6BIHO1rjnzuPxQ), code: `qwer`.

Data Storage:

Place the processed Weibo data in the `./data` directory.

Place the Weibo21 data in the `./Weibo_21` directory.

Data preparation: Use `clip_data_pre.py`, `data_pre.py`, `weibo21_data_pre.py`, and `weibo21_clip_data_pre.py` to preprocess the data of Weibo and Weibo21, respectively, in order to save time during the data loading phase.

## Pretrained Models

Roberta: You can download the pretrained Roberta model from [Roberta](https://drive.google.com/drive/folders/1y2k22iMG1i1f302NLf-bj7UEe9zwTwLR?usp=sharing) and move all files into the `./pretrained_model/chinese_roberta_wwm_base_ext_pytorch` directory.

BERT: Place the English BERT model in the `./pretrained_model/bert-base-uncased` directory.

CLIP: Place the English CLIP model in the `./pretrained_model/clip-vit-base-patch16` directory.

MAE: Download the pretrained MAE model from ["Masked Autoencoders: A PyTorch Implementation"](https://github.com/facebookresearch/mae) and move it into the root directory as `./mae_pretrain_vit_base.pth`.

Chinese-CLIP: Download the pretrained CLIP model from ["Chinese-CLIP"](https://github.com/OFA-Sys/Chinese-CLIP) and move it into the root directory as `./clip_cn_vit-b-16.pt`.

## Training

Start training with:

```bash
python main.py \
  --model_name <domain_finefake|domain_weibo> \
  --dataset <finefake|weibo|weibo21> \
  --batchsize 64 \
  --seed <seed>
```

Use `domain_finefake` for FineFake. Use `domain_weibo` for Weibo and Weibo21.
