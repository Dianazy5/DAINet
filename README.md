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

FineFake Dataset: For the FineFake dataset, we follow the work from [(Zhou et al., 2024)](https://arxiv.org/abs/2404.01336). You can download the dataset from the official [FineFake repository](https://github.com/Accuser907/FineFake).

Weibo21 Dataset: For the Weibo21 dataset, we follow the work from [(Ying et al., 2023)](https://github.com/yingqichao/fnd-bootstrap). You should send an email to Dr. Qiong Nan to get the complete multimodal multi-domain dataset Weibo21.

Weibo Dataset: For the Weibo dataset, we adhere to the work from [(Wang et al., 2022)](https://github.com/yaqingwang/EANN-KDD18) and follow the data processing procedure of [MMDFND](https://github.com/yutchina/MMDFND).

### Data Storage

Place the processed Weibo data in the `./data` directory.

Place the Weibo21 data in the `./Weibo_21` directory.

Place the FineFake data in the `./FineFake` directory.

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

### Training Hyperparameters

- Dropout: `0.2`.
- Weight decay: `5e-5`.
- Auxiliary-loss weight $\alpha_{aux}$: linearly decreases from `0.45` to `0.15` over 50 epochs.
- Orthogonal-constraint weight $\alpha_{orth}$: linearly increases from `0.005` to `0.03` over 50 epochs.

### Model Parameters

DAINet contains approximately 453.8M parameters in total, including the frozen pretrained encoders.

## Dataset Statistics

### Weibo21

| Domain | Real | Fake | Total |
| --- | ---: | ---: | ---: |
| Science | 143 | 93 | 236 |
| Military | 121 | 222 | 343 |
| Education | 243 | 248 | 491 |
| Disasters | 185 | 591 | 776 |
| Politics | 306 | 546 | 852 |
| Health | 485 | 515 | 1,000 |
| Finance | 959 | 362 | 1,321 |
| Entertainment | 1,000 | 440 | 1,440 |
| Society | 1,198 | 1,471 | 2,669 |
| **All** | **4,640** | **4,488** | **9,128** |

### FineFake

| Domain | Real | Fake | Total |
| --- | ---: | ---: | ---: |
| Politics | 3,722 | 2,005 | 5,727 |
| Entertainment | 2,514 | 1,185 | 3,699 |
| Business | 527 | 476 | 1,003 |
| Health | 438 | 272 | 710 |
| Society | 2,236 | 1,703 | 3,939 |
| Conflict | 979 | 739 | 1,718 |
| Uncategorized | 91 | 22 | 113 |
| **All** | **10,507** | **6,402** | **16,909** |
