# Noise-free Explanation for Driving Action Prediction

## Introduction
We propose a novel method for generating noise-free explanations for ViT and evaluate our method on driving action prediction task. 
<!-- ![image](/SNNA/paper_img/framework.png) -->
Our framework consists of two stages: self-supervised training and supervised multi-label classification fine-tuning.
<p align="center">
  <img width="500" src="/paper_img/framework.png">
</p>

## Environment Installation
conda (Recommended) - Clone the repository and then create and activate a the conda environment using the provided environment definition:

```
conda env create -f SNNA.yaml
conda activate SNNA
```

## Data
The data used in this project is the [BDD100k](https://doc.bdd100k.com/download.html) and [BDD-OIA](https://twizwei.github.io/bddoia_project/) The data should be downloaded and placed in the `dataset` directory. The directory structure should look like this:
    
```
|-- dataset
    |-- BDD100k
        |-- images
            |-- 100k
                |-- train
                |-- val
                |-- test
    |-- BDD-OIA
        |-- test
        |-- train
        |-- val
        test.json
        train.json
        val.json
```

## Training
To train the model, run the following command:

1. Self-supervised Fine-tuning the model on BDD100k
```
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py --patch_size 8 --batch_size_per_gpu 16 --epochs 200 --saveckp_freq 10 --data_path /dataset/BDD100k/images/100k --output_dir /ckp/
```

2. Supervised Multi-label classification Fine-tuning the classifier on BDD-OIA
```
python multi_label_train.py --num_labels 4 --patch_size 8 --batch_size_per_gpu 4  --epochs 100 --pretrained_weights /ckp/backbone_200.pth --data_path /dataset/BDD-OIA/ --output_dir /ckp/
```

## SNNA Explanation with Notebook *exp_vis.ipynb*
![image](/paper_img/demo.png)


## Credits
This repository is based on the following works:
- [DINO](https://github.com/facebookresearch/dino?tab=readme-ov-file)
- [Transformer-Explainability](Transformer-Explainability)
we thank the authors for their excellent works.

## Citing our paper
If you make use of our work, please cite our paper:
