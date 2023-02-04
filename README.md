# KnowledgeMiningWithSceneText
https://github.com/MCLAB-OCR/KnowledgeMiningWithSceneText

This is the official repo of the paper [Knowledge Mining with Scene Text for Fine-Grained Recognition](https://doi.org/10.1109/CVPR52688.2022.00458) in CVPR 2022. 
If you find our repo, paper or dataset helpful, please star this repo, cite our paper. Thank you!

## Installation
### 1. Get the code
First, clone this repo and cd into it.
Then clone these codes:
```
git clone https://github.com/AndresPMD/Fine_Grained_Clf
git clone https://github.com/jeonsworld/ViT-pytorch
git clone https://github.com/allenai/kb
git clone https://github.com/matt-peters/allennlp.git
```

### 2. Create environment
```
conda env create -f env.yml
conda activate vit_kb

pip install -r exact_requirements.txt

cd allennlp; git checkout 2d7ba1cb108428aaffe2dce875648253b44cb5ba
pip install -e .
cd ..

cd kb
pip install -r requirements.txt 
python -c "import nltk; nltk.download('wordnet')"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
pip install -e .
cd ..
```

## Dataset prepare
### 1. Download datasets
see [README_datasets.md](README_datasets.md).
### 2. Put in `datasets/`
Folder structure
```
datasets
├── activity
│   ├── texts
│   └── images
├── bottle
│   ├── google_ocr
│   └── images
└── context
    ├── google_ocr
    └── JPEGImages
```

## Usage
### 1. Train
```
python main.py -c CONFIG_PATH
```
for example:
```
python main.py -c configs/train_knowbert_attention_activity.toml
```
You can also pass parameters like this:
```
python main.py -c CONFIG_PATH --text_dir ./datasets/activity/texts --cfgs OUTPUT_DIR ./outputs NUM_EPOCHS 50 BATCH_SIZE_PERGPU 8
```
The parameters after "--cfgs" are config items in configs/*.toml
### 2. Test
```
python main.py -c TEST_CONFIG_PATH
```

## Trouble Shootings
### 1
TypeError: ArrayField.empty_field: return type \`None\` is not a \`<class 'allennlp.data.fields.field.Field'>\`.

Solution:

pip install overrides==3.1.0

### 2
ModuleNotFoundError: No module named 'sklearn.utils.linear_assignment_'

Solution:

pip install scikit-learn==0.22.1


### 3 
Error when run "pip install en_core_web_sm"

Solution:

conda install spacy-model-en_core_web_sm

### 4
if stuck when run `pip install -r kb/requirement.txt`, comment out the "git+git://" line in the kb/requirements.txt

## Sensitive Detection Demo
https://user-images.githubusercontent.com/33376945/198250812-59741ff5-2f2f-4363-a7e6-986773479fa6.mp4

## Citation
```
@inproceedings{Wang2022_KnowledgeMining,
  author    = {Wang, Hao and Liao, Junchao and Cheng, Tianheng and Gao, Zewen and Liu, Hao and Ren, Bo and Bai, Xiang and Liu, Wenyu},
  title     = {Knowledge Mining with Scene Text for Fine-Grained Recognition},
  booktitle = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4614--4623},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/CVPR52688.2022.00458},
  doi       = {10.1109/CVPR52688.2022.00458},
}
```

## Acknowledgments
https://github.com/AndresPMD/Fine_Grained_Clf

https://github.com/allenai/kb

https://github.com/rwightman/pytorch-image-models

https://github.com/huggingface/transformers



