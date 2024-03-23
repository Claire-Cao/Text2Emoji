# Text2Emoji
This is a project where we are trying to predict a few relavent emojis corresponding to an input sentence. This is done by building a sequence-to-sequence model (here we are using [BART]( https://arxiv.org/abs/1910.13461)). We are using the same approach to solve this problem as that in the paper [EmojiLM](https://arxiv.org/pdf/2311.01751.pdf), where you can find more details. What is different in this project is that we use our own training strategy to train a reasonable model. Since the author of EmojiLM does not publish the training code, all the training in this repo is from scratch using the pretrained BART [model](https://huggingface.co/facebook/bart-base). We encourage users to compare the result and any feedback is more than welcome.

## Useage
Download the model that has already been trained from [here](https://drive.google.com/drive/folders/1nv-jBNQ1zKWo_P45vg6OIHyOytktGcRP?usp=sharing), then put the folder with the name **models** to the root directory of this repo. You should be good to go for inference. If you want to train by yourself, you need to put the folder with the name **pretrained_models** to the root directory of this repo. Both **models** and **pretrained_models** will be downloaded using the same link as mentioned above.
### Install
We are using [conda](https://docs.anaconda.com/free/miniconda/index.html) with python 3.10 as the virtual environment.\
Run the following command to install necessary packages:\
```pip install -r requirements.txt```
### Inference
```python translate.py```
### Training
To train the model, you need to download the [dataset](https://huggingface.co/datasets/KomeijiForce/Text2Emoji) from hugging face and put the csv file into the directory of **dataset/Text2Emoji**. You do NOT need to run ```split_data.py``` to split the dataset into train/val/test, which is only used to train an encoder only (e.g. BERT) classifier on emoji types. We use 2 GPUs in the distributed training. We have not tested other number of GPUs, but it should work if the number of GPUs >= 2.\
```python finetune_bart.py```
