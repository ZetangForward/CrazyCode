# StrokeNUWA: Tokenizing Strokes for Vector Graphic Synthesis
Implementation of the paper [StrokeNUWA: Tokenizing Strokes for Vector Graphic Synthesis](https://arxiv.org/abs/2401.17093), which is a pioneering work exploring a better visual representation ''stroke tokens'' on vector graphics, which is inherently visual semantics rich, naturally compatible with LLMs, and highly compressed.
<p align="center">  
  <img src="assets/main_figure.png" width="100%" height="60%">  
</p>  

## Model Architecture
### VQ-Stroke
VQ-Stroke modules encompasses two main stages: “Code to Matrix” stage that transforms SVG code into the matrix format suitable for model input, and “Matrix to Token” stage that transforms the matrix data into stroke tokens.
<figure align="center">
  <img src="assets/VQ-Stroke.png" width="80%" height="50%">
  <figcaption>Overview of VQ-Stroke.</figcaption>
</figure>

<figure align="center">
  <img src="assets/ModelArchitecture.png" width="80%" height="50%">
  <figcaption>Overview of Down-Sample Blocks and Up-Sample Blocks.</figcaption>
</figure>



## Automatic Evaluation Results
<p align="center">  
  <img src="assets/evaluation.png" width="100%" height="50%">  
</p> 



## Setup

We check the reproducibility under this environment.
- Python 3.10.13
- CUDA 11.1

### Environment Installation

Prepare your environment with the following command
```Shell
git clone https://github.com/ProjectNUWA/StrokeNUWA.git
cd StrokeNUWA

conda create -n strokenuwa python=3.9
conda activate strokenuwa

# install conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install requirements
pip install -r requirements.txt
```

### Model Preparation
We utilize [Flan-T5 (3B)](https://huggingface.co/google/flan-t5-xl) as our backbone. Download the model and place it under the ``./ckpt`` directory.

### Dataset Preparation

#### [FIGR-8-SVG](https://github.com/marcdemers/FIGR-8-SVG) Dataset

Download the raw FIGR-8 dataset from [[Link]](https://github.com/marcdemers/FIGR-8-SVG) and follow [Iconshop](https://icon-shop.github.io/) to further preprocess the datasets. (We thank @[Ronghuan Wu](https://github.com/kingnobro) --- author of Iconshop for providing the preprocessing scripts.)


## Model Training and Inference
We customize the training code based on the [LLaMA-X](https://github.com/AetherCortex/Llama-X)

### Stage 1: Training the VQ-Stroke
Please Check ``trainer/src/configs/hostfile`` and ``trainer/src/configs/deepspeed_config_2.json`` first, where the current code is designed for 64 NVIDIA V100 GPUs (8 GPUs x 8 nodes).
```Shell
cd trainer/src/scripts
bash scripts/train.sh
```

### Inference
```Shell
cd trainer/src/scripts
bash scripts/inference.sh
```

## Evaluation
You can run the following command to evaluate the generated results for the RICO dataset (We have released generated results of RICO dataset in ``data/generated_results/rico`` as an example).

```Shell
python evaluate.py \
    --file_dir data/generated_results/rico \
    --intermediate_saved_path data/generated_results/rico/all_gen.pt \
    --golden_file data/generated_results/rico/golden.jsonl \
    --fid_model_name_or_path models/rico25-max25 \
    --cluster_model models/rico25-max25/rico25_max25_kmeans_train_clusters.pkl \
    --dataset_name rico25 \
    --dataset_path data/rico25-max25 \
```

``NOTICE``: just change the dataset name, dataset path, and generated results path if you want to evaluate other datasets.


## Acknowledgement
We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LLaMA-X](https://github.com/AetherCortex/Llama-X) &#8194;