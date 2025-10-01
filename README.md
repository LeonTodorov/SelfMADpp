# SelfMAD++
Official implementation of the paper:
> **SelfMAD++: Self-Supervised Foundation Model with Local Feature Enhancement for Generalized Morphing Attack Detection**  
> Marija Ivanovska, Leon Todorov, Peter Peer, Vitomir Štruc

SelfMAD++ is a self-supervised morphing attack detection framework that integrates CLIP with fine-grained spatial reasoning. It fuses vision–language alignment via LoRA-adapted CLIP with a high-resolution convolutional branch guided by segmentation, enabling the model to capture both semantic context and subtle spatial artifacts. The framework is trained end-to-end with a multi-objective loss that balances alignment, segmentation consistency, and classification accuracy.

![model](https://github.com/user-attachments/assets/1353de9c-baa7-4489-bcf6-7d189e837af0)
<p align="center"><em>Figure 1:</em>  Detector overview.</p>

![example](https://github.com/user-attachments/assets/6d00abc6-47a6-4839-afcb-53c49baa7d01)
<p align="center"><em>Figure 2:</em> Top to bottom rows — bona fide training sample, corresponding simulated face morphing attack, and facial region mask.</p>

## Python 3.10 - Dependencies
The model is implemented using PyTorch.  
#### Full list of used libraries:
torch==2.3.1  
torchvision==0.18.1  
opencv-python==4.10.0.84  
albumentations==1.4.10  
albucore==0.0.12  
timm==1.0.19  
segmentation_models_pytorch==0.5.0  
open-clip-torch==3.1.0  
#### Installation:
 The necessary packages can be installed via:
```bash
pip install -r requirements.txt
```

## Dataset configuration
#### Evaluation datasets:
Assumed to have the following directory structure:
```
<path to dataset>/<dataset name>
  └── <subset>      
    └── *.png/*.jpg/*.jpeg
```
The expected `<subset>` directories depend on the `<dataset_name>`:  
- **FRLL**: `raw`, `amsl`, `facemorpher`, `opencv`, `stylegan`, `webmorph`  
- **FRGC**: `raw`, `facemorpher`, `opencv`, `stylegan`  
- **FERET**: `raw`, `facemorpher`, `opencv`, `stylegan`  
- **MorDIFF**: `raw`, `morphs_neutral`, `morphs_smiling`  
- **DiMGreedy**: `raw`, `dim_a`, `dim_c`, `fast_dim`, `fast_dim_ode`, `greedy_dim`, `greedy_dim_s`  
- **MorphPIPE**: `raw`, `morphs`  
- **MorCode**: `raw`, `morphs`  
- **MIPGAN_I**: `raw`, `morphs`  
- **MIPGAN_II**: `raw`, `morphs`

Where `raw` corresponds to the subset of bona fide samples, and the remaining subsets denote the specific morphing attack generation methods. Images are assumed to be pre-cropped with a margin of 12.5\% with a relative margin around the ladnmark defined bounding box. The expected paths can be adjusted in `data/data_config.yaml`.

#### Landmarks & Labels:
Landmarks for a dataset can be generated via:
```bash
CUDA_VISIBLE_DEVICES=0,1 python utils/preprocesing/landmarks.py \
-i <path_to_dataset_root> \
```
Labels for a dataset can be generated via:
```bash
CUDA_VISIBLE_DEVICES=0,1 python utils/preprocesing/face_labels.py \
-i <path_to_dataset_root> \
```


#### Training datasets:
Assumed to have the following directory structure:
```
<path to dataset>/<dataset name>     
  └── *.png/*.jpg/*.jpeg

<path to dataset>/<dataset name>_lm
  └── *.npy

<path to dataset>/<dataset name>_lb
  └── *.npy
```

The directories `<dataset_name>_lm` and `<dataset_name>_lb` contain the extracted landmarks and the corresponding labels, respectively. The expected paths can be adjusted in `data/data_config.yaml`.

## Training     
During training checkpoints and logs are saved in the path specified in `data/data_config.yaml`. The training script can be run with:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
-m <model_type> \
-b <batch_size> \
-e <num_epochs> \
-t <train_dataset> \
-lr <learning_rate> \
-n <session_name> \
<--forget>
```
- `<model_type>` is one of `TIMMModel`, `HRNetSeg`, `CLIPVision`, `CLIPDual` or `SelfMADpp`
- `<train_dataset>` is one of `SMDD`, `FF++`, `FRGC`, `FRLL`, `ONOT` or `FLUXSynID`
- `<--forget>` disables saving of logs and checkpoints.

## Evaluation
The evaluation script can be run with:
```bash
CUDA_VISIBLE_DEVICES=0,1 python eval.py \
-p <path_to_checkpoint>
```

## Checkpoints
Different models trained on SMDD:

| TIMMModel | HRNetSeg | CLIPVision | CLIPDual | SelfMAD++ | 
| --------- | -------- | ---------- | -------- | --------- |
| [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/Ea6KQtESTBFBvAi6E46tpWIBqQ-EamQbp1GzC9IumgcEOA?e=HX1OsA) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EaDRPZApGSJApWksNj4h5lUBVAKWjnMnhTBtMQcP7LhRnA?e=GWAuGX) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/Ee7KxzzX1oZMq04jIXkBXL8Bfcz5WuE16nO2OQVn_6J8hQ?e=A4vmsq) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EXdg1ce6bpBIkzsi3mNN2XcBOu3sRvW2x0RdvLGRVoQSTA?e=IZZ7mf) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EXPI80C7nVJGm8L130BWR2IBhcaUoDusyDMBYtXdJxgnHQ?e=oJRNNV) |

SelfMAD++ model trained with different datasets:
| SMDD | SMDD w/o augments| FRGC | FRLL | FaceForensics++ | ONOT | FLUXSynID |  
| ---- | -----------------| ---- | ---- | --------------- | ---- | --------- |
| [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EXPI80C7nVJGm8L130BWR2IBhcaUoDusyDMBYtXdJxgnHQ?e=D4OiAr) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/ESk6lGqDvaNEqmhe2I19hMEBl62_WnKkpCJzEMhkfHgS6w?e=TkqYIx) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/ESoOdu-IHqxNv2tZZ0kLL84Bmemb17dBtm4ImuGkYjl6aA?e=9a1Ajb) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EU1BcLbWOD9KrT6EGJG1cTIBSW9hq8iSkIkGWk46BEyY0g?e=nyRNaa) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EQ1E4_8NcMtIvWz0xd_XgDoB8URmLxXF3zMpwMU3UXKLLw?e=G4IE4h) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EbQNfonN4HVHrdJxBfEHdiwBR1J1Ik8bYks9Te3Rshtteg?e=2dKMbp) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/leon_todorov_fe1_uni-lj_si/EeLT-ElIRgNIuVqV7uzVypgBouOl8VbdrfkaRqtLGF1w0g?e=xIM6bq) |
## Citation
If our work contributes to your research, we would appreciate it if you cite our paper:
```bibtex
# TODO
˛˛˛
