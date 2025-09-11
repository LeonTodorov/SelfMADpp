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
- `<model_type>` is one of `TIMMModel`, `HRNetSeg`, `CLIPVision`, `CLIPDual` or `CLIPFuse`
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

| TIMMModel | HRNetSeg | CLIPVision | CLIPDual | CLIPFuse | 
| --------- | -------- | ---------- | -------- | -------- |
| [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/Eb6qILq4CIlPnENe8QKZWrQBilU_0Zz6Qme3D9jM_zyIRA?e=Hrl5TB) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EXWGEAGzYjBFuVQCCaAfgLwBVrSCTHi6x_SJQWbCgRMEBw?e=yPTdg7) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/ETF1ufG-BcJLiQqaJQuAy6kB4Coy4KF5QQ9Kb0QV-UWR-w?e=Hvy76d) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EUWVQJJS93hEpIX-pQ-WgREBkY1AX0fnholv45ehqps6sg?e=LLzI4v) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EVXmFPD4EF1Lrzq-c6OHrhcBGLExLwlIcKnQ2fWYzP0HZQ?e=bevDd6) |

CLIPFuse model trained with different datasets:
| SMDD | SMDD w/o augments| FRGC | FRLL | FaceForensics++ | ONOT | FLUXSynID |  
| ---- | -----------------| ---- | ---- | --------------- | ---- | --------- |
| [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EVXmFPD4EF1Lrzq-c6OHrhcBGLExLwlIcKnQ2fWYzP0HZQ?e=yogT0O) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EZI0RrkMGaxHmPKTfE5I2LEBNWJGgBDFqn0vZKg_PfFjLw?e=HubdR1) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EYxia_Q8v8VGnziMy3jTrssB0AHV6he9R_Iy07LtTJ11tg?e=pnnAya) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EWgvx3YNGrlBsSeiWCxtFxYBPONbgj5GgO1Yt9Vgp1tPNQ?e=zqss07) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EfTCP0VQHVxKs0j9tjyItBcBYlOyT6rsKFFApElDDPwVMA?e=G2eZ1O) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EYinbIFaO9tAo18KdjpQ1VIBBN8wRhiMOo7it1tWjNQbCA?e=OHXbV7) | [Link](https://unilj-my.sharepoint.com/:u:/g/personal/lt9980_student_uni-lj_si/EaH-5UOlPSdJuhNQeamCSkoBus0PyFzgvlZO-Ct1as2CLA?e=pFfMcO) |
## Citation
If our work contributes to your research, we would appreciate it if you cite our paper:
```bibtex
# TODO
˛˛˛
