# Delving into Noisy Label Detection with Clean Data

This repository provides codes for the manuscript 'Delving into Noisy Label Detection with Clean Data'.

## Preferred Prerequisites

- Python (3.8)
- CUDA
- numpy~=1.21.2
- torch~=1.9.1
- matplotlib~=3.4.3
- Pillow~=8.3.2
- torchvision~=0.10.1
- pandas~=1.3.4
- statsmodels~=0.13.2

You can download the python packages manually or run `pip install -r requirements.txt`.

## Code Overview

```
.
├── data  
    ├── __init__.py
    ├── dataloader_clothing1M_BH.py # Code for load Clothing1M
    ├── utils.py        # Code for noisify clean dataset
├── models                # Code for defining network architectures
    ├── __init__.py
    ├── imagenet_resnet.py
    ├── nine_layer_cnn.py
    ├── resnet.py  
├── pvalue
		├── __init__.py
		├── methods.py # Code for calculate p-value
├── BHN.py # Code for method BHN
├── utility.py # Code for customized utilities (setting gpu, setting outdir, etc.)
└── utils_experiments.py # Code for evaluation
```

## Training the neural network with a part of clean data and detecting corrupted examples with the score function using BHN

Here is an example.

- Train ResNet-18 model on 20% of the overall training set of CIFAR-10.

```bash
python BHN.py --dataset cifar10 --noise_rate 0.4 --noise_type instance --leave_ratio 0.4 --net resnet18 --n_epochs 200 --l_train_second_ratio 1.0 --score_evaluation
```

- Detect corrupted labels on the synthetic CIFAR-10 with the Inst. 0.4 noise using final_model.tar in \$INPUT_DIR\$, where \$INPUT_DIR\$ is the directory of final_model.tar. 

```bash
python BHN.py --dataset cifar10 --net resnet18 --noise_type instance --leave_ratio 0.4 --input_dir $INPUT_DIR$ --model_file final_model.tar --load_model_direct_score --score_evaluation --alpha 0.1
```

## Training on the selected data by BHN

Here is an example. 

- Train ResNet-18 model on the selected CIFAR-10 examples in sel_clean.txt

```bash
python BHN.py --dataset cifar10 --noise_rate 0.4 --noise_type instance --leave_ratio 0.4 --net resnet18 --n_epochs 200 --l_train_second_ratio 1.0 --lr 0.1 --batch_size 128 --weight_decay 5e-4 --momentum 0.9 --input_dir $INPUT_DIR$ --labeledsetpth sel_clean.txt --retrain_selectedset --lr 0.1 --batch_size 128 --weight_decay 5e-4 --momentum 0.9 
```

, where \$INPUT_ DIR\$ is the directory where the file that stores the selected examples lies in.







