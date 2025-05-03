# Hybrid Neural Audio Effects

This code repository for the article _Hybrid Neural Audio Effects_, Proceedings of the SMC Conferences. SMC Network, 2024.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/Hybrid-Neural-Audio-Effects_pages/)

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)
3. [VST Download](#vst-download)

<br/>

# Datasets

Datsets are available at the following links:
- [Universal Audio 6176 Channel Strip Dataset](https://zenodo.org/records/3562442)
- [TubeTech CL 1B Dataset](https://www.kaggle.com/datasets/riccardosimionato/tubetech-cl-1b)
- [Akai 4000D reel-to-reel tape recorder Dataset](https://zenodo.org/records/8026272)


# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./Code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=8 )
* --units = The hidden layer size (amount of units) of the network. [ [int] ] (default=8)
* --mini_batch_size - The mini batch size [int] (default=2048)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --datasets TapePreamp --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --datasets TapePreamp --only_inference True
```

The repo include three pre-trained model having only one parameter:
* TapePreamp: 0 emulate the tape recorder, 1 the pre-amp
* CL1BTape: 0 emulate the optical compressor, 1 the tape recorder
* CL1BPreamp: 0 emulate the optical compressor, 1 the pre-amp

and one pre-trained model having 3 parameters:
* CL1BTapePreamp: each parameter indicate how much of a particular effect is to be added.

Example
[1., 0., 0.] only the optical compressor is acting
[0., 1., 0.] only the tape recorder is acting
[0., 0., 1.] only the pre-amp is acting


# VST Download

Coming soon...