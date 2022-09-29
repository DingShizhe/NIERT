## Code of NIERT

_NIERT: Accurate Numerical Interpolation through Unifying Scattered Data Representations using Transformer Encoder_

The implementation is based on [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales) and [TFR-HSS-Benchmark](https://github.com/shendu-sw/TFR-HSS-Benchmark).


## Preparation

1. We recommend using `conda` to create the environment:

```bash
conda create -n niert python=3.7
conda activate niert
```

2. Install third-party libraries:

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Note that we used **Weights & Biases** for tracking and visualizing metrics such as loss and accuracy, which needs a little setup.

## NeSymReS dataset construction

Note that the construction of NeSymReS dataset is based on the work of [NeuralSymbolicRegressionThatScales](https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales).

The construction configuration file of NeSymReS is `nesymres_dataset_configuration.json`. The dimension of variable $x$ of the generated fucntion is 2D by default. Changing the dimension requires to change `variables` in the configuration file.

Accordingly, `cfg_dim_input` in the training config file `./config/config_NeSymReS.yml` need to be changed (variable's dimension+1).

```bash
# generate training equations set
python3 -m src.data.nesymres.run_dataset_creation --number_of_equations 1000000 --no-debug

# generate testing equations set
python3 -m src.data.nesymres.run_dataset_creation --number_of_equations 150 --no-debug

mkdir -p nesymres_data/test_set

# convert the newly created validation dataset in a csv format
python3 -m src.data.nesymres.run_dataload_format_to_csv raw_test_path=nesymres_data/data/raw_datasets/150

# remove the validation equations from the training set
python3 -m src.data.nesymres.run_filter_from_already_existing --data_path nesymres_data/data/raw_datasets/1000000 --csv_path nesymres_data/test_set/test_nc.csv

python3 -m src.data.nesymres.run_apply_filtering --data_path nesymres_data/data/raw_datasets/1000000
```


## Train NIERT on NeSymReS dataset

To train NIERT on NeSymReS dataset, run

```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_NeSymReS.yml
```

## Train NIERT on D30 dataset

To train NIERT on D30 dataset, run

```bash
CUDA_VISIBLE_DEVICES="0" python main.py --config_path ./config/config_D30.yml
```


## TFRD-ADlet dataset and PhysioNet dataset

TFRD-ADlet dataset is downloadable at [here](https://pan.baidu.com/s/14BipTer1fkilbRjrQNbKiQ). Note that the password is `tfrd`.

PhysioNet dataset is downloadable at [here](https://physionet.org/content/challenge-2012/1.0.0/).

To train NIERT on TFRD-ADlet dataset or PhysioNet dataset, just run

```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_TFR.yml    # or config_PhysioNet.yml
```

Before training, the `data_root` in `config_TFR.yml` (or config_PhysioNet.yml) need be set as the path of downloaded TFRD-ADlet or PhysioNet.


## Pre-training and fine-tuning

Take TFR as an example. After pre-train NIERT on 2D NeSymReS dataset we will get the pre-trained model. Then we set `resume_from_checkpoint` in `./config/config_PhysioNet_Finetune.yml` as the path of the pre-trained model. Then run 

```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py --config_path ./config/config_TFR.yml    # or config_PhysioNet_Finetune.yml
```
for fine-tuning.



## Testing

For NeSymReS dataset, we certainly need to fix a interpolation task test set from the equation skeleton test set.

```bash
python main.py -m save_nesymres_testdataset_as_file
```

Then we can evaluate NIERT on such test set.

```bash
CUDA_VISIBLE_DEVICES="0,1" python main.py -m test_nesymres --resume_from_checkpoint path_of_niert_checkpoint
```
