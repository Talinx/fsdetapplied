Applying FsDet to novel classes from a different distribution
=============================================================

This project applies [FsDet](https://github.com/ucbdrive/few-shot-object-detection) to a data set with images from a different distribution than the pre-training data set using [PyTorch](https://pytorch.org/).


To use the source code, first install the required libraries:
```bash
pip install -r requirements.txt
```

## Adding the data sets

Before training and inference, the data sets have to be added.

Create a directory called `datasets` and insert the data sets into it, the directory structure should be as follows:
```
datasets
├── FoodQS
│   ├── Kastanie
│   └── Raps
└── pollen
    ├── anadenanthera
    ├── arecaceae
    ├── arrabidaea
    ├── cecropia
    ├── chromolaena
    ├── combretum
    ├── croton
    ├── dipteryx
    ├── eucalipto
    ├── faramea
    ├── hyptis
    ├── mabea
    ├── matayba
    ├── mimosa
    ├── myrcia
    ├── protium
    ├── qualea
    ├── schinus
    ├── senegalia
    ├── serjania
    ├── syagrus
    ├── tridax
    └── urochloa
```

`datasets/FoodQS/Kastanie` and `datasets/FoodQS/Raps` are expected to contain images and xml files for image meta data (bounding boxes, labels).

The directories under `datasets/pollen` should only contain images.

## Training

You can train a model using the following command:
```bash
python train.py
```

This will use the hyperparameters from `constants.py` to train a model. If you want more control over the training process or train multiple models one after the other, create a training schedule toml file and specify it when running `train.py`:

```bash
python train.py -c "training schedule.toml"
```

`training schedule.toml` contains the training schedule used for this work.

You can specify:

- the `k` to use
- the number of epochs
- (for non-FsDet training) the interval between training with different loss functions
- (for non-FsDet training) the epoch from which onward only the last layers are trained
- the dataset to use, either `FoodQS` or `Pollen`
- whether to start from a pre-trained model
- whether to use FsDet fine-tuning or train the whole model from scratch
- the model type, one of `fasterrcnn`, `mobilenet` or `mobilenet320`

Training creates the directories `logs`, `losses` and `models`.

### Plotting losses

Once training is completed, losses can be plotted for all models:

```bash
python plot.py
```

Or only the latest model:
```bash
python plot.py -t
```

## Inference

Use the following command for inference:

```bash
python inference.py
```

Creates a `labeled images` directory with sub-directories for labeled images from each model.

This will perform inference of all models. Each model will be used on the same data set as has been used for training.

You can use the following options for the inference script:

- `-l` or `--labels`: do not perform regular inference, instead label the images from the datasets as they would be used for training
- `-s` or `--stats`: calculate mAP stats during inference
- `-t` or `--threshold`: threshold for bounding boxes (only draw above a certain score), default is `0.9`, use `-t 0.0` to draw all bounding boxes
- `-c` or `--count`: count how many images have at least one bounding box
