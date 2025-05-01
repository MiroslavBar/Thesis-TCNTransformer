# EEG Classification Experiments

This repository contains code for training and evaluating a TCN-Transformer hybrid model for EEG signal classification. The model is designed to classify different types of motor execution (ME) and motor imagery (MI) tasks from EEG data.

## Prerequisites

### Installation
This project uses Python 3.10

The repository includes a `requirements.txt` file with all necessary dependencies. To install:

```bash
pip install -r requirements.txt
```

## Datasets

The project contains three packages each of which contains script for training and testing using a respective dataset. In each package there is a .yaml configuration file. Description of configurable parameters is in the .yaml file.

## data availability
- Shuqfa-103: https://data.mendeley.com/datasets/dpmtgrn8d8/4
- Brunner-9: https://www.bbci.de/competition/iv/
- Kodera-29: https://zenodo.org/records/7893847


## Running the Experiments

To run the experiment with the default configuration:

```bash
python shuqfa103_train.py
```

```bash
python kodera29_train.py
```

```bash
python brunner9_train.py
```

To specify a different configuration file:

```bash
python [dataset]_train.py --config your_custom_config.yaml
```

## Model Architecture

The architecture is a hybrid of:
- **Temporal Convolutional Network (TCN)**: Captures temporal patterns in the EEG signals
- **Transformer Encoder**: Learns attention-based relationships between time points

## Output Files

The following files will be generated after training:
- Trained model file
- Plot of training and validation metrics
- Confusion matrix on the test set


## Notes

- GPU acceleration is enabled by default and can be configured in the YAML file
- Mixed precision training is enabled by default for better performance on compatible GPUs
- The code automatically handles balanced sampling for rest vs. movement classes

