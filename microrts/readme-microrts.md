
# MicroRTS-Py

## Installation

Experiments require MicroRTS-Py.
Follow the instructions in the [MicroRTS-Py repo](https://github.com/Farama-Foundation/MicroRTS-Py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
pip install -r requirements.txt
```

## Generating datasets

Datasets are stored in the `data` directory.
Run the following script in order to generate the datasets and save them in our format:

```
python generate_random_dataset.py -n 100
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env microrts --dataset random --model_type dt
```

Adding `-w True` will log results to Weights and Biases.
