
# Karel

## Installation

Clone the repository and [LEAPS](https://github.com/clvrai/leaps) submodule, and install required dependencies.

```bash
git clone --recurse-submodules https://github.com/AbhijeetKrishnan/decision-transformer
cd karel
conda env create -f conda_env.yml
pip install -r requirements.txt
conda activate decision-transformer-karel
```

## Generating datasets

Datasets are stored in the `data` directory.
Run the following script in order to generate the datasets and save them in our format:

```
python generate_random_dataset.py -g microrts -n 100
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment.py --env microrts --dataset random --model_type dt
```

Adding `-w True` will log results to Weights and Biases.
