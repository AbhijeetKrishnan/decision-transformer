
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

## Running experiments

Generation of datasets and training with hyperparameters reported in the paper can be done with the command:

```bash
./run_exps.sh
```
