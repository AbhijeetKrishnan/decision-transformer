
# Karel

## Installation

Clone the repository and the [LEAPS](https://github.com/clvrai/leaps) submodule,
and install required dependencies. Recommended to use [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to create the environment instead of [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for speed.

```bash
git clone --recurse-submodules https://github.com/AbhijeetKrishnan/decision-transformer
cd karel
conda env create
conda activate dt-karel
```

## Running experiments

Generation of datasets and training with hyperparameters reported in the paper can be done with the command:

```bash
./run_exps.sh
```
