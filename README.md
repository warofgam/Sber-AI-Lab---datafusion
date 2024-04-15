![](vtb_pic.jpeg)

# Data Fusion Contest 2024. 2-st place on the Churn Task.

[Link to competition](https://ods.ai/competitions/data-fusion2024-churn)

Our solution acheived **2-st** place on the public and private leaderboard.

# Overview

## Validation

We used stacking pipenile for our model validation. At the start, we choose a test sample that will be used for validation. Next, we get predictions on 5 folds and train a meta-model on them. Logistic regression is used as a meta-model. Then, the models are trained on all the data for the final prediction.

## Models

This is neural network based solution. 8 out of 10 models in the ensemble use embeddings to obtain predictions. We use [pytorch-lifestream](https://github.com/dllllb/pytorch-lifestream) library to simplify work with sequences data.
The main models used in the solution are: CoLES, WTTE-RNN, CoLES on a uniform grid, Supervised NN.

# How to work with solution

## Install and use environment

```
pipenv sync --dev

pipenv shell
```

## Run
```
# Load data
sh get_data.sh

```

To get embeddings and models for finetuning run notebooks:
nn_train/coles-train_ensemble.ipynb
nn_train/wtte-coles_ensemble.ipynb
nn_train/wtte-rnn.ipynb

To get predicts for metamodel run notebooks:
nn_train/supervised-coles.ipynb
nn_train/supervised-wtte-coles.ipynb
and all notebooks at 2_lvl_train folder.

To get final predict run metamodel.ipynb



