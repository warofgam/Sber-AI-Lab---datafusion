![](vtb_pic.jpeg)

# Data Fusion Contest 2024. 2-st place on the Churn Task.

[Link to competition](https://ods.ai/competitions/data-fusion2024-churn)

Our solution acheived **2-st** place on the public and private leaderboard.

# Overview

## Validation

We used stacking pipenile for our model validation. At the start, we choose a test sample that will be used for validation. Next, we get predictions on 5 folds and train a meta-model on them. Logistic regression is used as a meta-model. Then, the models are trained on all the data for the final prediction.

## Models

