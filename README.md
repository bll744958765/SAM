# SAM
This is a pytorch implementation of SAM: A Structure-Attribute Match Method for Geographic Data Prediction
## Requirements
torch
torch.nn
numpy
datetime
os
pandas
sklearn
random

## Running examples
You can adjust the parameters:

datafile: sampled dataset in folder "Data/dataset"

batch_size: batch size

lr: learning rate

emb_dim: [input dimension, model dimension,]

k: k nearest neighbors

loss: loss function type

if_save_model: if save the best model or not

epochs:number of training epochs

lamba:trade-off parameter for MAE and MMD loss

MMD: Maximum Mean Discrepancy

residual_training：residual connection

seed:random_state

min_val:standardization
