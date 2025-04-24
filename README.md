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

## Comparison test

The results of the comparison experiments in Table 1 are available in the Comparison folder. The baseline methods have been named according to the dataset. The corresponding folder can be opened to find the corresponding baseline model code. For example, the Comparison/Generation/generation-LEGNNP-RNP folder is the model code for LEGNNP-RNP on the synthetic dataset.

The experimental results in Table 2 can be obtained by running the code SAM. where Set dataset=‘generation’, model_name=[‘SAM_GCN’, ‘SAM_GAT’, ‘SAM_GraphSage’], batch_size=128, epochs=128, lr=0.1, lamba=0.1, emb_dim= 128, residual_training=[True,False], MMD=[True,False], , k=5. that is. The response prediction results can be found in the result_out.csv file under the corresponding traind folder.
## Flexibility of the Model

The experimental results in Table 2 can be obtained by running the code SAM. where Set dataset=‘generation’, model_name=[‘SAM_GCN’, ‘SAM_GAT’, ‘SAM_GraphSage’], batch_size=128, epochs=128, lr=0.1, lamba=0.1, emb_dim= 128, residual_training=[True,False], MMD=[True,False],  k=5. that is. The response prediction results can be found in the result_out.csv file under the corresponding traind folder.

## Self-adaptive Loss Weights Studies
The experimental results in Table 3 can be obtained by running the SAM. where Set dataset=‘generation’, model_name=[‘SAM_GCN’, batch_size=128, epochs=128, lr=0.1, lamba=0.1, emb_dim= 128, residual_training=True, MMD=True,  k=5 lamba=0.1.   Running SAM_LAlpha0.1, SAM_Alpha0.5, SAM_Alpha1, the comparison results are obtained by setting lamba=[-1,-0.5,-0.1] in the hyperparameter selection. 


## Visualisation
The Figure folder contains all the data and code used to draw the graph. Where generation data.ipynb is used for the generation and visualisation code for the synthetic dataset and Figure(1).ipynb is used to draw all other figures. The data used to draw the plots is also contained in this folder.

If you need to generate raw data, you can run the appropriate code to do so.
For example, if you need to generate the raw data for Figure 4(a) in the manuscript, run the LEGNN-RNP.py file in the generation_LEGNN-RNP folder. There is no need to adjust the hyperparameters here, and the prediction_0.csv can be obtained for plotting in the resulting generation/generation_LEGNN-RNP/trained file at the end of the code run.
If you need to generate raw data for Figure 4(b) in the manuscript, run train.py in the SAM folder. Set dataset='generation', model_name='SAM_GCN', batch_size=128, epochs=128, lr=0.1, lamba=0.1, emb_dim=128, residual_training=True, MMD=True , k=5. After running the model, please get the file test_out0. in traind/generation-SAM-SAM_GCN-MAE-size0.1-k5-emb64-lr0.1-bs32-ep500-nor0-xuhao10_2MMD0.1_Apr/result.csv for plotting.
