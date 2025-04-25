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

The experimental results of SAM presented in Table 1 can be reproduced by running the SAM folder with the following configuration:
DATA = ['generation', 'cali', 'GData', 'Near-surface'], model_name='SAM_GCN', batch_size=128, epochs=500, lr=0.1, lambda=0.1, emb_dim=128, residual_training='True', MMD='True', and k=5.
The response prediction results are saved in the result_out.csv file, located in the corresponding traind folder after training.


The results of the comparative experiments presented in Table 1 can be found in the Comparison folder. Each baseline method is organized and named according to the corresponding dataset. To locate the code for a specific baseline model, navigate to the appropriate subfolder. For example, the code for the LEGNNP-RNP model applied to the synthetic dataset is located in the Comparison/Generation/generation-LEGNNP-RNP directory.

The experimental results presented in Table 2 can be reproduced by running the SAM code with the following configuration:
dataset='generation', model_name=['SAM_GCN', 'SAM_GAT', 'SAM_GraphSage'], batch_size=128, epochs=500, lr=0.1, lambda=0.1, emb_dim=128, residual_training=[True, False], MMD=[True, False], and k=5.
The response prediction results are saved in the result_out.csv file, located in the corresponding traind folder after training.

## Flexibility of the Model

The experimental results shown in Table 2 can be reproduced by running the SAM code with the following configuration:
dataset='generation', model_name=['SAM_GCN', 'SAM_GAT', 'SAM_GraphSage'], batch_size=128, epochs=128, lr=0.1, lambda=0.1, emb_dim=128, residual_training=[True, False], MMD=[True, False], and k=5.
The response prediction results will be saved in the result_out.csv file, located in the corresponding traind folder after training.

## Self-adaptive Loss Weights Studies
The experimental results presented in Table 3 can be obtained by running the SAM model with the following settings:

dataset='generation', model_name='SAM_GCN', batch_size=128, epochs=128, lr=0.1, lambda=0.1, emb_dim=128, residual_training=True, MMD=True, and k=5.

To evaluate the effect of the hyperparameter lambda and alpha, experiments were conducted in the SAM_Alpha folder by running. The comparison results in Table 3 correspond to setting alpha=[0.1, 0.5, 1],lambda=[-1, -0.5, -0.1] in the hyperparameter selection process.

## Ablation Study
The results shown in Figure 10 can be obtained by running the corresponding code in the Ablation Study folder. Each folder is named according to the respective method. To reproduce the results, simply set dataset=['cali', 'generation'] and execute the code directly.

The results in Figure 11 can be reproduced by setting the hyperparameters in the SAM folder as follows:
    DATA = 'generation'
    lr_list = [0.1, 0.005, 0.001]
    TRAIN_SIZE = [0.1, 0.2, 0.3]
    BATCHS = [32, 64, 128, 256]
    embedding = [32, 64, 128, 256]

## Visualisation
The Figure folder contains all the data and code required for plotting the figures in the manuscript. Specifically:

All necessary datasets for plotting are also provided in this folder.
Figure code.ipynb include all code for plotting the figures

If you need to regenerate raw data for specific figures, you can execute the corresponding scripts:

To generate the raw data for Figure 4(a):
Run the LEGNN-RNP.py script in the generation_LEGNN-RNP folder.
No hyperparameter adjustment is needed.
Upon completion, the predicted results will be saved in:
generation/generation_LEGNN-RNP/trained/prediction_0.csv

To generate the raw data for Figure 4(b):
Run the train.py script in the SAM folder with the following settings:
dataset = 'generation'
model_name = 'SAM_GCN'
batch_size = 128
epochs = 128
lr = 0.1
lamba = 0.1
emb_dim = 128
residual_training = True
MMD = True
k = 5

After training, the predicted results will be located at:
traind/generation-SAM-SAM_GCN-MAE-size0.1-k5-emb64-lr0.1-bs32-ep500-nor0-xuhao10_2MMD0.1_Apr/result.csv

## Notes
When reproducing the methodology, it should be noted that the experimental results may fluctuate due to a variety of random factors, including but not limited to the random way of dividing the dataset, differences in the random seed settings, differences in hardware configurations, and changes in software versions. It is normal that these uncontrollable variables may lead to slight deviations of the reproduced results from the original manuscript records. To enhance the credibility of the experimental results, we enhanced the stability of the conclusions by repeating the experiment 10 times independently and counting the mean ± standard deviation of the final metrics (e.g., 75.3 ± 0.8).
