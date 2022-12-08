## Files in the folder

~~~~
BGCL/
├── datasets/
│   └── wsdrean/
│   	├── cold: sample data for cold start 
│       │   ├── rt: data of response time
│       │   └── tp: data of response throughput
│       └── sparsity: sample data for sparsity    
│           ├── rt: data of response time
│           └── tp: data of response throughput
├── utils/
│   ├── attention.py: aggregating the feature of neighbors
│   ├── batchutils.py: some utils
│   ├── combiner.py: implementation of the combiner
│   └── encoder.py: together with aggregator to form the decomposer
│
├── bgcl.py: implementation of the bgcl
├── run.py: run the model
├── train.py: train the model
├── test.py: test the model
└── README.md
~~~~

## Environment Settings

* Python == 3.6.9
* torchvision == 0.4.2
* numpy == 1.17.3
* scikit-learn == 0.21.3

## Parameter Settings

- epochs: the number of epochs to train
- test_epochs: the number of epochs to test
- lr: learning rate
- optimizer: the optimizer
- momentum: the momentum of optimizer
- lr: learning rate
- embed_dim: embedding dimension
- droprate: dropout rate in layer
- temp: the temp
- batch_size: batch size for training
- test_batch_size: batch size for testing
- n_size: the size of neighbor
- dataset: the prefix of dataset
- density: the density of dataset
- cold_start_density: the density of cold_start
- p: the dropout rate
- a: the similiarity setting
- mode: the mode of experiment (sparsity or cold)
- coldStartType: the cold start type
- cold_start_density: the density of cold_start

## Basic Usage

~~~
python run.py 
~~~


## Example of running

unzip the dataset first 

example: unzip the rt_user_allData_0.8_0.7_0.500_0.005.7z to get rt_user_allData_0.8_0.7_0.500_0.005.p for running


~~~
Dataset: wsdream
Density: 0.005
-------------------- Hyperparams --------------------
dropout rate: 0.5
dimension of embedding: 128
type of optimizer: Adam
learning rate: 0.001
p: 0.8
a: 0.7
datatype: rt
mode: cold
coldStartType: user
cold_start_density: 0.500

...

2022-12-06 21:10:51.528380 Training: [200 epoch,  50 batch] loss: 4.67270, the best RMSE/MAE: 1.85315 / 0.71128
2022-12-06 21:10:52.077927 Training: [200 epoch,  60 batch] loss: 4.13545, the best RMSE/MAE: 1.85315 / 0.71128
2022-12-06 21:10:52.606055 Training: [200 epoch,  70 batch] loss: 4.26554, the best RMSE/MAE: 1.85315 / 0.71128
<Test> RMSE: 1.86849, MAE: 0.71469 
The best RMSE/MAE: 1.85315 / 0.71128
~~~

