# IFAE_GNNtools

Tools to implement graph neural networks in HEP analyses for example in classification and reconstruction problems. GNN framework is built with pytorch. 

# Requirements

* numpy
* pandas
* scikit-learn
* networkx
* pytorch
* pytorch-geometric

## Installing pytorch-geometric.

* First install pytorch:

`!pip install torch`

* Check version (I used version 1.8.1+cu102):

`import torch; print(torch.__version__)`

Example output: 

`1.8.1+cu102` 

* Install corresponding pytorch-geometric libraries (here replace `1.8.1+cu102` with output of `print(torch.__version__)`):


`!pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

`!pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

`!pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

`!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

`!pip install torch-geometric`


# Creating graphs

`gnn_tools.data` contains functions and classes create graphs from HEP events.

In training events are input with pandas dataframes. Use root_numpy to convert ntuples to dataframe and save as pickle.

`event2networkx` creates a Networkx graph for a single event. Event is read into function as a pandas dataframe row.

`CreateTorchGraphs` takes in a pandas Dataframe of HEP events and converts to list of graphs objects which are PyTorch_Geometric `Data` objects.

`customDataset` custom PyTorch dataset class, contains methods for saving and loading graphs to/from disk.

# Define model

Model class is defined in `gnn_tools.model`. To do: config parsing for model setup. Currently just hardcoded.

# Training

Function `runTraining` in `gnn_tools.train` takes care of training steps.

