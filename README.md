# IFAE_GNNtools

Tools to implement graph neural networks in HEP analyses for example in classification and reconstruction problems. GNN framework is built with pytorch. 

# Requirements

* numpy
* pandas
* scikit-learn
* networkx
* pytorch
* pytorch-geometric

Install/use our centralised kernel
====
1. Setup conda
    
    Conda is the friendly python environment that takes care of installing properly the compatible packages for you.
    
    `source /nfs/pic.es/user/s/salvador/conda_setup.sh`

2. Activate the centralised environment (you can deactivate it with `source deactivate`)

    While activated, you have all the packages needed to work! Avoid installing more packages, to do that check the next section.
    
    `conda activate /data/at3/scratch/salvador/condaenv_GNN`
    
3. Install kernel with the setup environment

    This has to be done ONCE.
    
    `python -m ipykernel install --user --name=IFAE_GNN` (or any other name)
    
    You can check which kernels you have installed with `jupyter kernelspec list`


Create a kernel from scratch
====

1. Setup conda
    
    Conda is the friendly python environment that takes care of installing properly the compatible packages for you.
    
    `source /nfs/pic.es/user/s/salvador/conda_setup.sh`

2. Create local conda environment
    
    `conda create --prefix env_NN` (`--prefix` is to create it locally, if you use `--name` the environment be eliminated at the end of the session)

3. Activate it (you can deactivate it with `source deactivate`)

    While activated, you can install packages locally and recover the setup activating the environment again!
    
    `conda activate env_GNN`

3. Install packages in environment

    Packages needed are tensorflow-gpu to run the gpu, pandas and tables to open the input, scikit-learn for BDTs and other tools, matplotlib for plotting tools, ipykernel to install the kernel and feather to save and load pandas efficiently.
    
    This can be tricky as the conda version may change and the compatibilities are well set. It has happened that installing new version of tensorflow-gpu with conda does not include the necessary cuda packages to work with GPU's
        
    `pip install ipykernel pandas tables joblib scikit-learn matplotlib feather-format pickle5`

    `conda install root` (requires confirmation)
   
    `pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

    `pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

    `pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

    `pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu102.html`

    `pip install torch-geometric`
    
    If another version of torch is used, replace `1.8.1+cu102` by the `print(torch.__version__)` output

4. Install kernel with the created environment

    This has to be done ONCE.
    
    `python -m ipykernel install --user --name=kernel_GNN` (or any other name)
    
    You can check which kernels you have installed with `jupyter kernelspec list`

5. Refresh the browser and make sure you see the new kernel option inside the jupyter nootebook or when creating a new one

_if there is anything wrong with this setup tell me_ 

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

