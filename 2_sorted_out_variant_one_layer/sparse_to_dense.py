"""
Script to more formally obtain training loss vs. step and test accuracy vs. step plots with different 
degrees of sparcity in the training dataset. 
"""

# Imports
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace 
import numpy as np

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # chooses the device for computation
m = 50 # input dimension
n = 100 # measurement dimension
seed = 1
network.set_random_mode(True, seed)

init_parameters = SimpleNamespace()
init_parameters.m = m 
init_parameters.n = n 
init_parameters.seed = seed
init_parameters.W = torch.randn(n,m).to(device)
store_image = True

# Save parameters
import pickle
import os

weight_folder = f'weights/m_{m}_n_{n}_s_{seed}'
os.makedirs(weight_folder, exist_ok=True)

param_path = f'{weight_folder}/init_params.pkl'
with open(param_path, 'wb') as f:
    pickle.dump(init_parameters, f)

# Creates the network and establishes parameters
enose_network = network.enose(init_parameters)
num_data=10000
sparse_training_frac=0.8
dense_training_frac=0.2
is_balanced=True
threshold=0.1
x_hat = threshold
num_training = int(num_data*sparse_training_frac)
bounds = (0,1)

list_of_sparsities = np.arange(1,49).tolist()

# Iterate over different sparsities
for sparsity_index in list_of_sparsities:

    # Creates sparse dataset
    sparse_dataset = network.generate_dataset(m, n, device, num_data, sparse_training_frac, is_balanced, threshold, bounds, sparsity_index)
    sparse_train_conc = sparse_dataset.train_conc
    sparse_train_labels = sparse_dataset.train_labels

    # Creates dense dataset
    dense_dataset = network.generate_dataset(m, n, device, num_data, dense_training_frac, is_balanced, threshold, bounds, None)
    dense_test_conc = dense_dataset.test_conc
    dense_test_labels = dense_dataset.test_labels
    test_batch_size = 50

    # Train the network







